import os
import argparse
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer

def calculate_confidence_score(p10, p50, p90, step_idx):
    """
    Calculate a realistic forecast confidence score that decays over time.
    """
    if p50 == 0:
        return 0.50
    # Interval width relative to point estimate
    relative_spread = (p90 - p10) / p50
    # Base confidence from interval spread
    base_conf = 1.0 - min(0.45, relative_spread * 0.15)
    # Time decay factor (confidence naturally drops further out)
    decay = 0.99 ** step_idx
    score = base_conf * decay
    return float(np.clip(score, 0.55, 0.96))

def get_confidence_label(score):
    if score >= 0.85:
        return f"High ({int(score*100)}%)"
    elif score >= 0.70:
        return f"Medium ({int(score*100)}%)"
    else:
        return f"Low ({int(score*100)}%)"

def main():
    parser = argparse.ArgumentParser(description="Predict E-commerce Revenue and ROAS")
    parser.add_argument("data_dir", type=str, help="Directory containing input data CSVs")
    parser.add_argument("model_path", type=str, help="Path to the packaged model pickle file")
    parser.add_argument("output_path", type=str, help="Path to save predictions.csv")
    args = parser.parse_args()
    
    # 1. Load models
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    with open(args.model_path, "rb") as f:
        model_pack = pickle.load(f)
        
    revenue_model = model_pack["revenue_model"]
    roas_model = model_pack["roas_model"]
    feature_cols = model_pack["feature_cols"]
    
    # 2. Ingest and aggregate historical data
    pipeline = DataPipeline(args.data_dir)
    df_historical = pipeline.aggregate_sources()
    
    # Sort and clean
    df_historical["Date"] = pd.to_datetime(df_historical["Date"])
    df_historical = df_historical.sort_values("Date").reset_index(drop=True)
    
    # 3. Create Feature Engineer and process historical data
    fe = FeatureEngineer()
    df_all = fe.generate_features(df_historical, is_training=False)
    
    # Calculate historical averages for future features
    last_30_days = df_historical.tail(30)
    avg_daily_spend = last_30_days["Total_Spend"].mean()
    
    # Historical ratios
    total_spend_30 = last_30_days["Total_Spend"].sum()
    avg_cpc = last_30_days["Total_Spend"].sum() / last_30_days["Total_Clicks"].sum() if last_30_days["Total_Clicks"].sum() > 0 else 1.20
    avg_ctr = last_30_days["Total_Clicks"].sum() / last_30_days["Total_Impressions"].sum() if last_30_days["Total_Impressions"].sum() > 0 else 0.025
    avg_cr = last_30_days["Total_Conversions"].sum() / last_30_days["Total_Clicks"].sum() if last_30_days["Total_Clicks"].sum() > 0 else 0.020
    
    # Spend channel shares
    google_share = last_30_days["Google_Spend"].sum() / total_spend_30 if total_spend_30 > 0 else 0.4
    meta_share = last_30_days["Meta_Spend"].sum() / total_spend_30 if total_spend_30 > 0 else 0.4
    ms_share = last_30_days["MS_Spend"].sum() / total_spend_30 if total_spend_30 > 0 else 0.2
    
    # 4. Recursive Forecasting Loop for 90 Days
    last_date = df_all["Date"].max()
    forecast_days = 90
    
    predictions_list = []
    
    for step in range(1, forecast_days + 1):
        future_date = last_date + timedelta(days=step)
        
        # Start a new row of data
        new_row = {"Date": future_date}
        
        # Spend for future: assume rolling 30-day average
        new_row["Total_Spend"] = avg_daily_spend
        new_row["Google_Spend"] = avg_daily_spend * google_share
        new_row["Meta_Spend"] = avg_daily_spend * meta_share
        new_row["MS_Spend"] = avg_daily_spend * ms_share
        
        new_row["Total_Clicks"] = int(new_row["Total_Spend"] / avg_cpc)
        new_row["Total_Impressions"] = int(new_row["Total_Clicks"] / avg_ctr)
        new_row["Total_Conversions"] = int(new_row["Total_Clicks"] / avg_cr)
        
        # Append temporary row to aggregate calculations
        temp_df = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True)
        
        # Re-engineer features to correctly populate lags and rolling windows
        temp_df = fe.generate_features(temp_df, is_training=False)
        
        # Get the row we just added (the last row)
        X_row = temp_df[feature_cols].tail(1)
        
        # Predict Revenue
        rev_p50 = revenue_model.predict_point(X_row, future_date)
        rev_p10, rev_p90 = revenue_model.predict_quantiles(X_row, rev_p50)
        
        # Predict ROAS
        roas_p50 = roas_model.predict_point(X_row, future_date)
        roas_p10, roas_p90 = roas_model.predict_quantiles(X_row, roas_p50)
        
        # Write predictions back to temp_df to feed the next recursive steps
        temp_df.loc[temp_df.index[-1], "Revenue"] = rev_p50
        temp_df.loc[temp_df.index[-1], "ROAS"] = roas_p50
        
        # Calculate confidence
        conf_score = calculate_confidence_score(rev_p10, rev_p50, rev_p90, step)
        conf_label = get_confidence_label(conf_score)
        
        # Save prediction
        predictions_list.append({
            "Forecast_Period": f"Day {step}",
            "Date": future_date.strftime("%Y-%m-%d"),
            "Revenue_P10": round(rev_p10, 2),
            "Revenue_P50": round(rev_p50, 2),
            "Revenue_P90": round(rev_p90, 2),
            "ROAS_P10": round(roas_p10, 2),
            "ROAS_P50": round(roas_p50, 2),
            "ROAS_P90": round(roas_p90, 2),
            "Forecast_Confidence": conf_label
        })
        
        # Update df_all to include this new day's prediction for next iterations
        df_all = temp_df.copy()
        
    df_predictions = pd.DataFrame(predictions_list)
    
    # 5. Save output predictions.csv (with exactly the 7 required columns)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_output = df_predictions[[
        "Forecast_Period",
        "Revenue_P10",
        "Revenue_P50",
        "Revenue_P90",
        "ROAS_P10",
        "ROAS_P50",
        "ROAS_P90"
    ]]
    df_output.to_csv(args.output_path, index=False)
    
    print(f"Predictions successfully written to '{args.output_path}'")
    print(df_output.head(5))

if __name__ == "__main__":
    main()
