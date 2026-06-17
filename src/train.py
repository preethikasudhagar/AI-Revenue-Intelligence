import os
import argparse
import pickle
import pandas as pd
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from models import RevenueForecastingEnsemble

def main():
    parser = argparse.ArgumentParser(description="Train AI Revenue Intelligence Ensemble Models")
    parser.add_argument("--data", type=str, default="data", help="Directory containing raw data CSVs")
    parser.add_argument("--pickle-dir", type=str, default="pickle", help="Directory to save serialized models")
    args = parser.parse_args()
    
    # 1. Ingest and aggregate data
    print("Step 1: Starting Data Ingestion and Validation...")
    pipeline = DataPipeline(args.data)
    df_merged = pipeline.aggregate_sources()
    
    # 2. Engineer features
    print("Step 2: Starting Feature Engineering...")
    fe = FeatureEngineer()
    df_features = fe.generate_features(df_merged, is_training=True)
    
    # Define features for training
    feature_cols = [
        "day_of_week", "day_of_month", "month", "quarter", "is_weekend",
        "sin_day_of_year", "cos_day_of_year",
        "Total_Spend", "blended_cpc", "blended_ctr", "blended_conv_rate",
        "google_spend_share", "meta_spend_share", "ms_spend_share",
        "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_30",
        "spend_lag_1", "spend_lag_7", "roas_lag_1", "roas_lag_7",
        "revenue_roll_mean_7", "revenue_roll_mean_30",
        "revenue_roll_std_7", "revenue_roll_std_30",
        "spend_roll_mean_7", "spend_roll_mean_30",
        "revenue_growth_7d", "revenue_growth_30d"
    ]
    
    # 3. Train models
    print("Step 3: Training Revenue Forecasting Ensemble...")
    revenue_ensemble = RevenueForecastingEnsemble()
    revenue_ensemble.train(df_features, target_name="Revenue", feature_cols=feature_cols)
    
    print("Step 4: Training ROAS Forecasting Ensemble...")
    roas_ensemble = RevenueForecastingEnsemble()
    roas_ensemble.train(df_features, target_name="ROAS", feature_cols=feature_cols)
    
    # 4. Package models together
    print("Step 5: Packaging and Saving Models...")
    os.makedirs(args.pickle_dir, exist_ok=True)
    model_path = os.path.join(args.pickle_dir, "model.pkl")
    
    model_pack = {
        "revenue_model": revenue_ensemble,
        "roas_model": roas_ensemble,
        "feature_cols": feature_cols
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_pack, f)
        
    print(f"Success! Model pack saved to '{model_path}'")

if __name__ == "__main__":
    main()
