import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def generate_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Generate time-based, marketing efficiency, lag, and rolling features.
        """
        df = df.copy()
        
        # Ensure Date is datetime type and sort
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # 1. TIME FEATURES
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["day_of_month"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        df["year"] = df["Date"].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Cyclic time features (sin/cos encoding of day of year for seasonality)
        day_of_year = df["Date"].dt.dayofyear
        df["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # 2. MARKETING EFFICIENCY FEATURES
        df["blended_cpc"] = np.where(df["Total_Clicks"] > 0, df["Total_Spend"] / df["Total_Clicks"], 0.0)
        df["blended_ctr"] = np.where(df["Total_Impressions"] > 0, df["Total_Clicks"] / df["Total_Impressions"], 0.0)
        df["blended_conv_rate"] = np.where(df["Total_Clicks"] > 0, df["Total_Conversions"] / df["Total_Clicks"], 0.0)
        
        # Channel-level share features
        df["google_spend_share"] = np.where(df["Total_Spend"] > 0, df["Google_Spend"] / df["Total_Spend"], 0.0)
        df["meta_spend_share"] = np.where(df["Total_Spend"] > 0, df["Meta_Spend"] / df["Total_Spend"], 0.0)
        df["ms_spend_share"] = np.where(df["Total_Spend"] > 0, df["MS_Spend"] / df["Total_Spend"], 0.0)
        
        # 3. LAG FEATURES (for ML models like XGBoost and LightGBM)
        # Lagged revenue
        df["revenue_lag_1"] = df["Revenue"].shift(1)
        df["revenue_lag_7"] = df["Revenue"].shift(7)
        df["revenue_lag_14"] = df["Revenue"].shift(14)
        df["revenue_lag_30"] = df["Revenue"].shift(30)
        
        # Lagged marketing variables
        df["spend_lag_1"] = df["Total_Spend"].shift(1)
        df["spend_lag_7"] = df["Total_Spend"].shift(7)
        df["roas_lag_1"] = df["ROAS"].shift(1)
        df["roas_lag_7"] = df["ROAS"].shift(7)
        
        # 4. ROLLING STATISTICS (Rolling averages, std, growth trends)
        # Revenue rolling statistics
        df["revenue_roll_mean_7"] = df["Revenue"].rolling(window=7, min_periods=1).mean()
        df["revenue_roll_mean_30"] = df["Revenue"].rolling(window=30, min_periods=1).mean()
        df["revenue_roll_std_7"] = df["Revenue"].rolling(window=7, min_periods=1).std().fillna(0.0)
        df["revenue_roll_std_30"] = df["Revenue"].rolling(window=30, min_periods=1).std().fillna(0.0)
        
        # Spend rolling statistics
        df["spend_roll_mean_7"] = df["Total_Spend"].rolling(window=7, min_periods=1).mean()
        df["spend_roll_mean_30"] = df["Total_Spend"].rolling(window=30, min_periods=1).mean()
        
        # 5. REVENUE GROWTH TRENDS
        df["revenue_growth_7d"] = (df["Revenue"] - df["revenue_lag_7"]) / (df["revenue_lag_7"] + 1.0)
        df["revenue_growth_30d"] = (df["Revenue"] - df["revenue_lag_30"]) / (df["revenue_lag_30"] + 1.0)
        
        # Backfill any initial NaNs caused by shift lags
        cols_to_fill = [
            "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_30",
            "spend_lag_1", "spend_lag_7", "roas_lag_1", "roas_lag_7",
            "revenue_growth_7d", "revenue_growth_30d"
        ]
        df[cols_to_fill] = df[cols_to_fill].bfill().fillna(0.0)
        
        return df

if __name__ == "__main__":
    from data_pipeline import DataPipeline
    dp = DataPipeline("data")
    df = dp.aggregate_sources()
    
    fe = FeatureEngineer()
    df_features = fe.generate_features(df)
    print("Features generated successfully. Shape:", df_features.shape)
    print("Sample feature columns:", [col for col in df_features.columns if "lag" in col or "roll" in col or "blended" in col])
