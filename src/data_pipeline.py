import os
import pandas as pd
import numpy as np
from datetime import datetime

class DataPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.validation_report = {}
        
        # Define expected columns for validation
        self.schemas = {
            "google": ["Date", "Campaign", "Spend", "Clicks", "Impressions", "Conversions"],
            "meta": ["Date", "Campaign", "Spend", "Clicks", "Impressions", "Conversions"],
            "microsoft": ["Date", "Campaign", "Spend", "Clicks", "Impressions", "Conversions"],
            "ga4": ["Date", "Channel", "Sessions", "Pageviews", "Conversions", "Revenue"],
            "shopify": ["Date", "Orders", "Gross_Revenue", "Refunds", "Net_Revenue", "Returned_Orders"]
        }

    def load_csv(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file {filename} not found in {self.data_dir}")
        return pd.read_csv(file_path)

    def validate_schema(self, df, source_name):
        expected_cols = self.schemas[source_name]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        report = {
            "total_rows": len(df),
            "missing_columns": missing_cols,
            "has_missing_columns": len(missing_cols) > 0,
            "null_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum())
        }
        self.validation_report[source_name] = report
        
        if report["has_missing_columns"]:
            raise ValueError(f"Schema validation failed for {source_name}. Missing columns: {missing_cols}")
        return True

    def clean_and_impute(self, df, date_col="Date"):
        # Ensure date format
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Handle duplicates: drop exact duplicates
        df = df.drop_duplicates()
        
        # Fill missing values: numeric columns filled with 0 or interpolated
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Categorical columns filled with 'Unknown'
        categorical_cols = df.select_dtypes(exclude=[np.number, np.datetime64]).columns
        df[categorical_cols] = df[categorical_cols].fillna("Unknown")
        
        return df

    def detect_anomalies_zscore(self, df, column, threshold=3.0):
        """
        Detect anomalies in a numeric column using rolling Z-Score.
        """
        if len(df) < 14:  # Needs minimum window size
            return []
        
        # Calculate rolling mean and std
        rolling_mean = df[column].rolling(window=14, min_periods=1).mean()
        rolling_std = df[column].rolling(window=14, min_periods=1).std().fillna(1.0)
        
        z_scores = (df[column] - rolling_mean) / rolling_std
        anomalies = df[abs(z_scores) > threshold]
        
        return anomalies.index.tolist()

    def aggregate_sources(self):
        # 1. Load Data
        df_google = self.load_csv("sample_google_ads.csv")
        df_meta = self.load_csv("sample_meta_ads.csv")
        df_ms = self.load_csv("sample_microsoft_ads.csv")
        df_ga4 = self.load_csv("sample_ga4.csv")
        df_shopify = self.load_csv("sample_shopify.csv")
        
        # 2. Validate
        self.validate_schema(df_google, "google")
        self.validate_schema(df_meta, "meta")
        self.validate_schema(df_ms, "microsoft")
        self.validate_schema(df_ga4, "ga4")
        self.validate_schema(df_shopify, "shopify")
        
        # 3. Clean
        df_google = self.clean_and_impute(df_google)
        df_meta = self.clean_and_impute(df_meta)
        df_ms = self.clean_and_impute(df_ms)
        df_ga4 = self.clean_and_impute(df_ga4)
        df_shopify = self.clean_and_impute(df_shopify)
        
        # 4. Daily Aggregations
        # Google
        df_g_daily = df_google.groupby("Date").agg({
            "Spend": "sum",
            "Clicks": "sum",
            "Impressions": "sum",
            "Conversions": "sum"
        }).rename(columns=lambda x: f"Google_{x}").reset_index()
        
        # Meta
        df_m_daily = df_meta.groupby("Date").agg({
            "Spend": "sum",
            "Clicks": "sum",
            "Impressions": "sum",
            "Conversions": "sum"
        }).rename(columns=lambda x: f"Meta_{x}").reset_index()
        
        # Microsoft
        df_ms_daily = df_ms.groupby("Date").agg({
            "Spend": "sum",
            "Clicks": "sum",
            "Impressions": "sum",
            "Conversions": "sum"
        }).rename(columns=lambda x: f"MS_{x}").reset_index()
        
        # GA4
        df_ga4_daily = df_ga4.groupby("Date").agg({
            "Sessions": "sum",
            "Pageviews": "sum",
            "Conversions": "sum",
            "Revenue": "sum"
        }).rename(columns=lambda x: f"GA4_{x}").reset_index()
        
        # Shopify
        df_shop_daily = df_shopify.groupby("Date").agg({
            "Orders": "sum",
            "Gross_Revenue": "sum",
            "Refunds": "sum",
            "Net_Revenue": "sum",
            "Returned_Orders": "sum"
        }).rename(columns=lambda x: f"Shopify_{x}").reset_index()
        
        # 5. Merge all datasets on Date
        merged = df_shop_daily.merge(df_g_daily, on="Date", how="outer")
        merged = merged.merge(df_m_daily, on="Date", how="outer")
        merged = merged.merge(df_ms_daily, on="Date", how="outer")
        merged = merged.merge(df_ga4_daily, on="Date", how="outer")
        
        # Fill missing dates/metrics with 0 since outer merge might create NaNs
        merged = merged.sort_values("Date").reset_index(drop=True)
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(0)
        
        # Calculate Blended Marketing Metrics
        merged["Total_Spend"] = merged["Google_Spend"] + merged["Meta_Spend"] + merged["MS_Spend"]
        merged["Total_Clicks"] = merged["Google_Clicks"] + merged["Meta_Clicks"] + merged["MS_Clicks"]
        merged["Total_Impressions"] = merged["Google_Impressions"] + merged["Meta_Impressions"] + merged["MS_Impressions"]
        merged["Total_Conversions"] = merged["Google_Conversions"] + merged["Meta_Conversions"] + merged["MS_Conversions"]
        
        merged["Revenue"] = merged["Shopify_Net_Revenue"]
        merged["ROAS"] = np.where(merged["Total_Spend"] > 0, merged["Revenue"] / merged["Total_Spend"], 0.0)
        
        # Detect Anomalies in consolidated Spend and Revenue
        rev_anomalies = self.detect_anomalies_zscore(merged, "Revenue")
        spend_anomalies = self.detect_anomalies_zscore(merged, "Total_Spend")
        
        self.validation_report["consolidated"] = {
            "total_days": len(merged),
            "revenue_anomalies_count": len(rev_anomalies),
            "spend_anomalies_count": len(spend_anomalies),
            "revenue_anomalies_indices": rev_anomalies,
            "spend_anomalies_indices": spend_anomalies
        }
        
        return merged

    def get_validation_report(self):
        return self.validation_report

if __name__ == "__main__":
    # Test data pipeline run
    pipeline = DataPipeline("data")
    merged_df = pipeline.aggregate_sources()
    print("Aggregate shape:", merged_df.shape)
    print("Anomalies found in consolidated data:", pipeline.get_validation_report()["consolidated"])
