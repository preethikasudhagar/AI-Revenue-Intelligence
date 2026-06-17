import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Define date range: Jan 1, 2024 to June 15, 2026
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 6, 15)
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    n_days = len(dates)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Base variables for trend and seasonality
    time_index = np.arange(n_days)
    weekly_seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * time_index / 7)
    annual_seasonality = 1.0 + 0.25 * np.sin(2 * np.pi * time_index / 365.25)
    
    # Nov-Dec holiday surge
    holiday_surge = np.ones(n_days)
    for idx, date in enumerate(dates):
        if date.month == 11:  # November (Black Friday / Cyber Monday)
            holiday_surge[idx] = 1.4 + 0.3 * (date.day >= 20)
        elif date.month == 12:  # December (Christmas shopping)
            holiday_surge[idx] = 1.5 - 0.02 * date.day
            
    trend = 1.0 + 0.0005 * time_index  # Upward business trend
    
    # 2. Google Ads Data
    g_spend = np.random.normal(1500, 200, n_days) * trend * weekly_seasonality * holiday_surge
    g_spend = np.clip(g_spend, 500, None)
    g_ctr = np.random.normal(0.035, 0.003, n_days)  # 3.5% CTR
    g_cpc = np.random.normal(1.20, 0.05, n_days)    # $1.20 CPC
    g_clicks = (g_spend / g_cpc).astype(int)
    g_impressions = (g_clicks / g_ctr).astype(int)
    g_conv_rate = np.random.normal(0.028, 0.002, n_days) # 2.8% Conversion Rate
    g_conversions = (g_clicks * g_conv_rate).astype(int)
    
    g_campaigns = ["Search_Brand", "Search_Generic", "PerformanceMax_Revenue"]
    g_data = []
    for i in range(n_days):
        # Distribute daily spend across campaigns
        for campaign in g_campaigns:
            pct = 0.3 if "Brand" in campaign else (0.4 if "Max" in campaign else 0.3)
            rand_pct = np.random.dirichlet(np.ones(3))[0] * 0.2 + pct * 0.8
            g_data.append({
                "Date": dates[i].strftime("%Y-%m-%d"),
                "Campaign": campaign,
                "Spend": round(g_spend[i] * rand_pct, 2),
                "Clicks": int(g_clicks[i] * rand_pct),
                "Impressions": int(g_impressions[i] * rand_pct),
                "Conversions": int(g_conversions[i] * rand_pct)
            })
    df_google = pd.DataFrame(g_data)
    df_google.to_csv(os.path.join(output_dir, "sample_google_ads.csv"), index=False)
    
    # 3. Meta Ads Data
    m_spend = np.random.normal(1200, 250, n_days) * trend * (1.2 - 0.1 * np.sin(2 * np.pi * time_index / 7)) * holiday_surge
    m_spend = np.clip(m_spend, 300, None)
    m_ctr = np.random.normal(0.018, 0.002, n_days)  # 1.8% CTR
    m_cpc = np.random.normal(0.85, 0.04, n_days)    # $0.85 CPC
    m_clicks = (m_spend / m_cpc).astype(int)
    m_impressions = (m_clicks / m_ctr).astype(int)
    m_conv_rate = np.random.normal(0.022, 0.002, n_days) # 2.2% Conversion Rate
    m_conversions = (m_clicks * m_conv_rate).astype(int)
    
    m_campaigns = ["Prospecting_Lookalike", "Retargeting_Catalog", "AdvantagePlus_Shopping"]
    m_data = []
    for i in range(n_days):
        for campaign in m_campaigns:
            pct = 0.4 if "Advantage" in campaign else (0.35 if "Prospecting" in campaign else 0.25)
            rand_pct = np.random.dirichlet(np.ones(3))[0] * 0.2 + pct * 0.8
            m_data.append({
                "Date": dates[i].strftime("%Y-%m-%d"),
                "Campaign": campaign,
                "Spend": round(m_spend[i] * rand_pct, 2),
                "Clicks": int(m_clicks[i] * rand_pct),
                "Impressions": int(m_impressions[i] * rand_pct),
                "Conversions": int(m_conversions[i] * rand_pct)
            })
    df_meta = pd.DataFrame(m_data)
    df_meta.to_csv(os.path.join(output_dir, "sample_meta_ads.csv"), index=False)
    
    # 4. Microsoft Ads Data
    ms_spend = np.random.normal(300, 60, n_days) * trend * weekly_seasonality * holiday_surge
    ms_spend = np.clip(ms_spend, 50, None)
    ms_ctr = np.random.normal(0.025, 0.003, n_days)  # 2.5% CTR
    ms_cpc = np.random.normal(0.95, 0.05, n_days)    # $0.95 CPC
    ms_clicks = (ms_spend / ms_cpc).astype(int)
    ms_impressions = (ms_clicks / ms_ctr).astype(int)
    ms_conv_rate = np.random.normal(0.020, 0.002, n_days) # 2.0% Conversion Rate
    ms_conversions = (ms_clicks * ms_conv_rate).astype(int)
    
    ms_campaigns = ["Bing_Search_Brand", "Bing_Shopping"]
    ms_data = []
    for i in range(n_days):
        for campaign in ms_campaigns:
            pct = 0.4 if "Brand" in campaign else 0.6
            rand_pct = np.random.dirichlet(np.ones(2))[0] * 0.1 + pct * 0.9
            ms_data.append({
                "Date": dates[i].strftime("%Y-%m-%d"),
                "Campaign": campaign,
                "Spend": round(ms_spend[i] * rand_pct, 2),
                "Clicks": int(ms_clicks[i] * rand_pct),
                "Impressions": int(ms_impressions[i] * rand_pct),
                "Conversions": int(ms_conversions[i] * rand_pct)
            })
    df_ms = pd.DataFrame(ms_data)
    df_ms.to_csv(os.path.join(output_dir, "sample_microsoft_ads.csv"), index=False)
    
    # 5. GA4 Data (Traffic Channels)
    ga_channels = ["Organic Search", "Direct", "Referral", "Email", "Paid Search", "Paid Social"]
    ga_data = []
    for i in range(n_days):
        # Base daily sessions
        base_sessions = np.random.normal(8000, 1000, n_days)[i] * trend[i] * weekly_seasonality[i] * annual_seasonality[i] * holiday_surge[i]
        base_sessions = max(2000, base_sessions)
        
        # Share rates
        shares = {
            "Organic Search": 0.35,
            "Direct": 0.20,
            "Paid Search": 0.25 * (g_spend[i] / 1500), # correlated with spend
            "Paid Social": 0.15 * (m_spend[i] / 1200),  # correlated with spend
            "Email": 0.08,
            "Referral": 0.02
        }
        # Normalize shares
        total_share = sum(shares.values())
        for k in shares:
            shares[k] /= total_share
            
        for channel in ga_channels:
            sessions = int(base_sessions * shares[channel])
            pageviews = int(sessions * np.random.normal(2.5, 0.2))
            
            # Conv rates
            cr = 0.015 if channel in ["Organic Search", "Direct"] else 0.025
            if channel == "Paid Search":
                cr = g_conv_rate[i]
            elif channel == "Paid Social":
                cr = m_conv_rate[i]
                
            conversions = int(sessions * cr)
            # Revenue from GA4
            aov = np.random.normal(85.0, 3.0)
            revenue = round(conversions * aov, 2)
            
            ga_data.append({
                "Date": dates[i].strftime("%Y-%m-%d"),
                "Channel": channel,
                "Sessions": sessions,
                "Pageviews": pageviews,
                "Conversions": conversions,
                "Revenue": revenue
            })
    df_ga4 = pd.DataFrame(ga_data)
    df_ga4.to_csv(os.path.join(output_dir, "sample_ga4.csv"), index=False)
    
    # 6. Shopify Data (Transactions & Realized Revenue)
    # Total revenue will be the sum of paid channel conversions + organic + some error
    # Let's model a realistic customer purchase database
    shopify_data = []
    for i in range(n_days):
        # Calculate daily total conversions and revenue
        ad_conversions = g_conversions[i] + m_conversions[i] + ms_conversions[i]
        organic_conversions = int(np.random.normal(120, 15) * trend[i] * weekly_seasonality[i] * annual_seasonality[i] * holiday_surge[i])
        total_orders = int(ad_conversions * 1.05 + organic_conversions) # some attribution overlap
        
        # AOV
        aov = np.random.normal(92.0, 2.0, n_days)[i]
        total_revenue = round(total_orders * aov, 2)
        
        # Returns (between 5% and 12%)
        return_rate = np.random.normal(0.08, 0.01)
        returned_orders = int(total_orders * return_rate)
        refunded_amount = round(returned_orders * aov * 0.95, 2)
        
        shopify_data.append({
            "Date": dates[i].strftime("%Y-%m-%d"),
            "Orders": total_orders,
            "Gross_Revenue": total_revenue,
            "Refunds": refunded_amount,
            "Net_Revenue": round(total_revenue - refunded_amount, 2),
            "Returned_Orders": returned_orders
        })
    df_shopify = pd.DataFrame(shopify_data)
    
    # Inject some anomalies (outliers) for data validation testing
    # 5 random days of abnormally high revenue (promo days)
    promo_days = np.random.choice(n_days, 5, replace=False)
    for idx in promo_days:
        df_shopify.loc[idx, "Net_Revenue"] = round(df_shopify.loc[idx, "Net_Revenue"] * 2.5, 2)
        df_shopify.loc[idx, "Orders"] = int(df_shopify.loc[idx, "Orders"] * 2.3)
        
    # 3 random days of data drop (technical tracking issues)
    drop_days = np.random.choice(n_days, 3, replace=False)
    for idx in drop_days:
        df_shopify.loc[idx, "Net_Revenue"] = 0.0
        df_shopify.loc[idx, "Orders"] = 0
        
    df_shopify.to_csv(os.path.join(output_dir, "sample_shopify.csv"), index=False)
    
    print(f"Mock data successfully written to '{output_dir}/'")

if __name__ == "__main__":
    generate_mock_data()
