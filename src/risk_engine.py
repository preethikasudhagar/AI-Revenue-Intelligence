import pandas as pd
import numpy as np

class MarketingRiskEngine:
    def __init__(self):
        pass

    def assess_risks(self, df_historical):
        """
        Assess marketing campaign risks based on historical metrics.
        Returns a dictionary containing risk levels and detailed check statuses.
        """
        df = df_historical.copy()
        if len(df) < 30:
            return {
                "overall_risk_level": "Low",
                "risk_score": 0.15,
                "summary": "Insufficient historical data to detect deep marketing anomalies. Currently stable.",
                "details": []
            }
            
        recent_14 = df.tail(14)
        prev_14 = df.iloc[-28:-14]
        
        risk_flags = []
        risk_score = 0.0
        
        # 1. ROAS Decay Check (Increasing Spend + Decreasing ROAS)
        spend_change = (recent_14["Total_Spend"].mean() - prev_14["Total_Spend"].mean()) / (prev_14["Total_Spend"].mean() + 1.0)
        roas_change = (recent_14["ROAS"].mean() - prev_14["ROAS"].mean()) / (prev_14["ROAS"].mean() + 1.0)
        
        if spend_change > 0.05 and roas_change < -0.05:
            # Spend is increasing, but efficiency is decreasing
            severity = "High" if roas_change < -0.15 else "Medium"
            risk_flags.append({
                "risk_name": "ROAS Decay",
                "severity": severity,
                "message": f"Ad spend increased by {spend_change*100:.1f}% over the last 14 days, but blended ROAS decreased by {roas_change*100:.1f}%. High risk of inefficient spend scale.",
                "recommendation": "Pause bid scaling. Audit creative performance and audience targeting across active channels."
            })
            risk_score += 0.35 if severity == "High" else 0.20
        elif roas_change < -0.10:
            risk_flags.append({
                "risk_name": "ROAS Decline",
                "severity": "Medium",
                "message": f"Blended ROAS fell by {roas_change*100:.1f}% without significant spend scale.",
                "recommendation": "Check for creative fatigue or website landing page conversion issues."
            })
            risk_score += 0.15

        # 2. Spend Saturation Check (Check if CPC is spikes and Conv Rate drops)
        cpc_change = (recent_14["blended_cpc"].mean() - prev_14["blended_cpc"].mean()) / (prev_14["blended_cpc"].mean() + 1.0)
        if cpc_change > 0.15:
            risk_flags.append({
                "risk_name": "Ad Fatigue & CPC Spike",
                "severity": "Medium",
                "message": f"Blended Cost Per Click (CPC) rose by {cpc_change*100:.1f}%, indicating rising ad auction competition or ad fatigue.",
                "recommendation": "Refresh ad creatives, expand target audiences, or shift budget to channels with lower CPC."
            })
            risk_score += 0.15

        # 3. Conversion Rate Drop Check
        cvr_recent = recent_14["blended_conv_rate"].mean()
        cvr_historical_mean = df["blended_conv_rate"].mean()
        cvr_historical_std = df["blended_conv_rate"].std()
        
        if cvr_recent < (cvr_historical_mean - 1.2 * cvr_historical_std):
            risk_flags.append({
                "risk_name": "Conversion Rate Drop-off",
                "severity": "High",
                "message": f"Current conversion rate ({cvr_recent*100:.2f}%) is significantly below historical average ({cvr_historical_mean*100:.2f}%). Potential Shopify funnel friction or cart issues.",
                "recommendation": "Perform technical website audit. Check cart checkout flow and ensure discount codes / promotions are functional."
            })
            risk_score += 0.30

        # 4. Revenue Volatility Check
        roll_std_30 = df["Revenue"].rolling(window=30).std()
        roll_mean_30 = df["Revenue"].rolling(window=30).mean()
        coeff_var = (roll_std_30 / roll_mean_30).iloc[-1]
        
        if coeff_var > 0.35:
            risk_flags.append({
                "risk_name": "High Revenue Volatility",
                "severity": "Medium",
                "message": f"Revenue volatility is high (coefficient of variation: {coeff_var:.2f}). Cash flow predictability is reduced.",
                "recommendation": "Incorporate diversified channels (e.g. Email/SMS marketing) to stabilize sales baseline."
            })
            risk_score += 0.15

        # Consolidate overall risk level
        if risk_score >= 0.50:
            overall_level = "High"
        elif risk_score >= 0.20:
            overall_level = "Medium"
        else:
            overall_level = "Low"
            
        summary = ""
        if overall_level == "High":
            summary = "CRITICAL: Multiple marketing inefficiency indicators triggered. ROAS and conversion rates are dropping. Action required."
        elif overall_level == "Medium":
            summary = "WARNING: Moderate budget inefficiencies detected. Creative fatigue or rising CPCs are impacting ROAS."
        else:
            summary = "STABLE: Marketing performance remains efficient with stable ROAS and conversion baselines."
            
        return {
            "overall_risk_level": overall_level,
            "risk_score": round(risk_score, 2),
            "summary": summary,
            "details": risk_flags
        }

if __name__ == "__main__":
    re = MarketingRiskEngine()
    print("Fallback Risk Check:\n", re.assess_risks(pd.DataFrame()))
