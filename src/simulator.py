import numpy as np
import pandas as pd

class BudgetSimulator:
    def __init__(self):
        self.calibrated = False
        self.alpha = {}  # Saturation coefficients
        self.organic_revenue = 0.0
        self.baseline_spend = {}
        self.baseline_revenue = 0.0
        self.baseline_roas = 0.0

    def calibrate(self, df_historical):
        """
        Calibrate logarithmic channel saturation curves using historical data.
        R_channel = alpha_channel * ln(Spend_channel + 1)
        """
        df = df_historical.copy()
        if len(df) < 14:
            # Fallback coefficients if data is too small
            self.alpha = {"google": 250.0, "meta": 220.0, "ms": 90.0}
            self.organic_revenue = 1500.0
            self.baseline_spend = {"google": 1000.0, "meta": 800.0, "ms": 200.0}
            self.baseline_revenue = 4000.0
            self.baseline_roas = 2.0
            self.calibrated = True
            return

        # Calculate historical averages
        avg_google_spend = df["Google_Spend"].mean()
        avg_meta_spend = df["Meta_Spend"].mean()
        avg_ms_spend = df["MS_Spend"].mean()
        avg_total_spend = df["Total_Spend"].mean()
        avg_revenue = df["Revenue"].mean()
        
        # Estimate organic revenue using GA4 organic channels
        # If GA4 channels aren't perfectly mapped, default to 30% of total revenue
        if "GA4_Revenue" in df.columns:
            # We can approximate organic as GA4 organic search + direct revenue
            # For simplicity in aggregate, let's assume organic is a baseline of 25% of sales
            self.organic_revenue = max(500.0, avg_revenue * 0.25)
        else:
            self.organic_revenue = avg_revenue * 0.25

        marketing_attributed_rev = avg_revenue - self.organic_revenue
        
        # Distribute marketing revenue based on spend shares or conversion shares
        google_conv_share = df["Google_Conversions"].sum() / (df["Total_Conversions"].sum() + 1)
        meta_conv_share = df["Meta_Conversions"].sum() / (df["Total_Conversions"].sum() + 1)
        ms_conv_share = df["MS_Conversions"].sum() / (df["Total_Conversions"].sum() + 1)
        
        # Fit coefficients: alpha_i = Attributed_Revenue_i / ln(Spend_i + 1)
        self.alpha["google"] = (marketing_attributed_rev * google_conv_share) / np.log(avg_google_spend + 2.0)
        self.alpha["meta"] = (marketing_attributed_rev * meta_conv_share) / np.log(avg_meta_spend + 2.0)
        self.alpha["ms"] = (marketing_attributed_rev * ms_conv_share) / np.log(avg_ms_spend + 2.0)
        
        self.baseline_spend = {
            "google": avg_google_spend,
            "meta": avg_meta_spend,
            "ms": avg_ms_spend
        }
        self.baseline_revenue = avg_revenue
        self.baseline_roas = avg_revenue / avg_total_spend if avg_total_spend > 0 else 0.0
        self.calibrated = True
        
        print("Simulator calibrated successfully.")
        print(f"Alphas: {self.alpha}")
        print(f"Organic Baseline: ${self.organic_revenue:.2f}")

    def simulate(self, spend_modifiers):
        """
        Simulate revenue and ROAS based on channel spend adjustments.
        spend_modifiers: dict containing percentage changes (e.g. {'google': 0.20, 'meta': -0.10, 'ms': 0.0})
        """
        if not self.calibrated:
            raise ValueError("Simulator has not been calibrated yet. Call calibrate(df) first.")
            
        sim_spends = {}
        sim_revenues = {}
        
        for channel in ["google", "meta", "ms"]:
            modifier = spend_modifiers.get(channel, 0.0)
            base_s = self.baseline_spend[channel]
            sim_spends[channel] = max(0.0, base_s * (1.0 + modifier))
            
            # Apply logarithmic saturation curve
            alpha_c = self.alpha[channel]
            sim_revenues[channel] = alpha_c * np.log(sim_spends[channel] + 2.0)
            
        # Total spend and revenue
        total_sim_spend = sum(sim_spends.values())
        total_sim_revenue = self.organic_revenue + sum(sim_revenues.values())
        total_sim_roas = total_sim_revenue / total_sim_spend if total_sim_spend > 0 else 0.0
        
        # Lifts and changes
        rev_lift = (total_sim_revenue - self.baseline_revenue) / self.baseline_revenue
        spend_change = (total_sim_spend - sum(self.baseline_spend.values())) / sum(self.baseline_spend.values())
        roas_change = (total_sim_roas - self.baseline_roas) / self.baseline_roas if self.baseline_roas > 0 else 0.0
        
        # Assess Simulator Risk
        # 1. High spend scale causes saturation (ROAS drops)
        # 2. Large budget cuts cause severe revenue drop
        risk_level = "Low"
        risk_reason = "Spend scaling is within efficient boundaries."
        
        if spend_change > 0.40:
            risk_level = "High"
            risk_reason = "Aggressive spend scaling (>40%) will trigger heavy ad auction saturation, severely lowering ROAS."
        elif spend_change > 0.15:
            risk_level = "Medium"
            risk_reason = "Moderate spend scaling may cause minor creative fatigue and slight ROAS decay."
        elif spend_change < -0.30:
            risk_level = "Medium"
            risk_reason = "Severe budget cuts (>30%) will trigger a sharp drop in transaction volume."
            
        # Channel mix shares
        channel_mix = {}
        for c in ["google", "meta", "ms"]:
            channel_mix[c] = sim_spends[c] / total_sim_spend if total_sim_spend > 0 else 0.0

        return {
            "baseline_spend": sum(self.baseline_spend.values()),
            "simulated_spend": total_sim_spend,
            "spend_change_pct": spend_change,
            
            "baseline_revenue": self.baseline_revenue,
            "simulated_revenue": total_sim_revenue,
            "revenue_lift_pct": rev_lift,
            
            "baseline_roas": self.baseline_roas,
            "simulated_roas": total_sim_roas,
            "roas_change_pct": roas_change,
            
            "channel_spends": sim_spends,
            "channel_mix": channel_mix,
            "risk_level": risk_level,
            "risk_reason": risk_reason
        }
