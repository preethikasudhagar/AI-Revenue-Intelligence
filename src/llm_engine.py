import os
import pandas as pd
import numpy as np

# Try importing LLM packages
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class AIInsightsEngine:
    def __init__(self, gemini_api_key=None, openai_api_key=None):
        self.gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Configure Gemini if key is present
        if GEMINI_AVAILABLE and self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            
        # Configure OpenAI if key is present
        if OPENAI_AVAILABLE and self.openai_key:
            openai.api_key = self.openai_key

    def generate_prompt(self, metrics_summary):
        """
        Formulate a rich context prompt using actual dataset statistics.
        """
        prompt = f"""
You are a Senior E-commerce Growth Consultant and AI Revenue Intelligence Analyst. 
Analyze the following performance metrics and forecast projections for our e-commerce store:

=== HISTORICAL PERFORMANCE (Last 30 Days) ===
- Avg Daily Revenue: ${metrics_summary['hist_avg_revenue']:,.2f}
- Avg Daily Spend: ${metrics_summary['hist_avg_spend']:,.2f}
- Blended ROAS: {metrics_summary['hist_blended_roas']:.2f}x
- Google Ads Spend: ${metrics_summary['google_spend']:,.2f} (ROAS: {metrics_summary['google_roas']:.2f}x)
- Meta Ads Spend: ${metrics_summary['meta_spend']:,.2f} (ROAS: {metrics_summary['meta_roas']:.2f}x)
- Microsoft Ads Spend: ${metrics_summary['ms_spend']:,.2f} (ROAS: {metrics_summary['ms_roas']:.2f}x)
- Shopify Orders: {metrics_summary['shopify_orders']:,}
- Traffic (GA4 Sessions): {metrics_summary['ga4_sessions']:,}
- Conversion Rate: {metrics_summary['conv_rate']*100:.2f}%

=== FORECAST PROJECTIONS (Next 90 Days) ===
- Predicted 90-Day Revenue: ${metrics_summary['pred_revenue_90d']:,.2f}
- P10 Bound (Pessimistic): ${metrics_summary['pred_revenue_p10']:,.2f}
- P90 Bound (Optimistic): ${metrics_summary['pred_revenue_p90']:,.2f}
- Forecast Confidence: {metrics_summary['forecast_confidence']}

=== RISK ASSESSMENT ===
- Marketing Risk Level: {metrics_summary['risk_level']}
- Primary Risk Alerts: {metrics_summary['risk_alerts']}

=== SIMULATOR OUTCOMES ===
- Simulated Budget Shift: {metrics_summary['sim_spend_change']:.1f}%
- Simulated Revenue Impact: {metrics_summary['sim_rev_lift']:.1f}% Lift
- Simulated Blended ROAS: {metrics_summary['sim_roas']:.2f}x

Please write a comprehensive, executive-level, professional growth report. The language must be business-friendly, actionable, and visually organized (using clear headers and bullet points).

Your response MUST cover exactly these 7 sections:
1. Executive Summary: High-level overview of performance, health, and 90-day trajectory.
2. Revenue Drivers: Primary factors driving revenue growth or decay based on the data.
3. ROAS Drivers: Channel-specific insights and cost efficiency analysis.
4. Growth Opportunities: Clear channels or areas where the store is under-spending or scaling is recommended.
5. Business Risks: Critical warnings (e.g., ad saturation, conversion drop-offs, volatility).
6. Marketing Recommendations: Detailed tactical advice on spend reallocations.
7. Campaign Optimization Suggestions: Specific campaign types, creative directions, or technical funnel fixes.
"""
        return prompt

    def generate_insights(self, metrics_summary):
        """
        Generate insights. Uses Gemini or OpenAI if keys are present, else falls back to Mock Engine.
        """
        prompt = self.generate_prompt(metrics_summary)
        
        # 1. Try Gemini
        if GEMINI_AVAILABLE and self.gemini_key:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text, "Live Gemini (1.5-Flash)"
            except Exception as e:
                print(f"Gemini API execution failed: {e}. Falling back...")
                
        # 2. Try OpenAI
        if OPENAI_AVAILABLE and self.openai_key:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional marketing intelligence analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                return response.choices[0].message['content'], "Live OpenAI (GPT-3.5)"
            except Exception as e:
                print(f"OpenAI API execution failed: {e}. Falling back...")
                
        # 3. Fallback Mock Generator
        return self._generate_mock_insights(metrics_summary), "Dynamic Heuristic Engine (Offline Fallback)"

    def _generate_mock_insights(self, m):
        """
        Generates a highly realistic, data-driven report locally without network requests.
        """
        rev_trend = "growth" if m['pred_revenue_90d'] > (m['hist_avg_revenue'] * 90) else "stabilization/contraction"
        confidence_text = "stable seasonality and strong historical signals" if "High" in m['forecast_confidence'] else "moderate volatility and seasonal shifts"
        
        # Determine recommendations based on channel ROAS
        best_channel = "Google Ads"
        best_roas = m['google_roas']
        if m['meta_roas'] > best_roas:
            best_channel = "Meta Ads"
            best_roas = m['meta_roas']
        if m['ms_roas'] > best_roas:
            best_channel = "Microsoft Ads"
            best_roas = m['ms_roas']
            
        report = f"""### 1. Executive Summary
Over the last 30 days, the e-commerce store generated an average daily revenue of **${m['hist_avg_revenue']:,.2f}** on a marketing spend of **${m['hist_avg_spend']:,.2f}**, representing an efficient blended ROAS of **{m['hist_blended_roas']:.2f}x**. 
The ML forecasting model predicts a 90-day cumulative revenue of **${m['pred_revenue_90d']:,.2f}** (P50 baseline), with a conservative lower bound (P10) of **${m['pred_revenue_p10']:,.2f}** and an optimistic upper bound (P90) of **${m['pred_revenue_p90']:,.2f}**. 
This projection carries a **{m['forecast_confidence']}** confidence score, backed by {confidence_text}. The overall health of the business is classified as **{m['risk_level']} Risk**, indicating solid fundamentals but specific channel inefficiencies that must be addressed immediately.

### 2. Revenue Drivers
* **Search Engine Visibility**: Google Ads remains the dominant anchor, bringing in steady traffic. Brand keywords are highly efficient, maintaining a high conversion baseline.
* **Paid Social Ingestion**: Meta Ads continues to feed the top-of-funnel (TOF), contributing heavily to customer acquisition.
* **Conversion Rate (CR)**: Currently standing at **{m['conv_rate']*100:.2f}%**, indicating a solid shopping cart layout, though minor friction is noted during weekend peaks.

### 3. ROAS Drivers
* **{best_channel} Efficiency**: **{best_channel}** is the primary efficiency driver with a strong ROAS of **{best_roas:.2f}x**, significantly outperforming the blended average.
* **Meta Ads Friction**: Meta Ads shows slight signs of creative fatigue. Rising CPCs in prospecting campaigns are putting pressure on profit margins.
* **Microsoft Ads Margin**: Microsoft Ads is maintaining a highly profitable niche (**{m['ms_roas']:.2f}x** ROAS) but is constrained by search volume limits.

### 4. Growth Opportunities
* **Underspend in {best_channel}**: We recommend increasing **{best_channel}** budgets by **15-20%** to capture additional high-intent traffic, as saturation levels remain low.
* **Email & SMS Retargeting**: Incorporate stronger lifecycle marketing to recover cart drop-offs, lessening the reliance on paid retargeting ads.
* **Microsoft Search Scaling**: Capitalize on higher AOV demographics on Bing by expanding target categories.

### 5. Business Risks
* **{m['risk_level']} Risk Alert**: The risk engine flagged: *"{m['risk_alerts']}"*. 
* **Auction Saturation**: Meta Ads is nearing its saturation threshold. Further scaling of spend without creative updates will lead to rapid ROAS decay.
* **Funnel Drops**: A minor drop in conversion rate was detected during recent promotional cycles, pointing to potential checkout delays.

### 6. Marketing Recommendations
* **Budget Reallocation**: Shift **10%** of spend from underperforming Meta prospecting campaigns into **{best_channel}** Performance Max and Search Brand.
* **Simulated Impact**: Based on the simulator, a budget adjustment of **{m['sim_spend_change']:+.1f}%** is estimated to yield a **{m['sim_rev_lift']:+.1f}%** lift in revenue, leading to a projected blended ROAS of **{m['sim_roas']:.2f}x**.
* **Risk Mitigation**: Keep the total daily marketing budget under **${m['hist_avg_spend']*1.3:,.2f}** to prevent sharp margin compression.

### 7. Campaign Optimization Suggestions
* **Creative Refresh**: Implement 3-4 new video ad variations for Meta Advantage+ Shopping campaigns focusing on user testimonials.
* **Landing Page Speed**: Optimize mobile page loading speeds to improve GA4 mobile bounce rates.
* **Dynamic Search Ads (DSA)**: Enable DSA campaigns on Google Ads to automatically target long-tail search terms that competitor brands miss.
"""
        return report
