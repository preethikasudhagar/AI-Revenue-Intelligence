# Live Demo Script - AI Revenue Intelligence

This script guides you through presenting the platform to judges.

---

## Part 1: Preparation & Startup (1 Minute)

1. **Start the App**:
   Run the following command in terminal:
   ```bash
   streamlit run streamlit_app.py
   ```
2. **Open the browser**: Navigate to `http://localhost:8501`.
3. **Sidebar Check**: Point out the branding logo and version in the sidebar navigation.

---

## Part 2: Dashboard Home - Executive View (2 Minutes)

* **Action**: Click on **"Dashboard Home"** in the sidebar.
* **Script**:
  > "Welcome to the Executive Dashboard. At a glance, we see our key blended marketing metrics for the last 30 days: our Net Revenue is **$785K** on a spend of **$272K**, giving us a highly efficient blended ROAS of **2.88x** with a solid checkout conversion rate of **2.30%**."
* **Action**: Point to the **"Blended Marketing Spend vs. Net Revenue Trend"** graph.
  > "Here we see our daily revenue trends mapped against total marketing spend. Notice the strong seasonal correlation: when we scale spend during peak periods, revenue scales. But how efficiently does it scale?"
* **Action**: Hover over the **"Ad Spend Channel Contribution"** donut chart and channel table.
  > "Below, we see our budget mix: Google Ads is our largest spend channel, followed closely by Meta, while Bing represents a smaller, highly efficient channel."

---

## Part 3: Data Ingestion & Quality Audit (1.5 Minutes)

* **Action**: Click on **"Data Ingestion & Validation"** in the sidebar.
* **Script**:
  > "Before running any models, we audit data quality. Our ingestion pipeline dynamically validates schemas for Google, Meta, Bing, GA4, and Shopify CSVs. Green badges verify that all files conform to expected formats."
* **Action**: Scroll to the **"Outliers & Anomaly Detection"** section.
  > "The pipeline also runs rolling Z-Score checks. It detected 8 anomalies where Shopify revenue spiked or dropped. Spikes indicate successful flash promotions, while drops indicate tracking errors. Clean data is vital for forecasting."

---

## Part 4: Forecasting Engine & Explainability (2.5 Minutes)

* **Action**: Click on **"Forecasting Engine"** in the sidebar.
* **Script**:
  > "Let's review our forecasts. We combine Prophet (for seasonality) with XGBoost and LightGBM (for marketing interactions). Quantile Regression layers generate pessimistic P10 and optimistic P90 bounds."
* **Action**: Click on the **30**, **60**, and **90** days horizon radio buttons.
  > "We predict a 90-day Net Revenue of **$2.01M** with a **High (95%)** confidence score. The shaded green area shows our statistical uncertainty band."
* **Action**: Scroll down to the **"Model Explainability & Causal Revenue Drivers"** section.
  > "Judges, we demystify the black-box models. The chart shows that **Ad Spend** and **Conversion Rate** have the highest feature importance. Causal drivers highlight that a recent +42% change in spend was the primary driver of revenue growth."

---

## Part 5: Budget Simulator & Channel Saturation (2 Minutes)

* **Action**: Click on **"Budget Simulator"** in the sidebar.
* **Script**:
  > "Now, the budget simulator. Traditional simulators assume linear returns. Ours models logarithmic diminishing returns for each channel based on historical performance."
* **Action**: In the sidebar, slide the **Google Ads Shift to +20%**, **Meta Ads to -10%**, and **Microsoft Ads to +15%**.
  > "Let's run a scenario: we increase Google by 20%, prune Meta by 10% due to rising acquisition costs, and scale Microsoft by 15%. Immediately, the system projects a **+8.4% Revenue Lift** and a simulated ROAS of **2.75x**."
* **Action**: Point to the **"Saturation & Volatility Risk Analysis"** warning card.
  > "Importantly, the simulator flags the saturation risk level. A minor spend shift triggers a **Low Risk** alert, confirming our budget reallocation remains efficient."

---

## Part 6: Generative AI Insights & Printable Reports (2 Minutes)

* **Action**: Click on **"AI Insights Engine"** in the sidebar.
* **Script**:
  > "Our platform translates raw statistics into business-ready strategies. I'll click 'Generate Executive Insights'."
* **Action**: Click the button. Let it load (takes 1-2 seconds with local fallback).
  > "Our AI Engine drafts a comprehensive, executive report. It organizes key findings into: Executive Summary, Revenue Drivers, ROAS Drivers, and tactical Campaign Optimization Suggestions."
* **Action**: Click on **"Reports Hub"** in the sidebar.
  > "Finally, we go to the Reports Hub. Here, users can download the entire interactive summary as a printable HTML document."
* **Action**: Click the **"Download HTML Executive Report"** button.
  > "This report uses a clean, print-friendly light template. Executives can open it and press Ctrl+P to save it directly as a premium PDF document."

---

## Part 7: Conclusion (1 Minute)

* **Script**:
  > "To summarize: AI Revenue Intelligence combines robust ingestion, advanced ensemble forecasting, probabilistic risk bands, calibrated simulator curves, and GenAI summaries into a unified growth tool. Thank you, judges. We are open for questions."
