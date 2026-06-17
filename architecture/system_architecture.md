# System Architecture - AI Revenue Intelligence

This document outlines the end-to-end technical system architecture of the AI Revenue Intelligence platform, designed as a production-ready marketing decision-support system.

## High-Level Architecture Flow

```mermaid
graph TD
    subgraph Data Sources
        GA4[Google Analytics 4] --> |Sessions, Channel Conv| DP[Data Pipeline]
        Shopify[Shopify CSV] --> |Orders, Net Revenue| DP
        GoogleAds[Google Ads CSV] --> |Campaign Spend, Clicks| DP
        MetaAds[Meta Ads CSV] --> |Campaign Spend, Clicks| DP
        MSAds[Microsoft Ads CSV] --> |Campaign Spend, Clicks| DP
    end

    subgraph Preprocessing & Validation
        DP --> |1. Schema Match| VAL[Validation Layer]
        VAL --> |2. Missing Value Imputation| CLEAN[Cleaning & Aggregation]
        CLEAN --> |3. Z-Score Outlier Detection| FE[Feature Engineering Engine]
    end

    subgraph Feature Engineering
        FE --> |Lags, Lags of Lags| LAG[Lag & Growth Features]
        FE --> |Rolling Averages, Rolling Std| ROLL[Rolling Stats]
        FE --> |Cyclic Day-of-Year Sin/Cos| TIME[Time Seasonality]
        FE --> |Blended CPC, CTR, Conv Rate| EFF[Marketing Efficiency]
    end

    subgraph Machine Learning Core
        LAG & ROLL & TIME & EFF --> |Historical Features| TR[Model Training - Offline]
        TR --> |Fit Prophet, XGBoost, LightGBM| FIT[Model Fitting]
        FIT --> |Save Serialized Model Pack| PKL[pickle/model.pkl]
    end

    subgraph Inference & Simulator (CLI / Dashboard)
        PKL --> |Load Models| PRED[Prediction Engine]
        PRED --> |Recursive Multi-Step Loop| PROP[Probabilistic Layer: Quantile Reg]
        PROP --> |Generate P10, P50, P90 Projections| OUT[predictions.csv]
        
        SIM[Budget Simulator] --> |Diminishing Returns Saturation Curve| SIM_OUT[Simulated Revenue, ROAS, Saturation Risk]
    end

    subgraph AI Insights & Reports
        OUT & SIM_OUT --> |Context Metrics Summary| LLM[Gemini / OpenAI Engine]
        LLM --> |Narrative Executive Report| REP[HTML Report Generator]
        REP --> |Download Report| PDF[Executive PDF Download]
    end

    subgraph Presentation UI
        OUT --> |Plotly Visualizer Chart| ST[Streamlit Dashboard]
        SIM_OUT --> |Interactive Sliiders & Mix Pie| ST
        LLM --> |Generative Report Viewer| ST
    end
```

---

## Detailed Components

### 1. Ingestion & Validation Layer (`src/data_pipeline.py`)
- **Schema Validation**: Explicitly verifies column structure, types, and date strings for 5 standard data sources (Google Ads, Meta Ads, Microsoft Ads, GA4, Shopify).
- **Quality Cleaning**: Performs deduplication, fills null gaps using forward-fill/interpolation, and maps records to a daily consolidated timestamp.
- **Z-Score Anomaly Engine**: Analyzes rolling standard deviations of Net Revenue and Marketing Spend. Anomalous drops or promotional surges are flagged as warnings for marketing teams.

### 2. Feature Engineering Module (`src/feature_engineering.py`)
- **Temporal Signals**: Cyclic seasonality modeled via sine/cosine transformations of the day of year, plus weekend/quarter/month indexes.
- **Attribution Efficiency**: Blends spend, clicks, impressions, and conversions into single, store-level CPC, CTR, and Conversion Rate indicators.
- **Autoregressive Features**: Shifts data to generate historical lags (1, 7, 14, and 30 days) and rolling trends (7 and 30-day means/standard deviations) to model momentum.

### 3. ML Ensemble & Probabilistic Layer (`src/models.py`)
- **Prophet Model**: Extracts global linear trend shifts and additive yearly, weekly, and holiday seasonal components.
- **Gradient Boosted Decision Trees (XGBoost & LightGBM)**: Models non-linear interactions between marketing spend channels, historical lags, and conversion rates.
- **Weighted Blending**: Predictions are combined: `0.3 * Prophet + 0.4 * XGBoost + 0.3 * LightGBM`.
- **Quantile Regressors (GradientBoostingRegressor with Quantile Loss)**: Fit to target features using `alpha=0.10` (P10) and `alpha=0.90` (P90) to generate risk-bound margins around the blended P50 point estimate.

### 4. Saturation Simulation Engine (`src/simulator.py`)
- **Diminishing Returns**: Relates spend per channel to sales attribution using logarithmic saturation curves: $Revenue = \alpha \cdot \ln(Spend + 2)$.
- **Automatic Calibration**: Calibrates saturation alphas dynamically based on historical conversions share per marketing channel.
- **Mix Optimization**: Models changes in spend distribution (mix), calculating expected revenue lifts and flagging saturation risks when scaling.

### 5. Generative AI & Reporting Layer (`src/llm_engine.py`, `src/report_generator.py`)
- **Contextual Prompts**: Compiles historical averages, forecasted targets, risk statuses, and simulator outcomes into a structured prompt.
- **Generative Report Writer**: Prompts LLMs (Gemini/OpenAI) to generate executive summaries, driver analyses, risks, and recommendations. Includes a dynamic offline fallback generator.
- **Print-to-PDF Exporter**: Formats predictions, risks, and insights into a clean, printable HTML report file utilizing responsive typography.
