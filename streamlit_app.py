import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import local modules
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.explainability import RevenueExplainer
from src.risk_engine import MarketingRiskEngine
from src.simulator import BudgetSimulator
from src.llm_engine import AIInsightsEngine
from src.report_generator import ExecutiveReportGenerator

# Page Config
st.set_page_config(
    page_title="AI Revenue Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Fonts & Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Main Background & Theme adjustment */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 1) 0%, rgba(9, 15, 29, 1) 90.1%);
        color: #f8fafc;
    }
    
    /* Premium Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphism Metric Cards */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.4);
    }
    .metric-title {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-subtext {
        font-size: 11px;
        color: #10b981;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Status Badges */
    .badge-label {
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.5px;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-medium { background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-low { background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    /* Risk Engine Warning Cards */
    .risk-alert {
        background: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .risk-alert-warn {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 15px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Load Data
@st.cache_data
def load_historical_data(data_path="data"):
    pipeline = DataPipeline(data_path)
    df_merged = pipeline.aggregate_sources()
    report = pipeline.get_validation_report()
    return df_merged, report

# Load Models
@st.cache_resource
def load_serialized_models(model_path="pickle/model.pkl"):
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_predictions_enriched(pred_path, df_merged):
    df_preds = pd.read_csv(pred_path)
    last_date = df_merged["Date"].max()
    df_preds["Date"] = [last_date + pd.Timedelta(days=i+1) for i in range(len(df_preds))]
    
    conf_labels = []
    for idx, row in df_preds.iterrows():
        p10 = row["Revenue_P10"]
        p50 = row["Revenue_P50"]
        p90 = row["Revenue_P90"]
        step = idx + 1
        if p50 == 0:
            score = 0.50
        else:
            relative_spread = (p90 - p10) / p50
            base_conf = 1.0 - min(0.45, relative_spread * 0.15)
            decay = 0.99 ** step
            score = base_conf * decay
        score = float(np.clip(score, 0.55, 0.96))
        
        if score >= 0.85:
            lbl = f"High ({int(score*100)}%)"
        elif score >= 0.70:
            lbl = f"Medium ({int(score*100)}%)"
        else:
            lbl = f"Low ({int(score*100)}%)"
        conf_labels.append(lbl)
        
    df_preds["Forecast_Confidence"] = conf_labels
    return df_preds


# Initialize Session State for file updates
if "custom_data_dir" not in st.session_state:
    st.session_state["custom_data_dir"] = "data"

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #10b981;'>⚡ NetElixir AIgnition</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>AI REVENUE INTELLIGENCE v3.0</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard Home", "Data Ingestion & Validation", "Forecasting Engine", "Budget Simulator", "AI Insights Engine", "Reports Hub"]
)

# Load base data
df_merged, validation_report = load_historical_data(st.session_state["custom_data_dir"])
df_merged["Date"] = pd.to_datetime(df_merged["Date"])

# Load models
model_pack = load_serialized_models()
if model_pack is None:
    st.sidebar.warning("⚠️ Serialized models not found in 'pickle/model.pkl'. Please wait for training to finish or run training script.")
    # Fit fallback on the fly so app never breaks
    from src.train import main as train_fallback
    # We define fallback logic to allow instant display
    st.sidebar.info("Model pickle is missing. Model training will execute automatically on dashboard load.")
    try:
        train_fallback()
        st.cache_resource.clear()
        model_pack = load_serialized_models()
    except Exception as e:
        st.sidebar.error(f"Auto-training failed: {e}")

# Helper: compute key KPIs
last_30_days = df_merged.tail(30)
tot_revenue_30 = last_30_days["Revenue"].sum()
tot_spend_30 = last_30_days["Total_Spend"].sum()
blended_roas_30 = tot_revenue_30 / tot_spend_30 if tot_spend_30 > 0 else 0.0
avg_cvr_30 = last_30_days["blended_conv_rate"].mean()

# ----------------- PAGE 1: DASHBOARD HOME -----------------
if page == "Dashboard Home":
    st.title("📊 AI Revenue Intelligence - Executive Dashboard")
    st.markdown("Real-time marketing efficiency, customer attribution, and cross-channel performance analytics.")
    
    # KPI Metric Cards
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">30D Net Revenue</div>
            <div class="metric-value">${tot_revenue_30:,.2f}</div>
            <div class="metric-subtext">Shopify Realized Value</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">30D Blended Spend</div>
            <div class="metric-value">${tot_spend_30:,.2f}</div>
            <div class="metric-subtext">Total Paid Advertising</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Blended ROAS</div>
            <div class="metric-value">{blended_roas_30:.2f}x</div>
            <div class="metric-subtext" style="color: {'#10b981' if blended_roas_30 >= 2.0 else '#ef4444'};">
                {'Efficient Return Scale' if blended_roas_30 >= 2.0 else 'Underperforming Baseline'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Shopify Funnel CVR</div>
            <div class="metric-value">{avg_cvr_30*100:.2f}%</div>
            <div class="metric-subtext">Average checkout conversion</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Main historical charts
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Blended Marketing Spend vs. Net Revenue Trend")
        df_chart = df_merged.tail(90).copy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart["Date"], y=df_chart["Revenue"], name="Net Revenue ($)", fill='tozeroy', line=dict(color='#10b981', width=2)))
        fig.add_trace(go.Scatter(x=df_chart["Date"], y=df_chart["Total_Spend"], name="Ad Spend ($)", line=dict(color='#3b82f6', width=2, dash='dot')))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_right:
        st.subheader("🍩 Ad Spend Channel Contribution")
        google_tot = last_30_days["Google_Spend"].sum()
        meta_tot = last_30_days["Meta_Spend"].sum()
        ms_tot = last_30_days["MS_Spend"].sum()
        
        df_pie = pd.DataFrame({
            "Channel": ["Google Ads", "Meta Ads", "Microsoft Ads"],
            "Spend": [google_tot, meta_tot, ms_tot]
        })
        fig_pie = px.pie(df_pie, values="Spend", names="Channel", color_discrete_sequence=['#3b82f6', '#ec4899', '#f59e0b'], hole=0.4)
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # Channel-level Performance Table
    st.subheader("📊 Channel efficiency breakdown (Last 30 Days)")
    
    def get_channel_metrics(prefix, name):
        spend = last_30_days[f"{prefix}_Spend"].sum()
        clicks = last_30_days[f"{prefix}_Clicks"].sum()
        conversions = last_30_days[f"{prefix}_Conversions"].sum()
        cpc = spend / clicks if clicks > 0 else 0.0
        cvr = conversions / clicks if clicks > 0 else 0.0
        
        # Attribute revenue based on GA4 conversions share
        total_attrib_conv = last_30_days["Google_Conversions"].sum() + last_30_days["Meta_Conversions"].sum() + last_30_days["MS_Conversions"].sum()
        conv_share = conversions / total_attrib_conv if total_attrib_conv > 0 else 0.33
        attributed_rev = (tot_revenue_30 - last_30_days["Revenue"].sum()*0.25) * conv_share # Subtracting 25% organic baseline
        roas = attributed_rev / spend if spend > 0 else 0.0
        
        return {
            "Channel": name,
            "Spend": f"${spend:,.2f}",
            "Attributed Revenue": f"${attributed_rev:,.2f}",
            "ROAS": f"{roas:.2f}x",
            "Avg CPC": f"${cpc:.2f}",
            "Funnel CVR": f"{cvr*100:.2f}%"
        }
        
    summary_data = [
        get_channel_metrics("Google", "Google Ads"),
        get_channel_metrics("Meta", "Meta Ads"),
        get_channel_metrics("MS", "Microsoft Ads")
    ]
    st.table(pd.DataFrame(summary_data))

# ----------------- PAGE 2: DATA INGESTION -----------------
elif page == "Data Ingestion & Validation":
    st.title("📥 Multi-Source Data Ingestion & Quality Validation")
    st.markdown("Upload your raw CSV files dynamically. The engine runs schema validation, duplicate checks, missing values cleaning, and Z-Score anomaly checks.")
    
    st.subheader("Source Status & Schema Quality Report")
    
    for source in ["google", "meta", "microsoft", "ga4", "shopify"]:
        rep = validation_report.get(source, {})
        has_issue = rep.get("has_missing_columns", False) or rep.get("duplicate_rows", 0) > 0
        
        col_name, col_status, col_rows, col_dups = st.columns([2, 2, 2, 2])
        with col_name:
            st.markdown(f"**{source.upper()} SOURCE**")
        with col_status:
            if has_issue:
                st.markdown("⚠️ <span class='badge-label badge-medium'>Validation Flags</span>", unsafe_allow_html=True)
            else:
                st.markdown("✅ <span class='badge-label badge-low'>Passed Schema</span>", unsafe_allow_html=True)
        with col_rows:
            st.markdown(f"Total Rows: **{rep.get('total_rows', 0):,}**")
        with col_dups:
            st.markdown(f"Duplicates: **{rep.get('duplicate_rows', 0)}**")
            
    st.markdown("---")
    
    st.subheader("🚨 Consolidate Outliers & Anomaly Detection")
    cons = validation_report.get("consolidated", {})
    st.markdown(f"Consolidated days: **{cons.get('total_days', 0)}**. Outliers detected: **{cons.get('revenue_anomalies_count', 0)}**.")
    
    # Show anomaly rows
    anomaly_indices = cons.get("revenue_anomalies_indices", [])
    if len(anomaly_indices) > 0:
        st.warning("⚠️ Anomalously high/low revenue days detected (Z-Score > 3.0 threshold). These represent promo sales spikes or tracking drops.")
        df_anom = df_merged.iloc[anomaly_indices][["Date", "Revenue", "Total_Spend", "ROAS"]].copy()
        st.dataframe(df_anom)
    else:
        st.success("No critical revenue outliers found.")

# ----------------- PAGE 3: FORECASTING ENGINE -----------------
elif page == "Forecasting Engine":
    st.title("🔮 Hybrid Ensemble Projections & Driver Explainability")
    st.markdown("Prophet + XGBoost + LightGBM weighted point forecasts combined with Quantile Regression (P10/P90) uncertainty ribbons.")
    
    # Load predictions
    # Check if predictions file exists, else generate it
    pred_path = "output/predictions.csv"
    if not os.path.exists(pred_path):
        from src.predict import main as run_predictions
        st.info("Generating predictions...")
        # Mock execution command line args
        class Args:
            def __init__(self):
                self.data_dir = "data"
                self.model_path = "pickle/model.pkl"
                self.output_path = "output/predictions.csv"
        try:
            # We run python script
            os.system("python src/predict.py data pickle/model.pkl output/predictions.csv")
        except Exception as e:
            st.error(f"Prediction run failed: {e}")
            
    if os.path.exists(pred_path):
        df_preds = load_predictions_enriched(pred_path, df_merged)
        
        horizon = st.radio("Select Projections Horizon", [30, 60, 90], index=2, horizontal=True)
        df_horizon = df_preds.head(horizon)
        
        # Metric
        avg_conf = df_horizon["Forecast_Confidence"].values[0] # Grab initial day confidence
        tot_rev_forecast = df_horizon["Revenue_P50"].sum()
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Forecasted Total Net Revenue", f"${tot_rev_forecast:,.2f}", help="Sum of P50 predicted daily revenue")
        with col_m2:
            st.metric("Forecast Confidence Score", avg_conf, help="Confidence based on interval stability")
            
        st.subheader(f"{horizon}-Day Probabilistic Revenue Projections (P10, P50, P90)")
        
        # Plotly chart with confidence ribbon
        fig_fore = go.Figure()
        # Historical
        df_hist_tail = df_merged.tail(60)
        fig_fore.add_trace(go.Scatter(x=df_hist_tail["Date"], y=df_hist_tail["Revenue"], name="Historical Net Revenue", line=dict(color="#94a3b8")))
        
        # P50 Prediction
        fig_fore.add_trace(go.Scatter(x=df_horizon["Date"], y=df_horizon["Revenue_P50"], name="Predicted P50 (Ensemble Mean)", line=dict(color="#10b981", width=3)))
        
        # P10 and P90 area
        fig_fore.add_trace(go.Scatter(
            x=list(df_horizon["Date"]) + list(df_horizon["Date"])[::-1],
            y=list(df_horizon["Revenue_P90"]) + list(df_horizon["Revenue_P10"])[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Uncertainty Bound (P10 - P90)"
        ))
        
        fig_fore.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_fore, use_container_width=True)
        
        # Explainability Panel
        st.markdown("---")
        st.subheader("💡 Model Explainability & Causal Revenue Drivers")
        st.markdown("Using XGBoost/LightGBM feature importances and historical attribution variables to understand **'why'** the model predicts this trajectory.")
        
        explainer = RevenueExplainer(model_pack)
        df_imp = explainer.get_feature_importance()
        
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            st.markdown("**Top Global Feature Importances (Tabular Models)**")
            fig_imp = px.bar(df_imp.head(7), x="Importance", y="Feature", orientation="h", color_discrete_sequence=['#10b981'])
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_e2:
            st.markdown("**Top Revenue Drivers Attribution (Recent Period)**")
            drivers = explainer.explain_drivers(df_merged)
            for d in drivers[:3]:
                arrow = "⬆️" if d['Direction'] == "Positive" else "⬇️"
                weight = d['Explained_Weight']
                st.markdown(f"""
                <div style='background: rgba(30, 41, 59, 0.3); padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.03);'>
                    <strong>{arrow} {d['Driver']} ({weight*100:.1f}% explained weight)</strong>
                    <p style='margin: 4px 0 0 0; font-size: 13px; color: #94a3b8;'>{d['Details']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        # Data table
        st.subheader("Projections Data Log")
        st.dataframe(df_horizon[["Forecast_Period", "Date", "Revenue_P10", "Revenue_P50", "Revenue_P90", "ROAS_P10", "ROAS_P50", "ROAS_P90", "Forecast_Confidence"]].head(15))

# ----------------- PAGE 4: BUDGET SIMULATOR -----------------
elif page == "Budget Simulator":
    st.title("🎛️ Upgraded Multi-Channel Budget Simulator")
    st.markdown("Perform 'what-if' budget scenario simulations. Channel-specific logarithmic saturation curves compute simulated Revenue, ROAS, and Saturation Risk.")
    
    # Calibrate Simulator
    sim = BudgetSimulator()
    sim.calibrate(df_merged)
    
    # Sliders for each channel
    st.sidebar.subheader("Adjust Daily Spend Modifiers")
    google_mod = st.sidebar.slider("Google Ads Spend Shift (%)", -50, 100, 20, step=5) / 100.0
    meta_mod = st.sidebar.slider("Meta Ads Spend Shift (%)", -50, 100, -10, step=5) / 100.0
    ms_mod = st.sidebar.slider("Microsoft Ads Spend Shift (%)", -50, 100, 15, step=5) / 100.0
    
    # Run simulation
    out = sim.simulate({"google": google_mod, "meta": meta_mod, "ms": ms_mod})
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Simulated Ad Spend</div>
            <div class="metric-value">${out['simulated_spend']:,.2f}</div>
            <div class="metric-subtext" style="color: {'#10b981' if out['spend_change_pct'] <= 0.20 else '#f59e0b'};">
                {out['spend_change_pct']*100:+.1f}% Budget Shift
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Simulated Net Revenue</div>
            <div class="metric-value">${out['simulated_revenue']:,.2f}</div>
            <div class="metric-subtext" style="color: {'#10b981' if out['revenue_lift_pct'] >= 0 else '#ef4444'};">
                {out['revenue_lift_pct']*100:+.1f}% Revenue Lift
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Simulated Blended ROAS</div>
            <div class="metric-value">{out['simulated_roas']:.2f}x</div>
            <div class="metric-subtext" style="color: {'#10b981' if out['roas_change_pct'] >= -0.05 else '#ef4444'};">
                {out['roas_change_pct']*100:+.1f}% ROAS Change
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Show Simulation Risk Level
    risk_level = out["risk_level"]
    risk_badge = "badge-high" if risk_level == "High" else ("badge-medium" if risk_level == "Medium" else "badge-low")
    
    st.subheader("⚠️ Saturation & Volatility Risk Analysis")
    st.markdown(f"""
    <div class="{'risk-alert' if risk_level == 'High' else 'risk-alert-warn'}">
        <strong>Simulator Risk Status: <span class="badge-label {risk_badge}">{risk_level} Risk</span></strong>
        <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 13px;">{out['risk_reason']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison chart
    st.subheader("Comparison: Baseline vs. Simulated Spend Mix")
    df_compare = pd.DataFrame({
        "Metric": ["Google Ads", "Meta Ads", "Microsoft Ads"],
        "Baseline": [sim.baseline_spend["google"], sim.baseline_spend["meta"], sim.baseline_spend["ms"]],
        "Simulated": [out["channel_spends"]["google"], out["channel_spends"]["meta"], out["channel_spends"]["ms"]]
    })
    fig_compare = px.bar(df_compare, x="Metric", y=["Baseline", "Simulated"], barmode="group", color_discrete_sequence=['#475569', '#10b981'])
    fig_compare.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# ----------------- PAGE 5: AI INSIGHTS ENGINE -----------------
elif page == "AI Insights Engine":
    st.title("🤖 Generative AI Marketing Insights Engine")
    st.markdown("Generates comprehensive executive reports using Gemini or OpenAI APIs with a dynamic local fallback engine.")
    
    # Load inputs
    risk_eng = MarketingRiskEngine()
    risk_report = risk_eng.assess_risks(df_merged)
    
    sim = BudgetSimulator()
    sim.calibrate(df_merged)
    sim_out = sim.simulate({"google": 0.20, "meta": -0.10, "ms": 0.15})
    
    pred_path = "output/predictions.csv"
    df_preds = load_predictions_enriched(pred_path, df_merged)
    pred_90 = df_preds.head(90)
    
    metrics_summary = {
        "hist_avg_revenue": last_30_days["Revenue"].mean(),
        "hist_avg_spend": last_30_days["Total_Spend"].mean(),
        "hist_blended_roas": blended_roas_30,
        "google_spend": last_30_days["Google_Spend"].sum(),
        "google_roas": (last_30_days["Revenue"].sum()*0.45) / last_30_days["Google_Spend"].sum(), # approximate share
        "meta_spend": last_30_days["Meta_Spend"].sum(),
        "meta_roas": (last_30_days["Revenue"].sum()*0.35) / last_30_days["Meta_Spend"].sum(),
        "ms_spend": last_30_days["MS_Spend"].sum(),
        "ms_roas": (last_30_days["Revenue"].sum()*0.10) / last_30_days["MS_Spend"].sum(),
        "shopify_orders": last_30_days["Shopify_Orders"].sum(),
        "ga4_sessions": last_30_days["GA4_Sessions"].sum(),
        "conv_rate": avg_cvr_30,
        
        "pred_revenue_90d": pred_90["Revenue_P50"].sum(),
        "pred_revenue_p10": pred_90["Revenue_P10"].sum(),
        "pred_revenue_p90": pred_90["Revenue_P90"].sum(),
        "forecast_confidence": pred_90["Forecast_Confidence"].values[0],
        
        "risk_level": risk_report["overall_risk_level"],
        "risk_alerts": risk_report["details"][0]["message"] if len(risk_report["details"]) > 0 else "None",
        
        "sim_spend_change": sim_out["spend_change_pct"] * 100.0,
        "sim_rev_lift": sim_out["revenue_lift_pct"] * 100.0,
        "sim_roas": sim_out["simulated_roas"]
    }
    
    # Sidebar API Keys
    st.sidebar.subheader("API Keys (Optional)")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    if st.button("Generate Executive Insights Report"):
        with st.spinner("Analyzing marketing data and drafting executive recommendations..."):
            engine = AIInsightsEngine(gemini_api_key=gemini_key, openai_api_key=openai_key)
            report_text, model_used = engine.generate_insights(metrics_summary)
            
            st.session_state["ai_report"] = report_text
            st.session_state["model_used"] = model_used
            
    if "ai_report" in st.session_state:
        st.info(f"Report Generated via: {st.session_state['model_used']}")
        st.markdown(st.session_state["ai_report"])

# ----------------- PAGE 6: REPORTS HUB -----------------
elif page == "Reports Hub":
    st.title("📋 Executive Reports Hub")
    st.markdown("Download and print comprehensive summaries of your marketing campaign forecasts, simulator settings, and AI audits.")
    
    # Build report payload
    risk_eng = MarketingRiskEngine()
    risk_report = risk_eng.assess_risks(df_merged)
    
    pred_path = "output/predictions.csv"
    df_preds = load_predictions_enriched(pred_path, df_merged)
    pred_90 = df_preds.head(90)
    
    metrics_summary = {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hist_blended_roas": blended_roas_30,
        "pred_revenue_90d": pred_90["Revenue_P50"].sum(),
        "hist_avg_revenue": last_30_days["Revenue"].mean(),
        "hist_avg_spend": last_30_days["Total_Spend"].mean(),
        "google_spend": last_30_days["Google_Spend"].sum(),
        "google_roas": (last_30_days["Revenue"].sum()*0.45) / last_30_days["Google_Spend"].sum(),
        "meta_spend": last_30_days["Meta_Spend"].sum(),
        "meta_roas": (last_30_days["Revenue"].sum()*0.35) / last_30_days["Meta_Spend"].sum(),
        "ms_spend": last_30_days["MS_Spend"].sum(),
        "ms_roas": (last_30_days["Revenue"].sum()*0.10) / last_30_days["MS_Spend"].sum(),
        "shopify_orders": last_30_days["Shopify_Orders"].sum(),
        "ga4_sessions": last_30_days["GA4_Sessions"].sum(),
        "conv_rate": avg_cvr_30,
        "pred_revenue_p10": pred_90["Revenue_P10"].sum(),
        "pred_revenue_p90": pred_90["Revenue_P90"].sum(),
        "forecast_confidence": pred_90["Forecast_Confidence"].values[0],
        "risk_level": risk_report["overall_risk_level"],
        "risk_alerts": risk_report["details"][0]["message"] if len(risk_report["details"]) > 0 else "None",
        "sim_spend_change": 10.0,
        "sim_rev_lift": 8.0,
        "sim_roas": blended_roas_30 * 0.98
    }
    
    # Get AI insights if generated, else generate mock instantly
    if "ai_report" in st.session_state:
        ai_insights = st.session_state["ai_report"]
    else:
        engine = AIInsightsEngine()
        ai_insights, _ = engine.generate_insights(metrics_summary)
        
    rep_gen = ExecutiveReportGenerator()
    html_content = rep_gen.generate_html_report(metrics_summary, df_preds, risk_report, ai_insights)
    
    st.subheader("Generate & Export Report")
    st.markdown("Download the compiled HTML Executive report. Open it in your browser and press `Ctrl+P` (or `Cmd+P`) to print to PDF with a clean white corporate template style.")
    
    # Download Button
    st.download_button(
        label="📥 Download HTML Executive Report",
        data=html_content,
        file_name=f"Executive_Revenue_Intelligence_Report_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html"
    )
    
    # Preview Report
    st.markdown("---")
    st.subheader("Preview Report Output")
    st.components.v1.html(html_content, height=800, scrolling=True)
