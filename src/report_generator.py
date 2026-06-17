import os

class ExecutiveReportGenerator:
    def __init__(self):
        pass

    def generate_html_report(self, metrics, df_predictions, risk_report, ai_insights):
        """
        Compiles all dashboard sections into a premium, printable HTML executive report.
        """
        # Style predictions table
        pred_rows = ""
        for i, row in df_predictions.head(15).iterrows():
            pred_rows += f"""
            <tr>
                <td>{row['Forecast_Period']} ({row['Date']})</td>
                <td>${row['Revenue_P10']:,.2f}</td>
                <td><strong>${row['Revenue_P50']:,.2f}</strong></td>
                <td>${row['Revenue_P90']:,.2f}</td>
                <td>{row['ROAS_P10']:.2f}x</td>
                <td><strong>{row['ROAS_P50']:.2f}x</strong></td>
                <td>{row['ROAS_P90']:.2f}x</td>
                <td><span class="badge badge-{row['Forecast_Confidence'].split(' ')[0].lower()}">{row['Forecast_Confidence']}</span></td>
            </tr>
            """
            
        risk_rows = ""
        for flag in risk_report.get("details", []):
            sev_class = "danger" if flag['severity'] == "High" else "warning"
            risk_rows += f"""
            <div class="risk-card">
                <div class="risk-header">
                    <span class="badge badge-{sev_class}">{flag['severity']}</span>
                    <strong>{flag['risk_name']}</strong>
                </div>
                <div class="risk-body">
                    <p>{flag['message']}</p>
                    <p><strong>Recommendation:</strong> {flag['recommendation']}</p>
                </div>
            </div>
            """
            
        if not risk_rows:
            risk_rows = "<p class='no-risks'>No critical marketing risks identified. Campaigns are running optimally.</p>"

        # Convert markdown-style AI insights to HTML paragraphs
        formatted_insights = ai_insights.replace("### ", "<h3>").replace("## ", "<h2>").replace("**", "<strong>")
        formatted_insights = formatted_insights.replace("\n* ", "<br>&bull; ").replace("\n", "<br>")

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Revenue Intelligence - Executive Report</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #333333;
            line-height: 1.6;
            margin: 0;
            padding: 40px;
            background-color: #ffffff;
        }}
        .header-container {{
            border-bottom: 3px solid #1e3a8a;
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .title-section h1 {{
            margin: 0;
            color: #1e3a8a;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }}
        .title-section p {{
            margin: 5px 0 0 0;
            color: #6b7280;
            font-size: 14px;
        }}
        .logo-section {{
            font-size: 18px;
            font-weight: 700;
            color: #10b981;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            text-align: left;
        }}
        .metric-card .label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .metric-card .value {{
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
        }}
        .metric-card .subtext {{
            font-size: 12px;
            color: #10b981;
            margin-top: 4px;
        }}
        .section-title {{
            color: #0f172a;
            font-size: 18px;
            font-weight: 700;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th {{
            background-color: #f1f5f9;
            color: #475569;
            text-align: left;
            padding: 12px;
            font-size: 13px;
            font-weight: 600;
            border-bottom: 2px solid #cbd5e1;
        }}
        td {{
            padding: 12px;
            font-size: 13px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-high {{ background-color: #d1fae5; color: #065f46; }}
        .badge-medium {{ background-color: #fef3c7; color: #92400e; }}
        .badge-low {{ background-color: #fee2e2; color: #991b1b; }}
        .badge-danger {{ background-color: #fee2e2; color: #991b1b; }}
        .badge-warning {{ background-color: #fef3c7; color: #92400e; }}
        .risk-card {{
            border-left: 4px solid #f59e0b;
            background-color: #fff9f2;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
        }}
        .risk-card:has(.badge-danger) {{
            border-left-color: #ef4444;
            background-color: #fef2f2;
        }}
        .risk-header {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 8px;
        }}
        .risk-body p {{
            margin: 4px 0;
            font-size: 13px;
        }}
        .insights-container {{
            background: #fafafa;
            border: 1px dashed #cccccc;
            padding: 25px;
            border-radius: 8px;
            font-size: 14px;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 12px;
            color: #94a3b8;
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
        }}
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
            .page-break {{ page-break-before: always; }}
        }}
    </style>
</head>
<body>
    <div class="header-container">
        <div class="title-section">
            <h1>AI Revenue Intelligence</h1>
            <p>E-commerce Forecasting & Decision Support Summary • Generated {metrics['current_time']}</p>
        </div>
        <div class="logo-section">NetElixir AIgnition 3.0</div>
    </div>

    <div class="grid-3">
        <div class="metric-card">
            <div class="label">Historical Blended ROAS</div>
            <div class="value">{metrics['hist_blended_roas']:.2f}x</div>
            <div class="subtext">Based on last 30 days of spend</div>
        </div>
        <div class="metric-card">
            <div class="label">90-Day Revenue Projection</div>
            <div class="value">${metrics['pred_revenue_90d']:,.2f}</div>
            <div class="subtext">P50 Ensemble Forecast</div>
        </div>
        <div class="metric-card">
            <div class="label">Consolidated Risk Level</div>
            <div class="value" style="color: { '#ef4444' if risk_report['overall_risk_level'] == 'High' else ('#f59e0b' if risk_report['overall_risk_level'] == 'Medium' else '#10b981') };">{risk_report['overall_risk_level']}</div>
            <div class="subtext">Consolidated Risk Score: {risk_report['risk_score']}</div>
        </div>
    </div>

    <div class="section-title">90-Day Probabilistic Forecast Summary (First 15 Days)</div>
    <table>
        <thead>
            <tr>
                <th>Forecast Period</th>
                <th>Revenue (P10)</th>
                <th>Revenue (P50)</th>
                <th>Revenue (P90)</th>
                <th>ROAS (P10)</th>
                <th>ROAS (P50)</th>
                <th>ROAS (P90)</th>
                <th>Confidence Level</th>
            </tr>
        </thead>
        <tbody>
            {pred_rows}
        </tbody>
    </table>

    <div class="page-break"></div>

    <div class="section-title">Marketing Risk Analysis</div>
    <div style="margin-bottom: 30px;">
        <p><strong>Overall Status:</strong> {risk_report['summary']}</p>
        {risk_rows}
    </div>

    <div class="section-title">AI Generative Insights Report</div>
    <div class="insights-container">
        {formatted_insights}
    </div>

    <div class="footer">
        Confidential Report • Generated by AI Revenue Intelligence Platform for NetElixir AIgnition 3.0. All projections are statistical probabilities.
    </div>
</body>
</html>
"""
        return html_template

    def save_report(self, html_content, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filepath
