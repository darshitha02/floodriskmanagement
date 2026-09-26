"""
Report Generator Module for FloodRiskAI
Generates printable HTML reports for Risk Assessments and Emergency Playbooks.
"""

from datetime import datetime

def generate_risk_report_html(pred_data, input_features, station_name="Indo-Floods Hydrological Gauge Station #402"):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    risk_level = pred_data.get("risk_level", "Medium")
    risk_score = pred_data.get("risk_score", 50.0)
    prediction = pred_data.get("prediction", "Moderate Flood")
    model_used = pred_data.get("model_used", "Random Forest")
    attributions = pred_data.get("attributions", [])

    color_map = {
        "Low": "#2ea86a",
        "Medium": "#e6ab1f",
        "High": "#e2483d"
    }
    badge_color = color_map.get(risk_level, "#0d6efd")

    attributions_html = ""
    for attr in attributions:
        direction_badge = f'<span style="color: {"#e2483d" if attr["direction"]=="increases" else "#2ea86a"}; font-weight:bold;">{attr["direction"].upper()}</span>'
        attributions_html += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #eee;">{attr['feature']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee;">{attr['value']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee;">{direction_badge} risk by ~{attr['impact_pct']}%</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Flood Risk Assessment Report - FloodRiskAI</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #1c2b3a; background: #fff; margin: 0; padding: 40px; }}
            .header {{ border-bottom: 3px solid #0d6efd; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }}
            .title {{ font-size: 24px; font-weight: bold; color: #0b2e4f; margin: 0; }}
            .subtitle {{ color: #6c7a89; font-size: 14px; margin-top: 4px; }}
            .badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; color: #fff; font-weight: bold; font-size: 14px; background: {badge_color}; }}
            .section-title {{ font-size: 18px; font-weight: bold; color: #0b2e4f; border-bottom: 1px solid #dee2e6; padding-bottom: 8px; margin-top: 30px; margin-bottom: 15px; }}
            .summary-box {{ background: #f8f9fa; border-left: 4px solid {badge_color}; padding: 20px; border-radius: 6px; margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background: #0b2e4f; color: #fff; text-align: left; padding: 10px; font-size: 14px; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #888; border-top: 1px solid #eee; padding-top: 15px; text-align: center; }}
            @media print {{ body {{ padding: 0; }} .no-print {{ display: none; }} }}
        </style>
    </head>
    <body>
        <div class="no-print" style="margin-bottom: 20px; text-align: right;">
            <button onclick="window.print()" style="background: #0d6efd; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: bold;">🖨️ Print / Save as PDF</button>
        </div>

        <div class="header">
            <div>
                <div class="title">🌊 FloodRiskAI — Official Risk Assessment</div>
                <div class="subtitle">DataQuest 2026 · Disaster Management Analytics | Generated: {now_str}</div>
            </div>
            <div class="badge">{risk_level.upper()} RISK ({risk_score}%)</div>
        </div>

        <div class="summary-box">
            <h3 style="margin-top:0; color:#0b2e4f;">Assessment Summary — {station_name}</h3>
            <p><strong>Predicted Event Class:</strong> {prediction}</p>
            <p><strong>AI Model Applied:</strong> {model_used}</p>
            <p><strong>Risk Index Score:</strong> {risk_score} / 100</p>
        </div>

        <div class="section-title">Hydrological Event Input Parameters</div>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Input Value</th>
                </tr>
            </thead>
            <tbody>
                {"".join([f'<tr><td style="padding:8px; border-bottom:1px solid #eee;">{k}</td><td style="padding:8px; border-bottom:1px solid #eee;">{v}</td></tr>' for k, v in input_features.items()])}
            </tbody>
        </table>

        <div class="section-title">SHAP Local Feature Contributions</div>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Attribution Impact</th>
                </tr>
            </thead>
            <tbody>
                {attributions_html}
            </tbody>
        </table>

        <div class="section-title">Recommended Emergency Action Plan</div>
        <div style="background:#eef6ff; padding:15px; border-radius:6px; font-weight:500;">
            {"Routine gauge monitoring and standard data logging." if risk_level == "Low" else ("Alert local disaster management authorities, increase gauge monitoring frequency to hourly intervals, and issue preliminary flood warnings." if risk_level == "Medium" else "IMMEDIATE EMERGENCY DISPATCH: Issue red alert to District Disaster Management Authority (DDMA), activate NDRF rescue teams, initiate evacuation along floodplains, and ready relief shelters.")}
        </div>

        <div class="footer">
            FloodRiskAI Disaster Management Platform · Vignan's Foundation for Science, Technology & Research · Official Assessment Report
        </div>
    </body>
    </html>
    """
    return html_content
