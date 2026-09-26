"""
FloodRiskAI — Streamlit Disaster Intelligence & Location Risk Predictor
DataQuest 2026 · Disaster Management Analytics
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from ml_engine import ml_engine, FEATURE_META, FEATURE_COLS
from location_engine import search_locations
from report_generator import generate_risk_report_html

# Page Config
st.set_page_config(
    page_title="FloodRiskAI — Location Search & Risk Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme CSS
st.markdown("""
<style>
    .main { background: #071927; color: #e2e8f0; }
    .stApp { background-color: #071927; }
    .hero-banner-st {
        background: linear-gradient(135deg, #071927 0%, #0d2a45 40%, #0f3d66 100%);
        padding: 2.2rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 210, 255, 0.2);
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-title-st { font-size: 2.4rem; font-weight: 700; color: #ffffff; font-family: 'Poppins', sans-serif; }
    .hero-sub-st { color: #94a3b8; font-size: 1.05rem; }
    .card-stat {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .stat-val-st { font-size: 2.2rem; font-weight: 700; color: #00d2ff; }
    .stat-lbl-st { font-size: 0.78rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_data
def load_data():
    flood_path = os.path.join(DATA_DIR, "floodevents_indofloods.csv")
    inv_path = os.path.join(DATA_DIR, "India_Flood_Inventory_v3.csv")

    df_flood = pd.read_csv(flood_path)
    df_flood["Start Date"] = pd.to_datetime(df_flood["Start Date"], format="%d-%m-%Y", errors="coerce")
    df_flood.dropna(subset=["Start Date"], inplace=True)
    df_flood["Year"] = df_flood["Start Date"].dt.year

    df_inv = pd.read_csv(inv_path)
    df_inv["Main Cause"] = df_inv["Main Cause"].fillna("Unknown")

    return df_flood, df_inv

df_flood, df_inv = load_data()

STATE_COORDS = {
    "Maharashtra": (19.7515, 75.7139), "Assam": (26.2006, 92.9376), "Kerala": (10.8505, 76.2711),
    "Karnataka": (15.3173, 75.7139), "Uttar Pradesh": (26.8467, 80.9462), "Himachal Pradesh": (31.1048, 77.1734),
    "West Bengal": (22.9868, 87.8550), "Rajasthan": (27.0238, 74.2179), "Madhya Pradesh": (22.9734, 78.6569),
    "Tamil Nadu": (11.1271, 78.6569), "Bihar": (25.0961, 85.3131), "Odisha": (20.9517, 85.0985),
    "Andhra Pradesh": (15.9129, 79.7400), "Gujarat": (22.2587, 71.1924)
}

RIVER_STATIONS = [
    {"station": "Guwahati Station", "river": "Brahmaputra", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "current": 50.45, "danger": 49.68, "status": "CRITICAL"},
    {"station": "Patna Gauge", "river": "Ganga", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "current": 49.20, "danger": 50.00, "status": "WARNING"},
    {"station": "Mahanadi Delta", "river": "Mahanadi", "state": "Odisha", "lat": 20.4625, "lon": 85.8828, "current": 27.15, "danger": 26.50, "status": "CRITICAL"},
    {"station": "Rajahmundry Barrage", "river": "Godavari", "state": "Andhra Pradesh", "lat": 17.0005, "lon": 81.8040, "current": 13.90, "danger": 14.50, "status": "WARNING"},
    {"station": "Idukki Gauge", "river": "Periyar", "state": "Kerala", "lat": 9.8497, "lon": 76.9813, "current": 34.80, "danger": 38.00, "status": "NORMAL"}
]

st.sidebar.markdown("# 🌊 FloodRiskAI")
st.sidebar.caption("DataQuest 2026 · Disaster Intelligence Platform")

page = st.sidebar.radio(
    "Navigation Mode",
    [
        "🔍 Search Location & Predict",
        "📊 Command Center Dashboard",
        "🗺️ Interactive GIS Risk Map",
        "🧠 AI Risk Predictor & SHAP",
        "🎛️ Climate Stress Simulator",
        "🚨 Emergency Alert & Playbook",
        "🧪 Multi-Model ML Benchmarks",
        "⚡ Developer API Specs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("✅ **Location Search & AI Models Active**")

# -----------------------------------------------------------------------------
# PAGE: SEARCH LOCATION & PREDICT
# -----------------------------------------------------------------------------
if page == "🔍 Search Location & Predict":
    st.markdown("""
    <div class="hero-banner-st">
        <div class="hero-title-st">Search Location & Predict Flood Risk</div>
        <div class="hero-sub-st">Type any Indian city, district, or state to predict location-specific flood severity & SHAP attributions.</div>
    </div>
    """, unsafe_allow_html=True)

    loc_query = st.text_input("📍 Search Any Place, City, or District in India", "Guwahati", help="Type Guwahati, Wayanad, Patna, Kochi, Cuttack, Mumbai, Assam...")

    if loc_query:
        loc_results = search_locations(loc_query)
        if loc_results:
            top_loc = loc_results[0]
            st.markdown(f"### 📍 Flood Risk Analysis for: **{top_loc['name']}, {top_loc['state']}**")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Location Risk Score", f"{top_loc['risk_score']}%", top_loc['risk_level'].upper() + " RISK")
            mc2.metric("Predicted Severity Class", top_loc['prediction'], f"River: {top_loc['river']}")
            mc3.metric("Peak Discharge (cumec)", f"{top_loc['peak_discharge']:,}", top_loc['cause'])

            st.markdown("---")

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("##### 🎛️ Location Hydrological Telemetry Profile")
                st.write(f"• **Peak Discharge:** {top_loc['peak_discharge']:,} cumec")
                st.write(f"• **Cumulative Volume:** {top_loc['flood_volume']:,} cumec")
                st.write(f"• **Monsoon Duration:** {top_loc['event_duration']} days")
                st.write(f"• **Surge Wave Count:** {top_loc['num_peak_fl']} peaks")
                st.write(f"• **Primary Cause:** {top_loc['cause']}")

            with col_r:
                st.markdown("##### 📊 Top Contributing Risk Factors")
                factors_df = pd.DataFrame(top_loc.get("top_factors", []))
                if not factors_df.empty:
                    st.table(factors_df[["feature", "value", "impact_pct", "direction"]])

# -----------------------------------------------------------------------------
# COMMAND CENTER DASHBOARD
# -----------------------------------------------------------------------------
elif page == "📊 Command Center Dashboard":
    st.markdown("""
    <div class="hero-banner-st">
        <div class="hero-title-st">Next-Generation AI Disaster Intelligence Platform</div>
        <div class="hero-sub-st">Real-time geospatial risk mapping, multi-model machine learning predictions, and emergency alert dispatch.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card-stat'><div class='stat-val-st'>{len(df_flood):,}</div><div class='stat-lbl-st'>Gauge Flood Events</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card-stat'><div class='stat-val-st'>{len(df_inv):,}</div><div class='stat-lbl-st'>Inventory Records</div></div>", unsafe_allow_html=True)
    with c3:
        sev_pct = round((df_flood['Flood Type'] == 'Severe Flood').mean() * 100, 1)
        st.markdown(f"<div class='card-stat'><div class='stat-val-st' style='color:#f59e0b;'>{sev_pct}%</div><div class='stat-lbl-st'>Historically Severe</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card-stat'><div class='stat-val-st' style='color:#10b981;'>82.5%</div><div class='stat-lbl-st'>ML Model Accuracy</div></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([7, 5])
    with col_l:
        st.subheader("📈 Historical Flood Event Frequency (India)")
        year_counts = df_flood["Year"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Events"]
        fig_year = px.line(year_counts, x="Year", y="Events", markers=True, color_discrete_sequence=["#00d2ff"], template="plotly_dark")
        fig_year.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_year, width="stretch")

    with col_r:
        st.subheader("🍩 Severity Class Distribution")
        type_counts = df_flood["Flood Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig_type = px.pie(type_counts, names="Type", values="Count", hole=0.5, color_discrete_sequence=["#00d2ff", "#ef4444"], template="plotly_dark")
        fig_type.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_type, width="stretch")

# -----------------------------------------------------------------------------
# INTERACTIVE GIS MAP
# -----------------------------------------------------------------------------
elif page == "🗺️ Interactive GIS Risk Map":
    st.subheader("🗺️ Interactive India GIS Flood Risk Map")
    st.caption("State-level risk heatmaps & active river basin telemetry stations")

    m = folium.Map(location=[22.5937, 78.9629], zoom_start=5, tiles="CartoDB dark_matter")

    state_agg = df_inv["State"].astype(str).str.split(",").explode().str.strip().value_counts().to_dict()
    for state, count in state_agg.items():
        if state in STATE_COORDS:
            lat, lon = STATE_COORDS[state]
            color = "#ef4444" if count > 500 else ("#f59e0b" if count > 200 else "#00d2ff")
            radius = min(45000, 15000 + count * 35)
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                popup=f"<b>{state}</b><br>Historical Events: {count}"
            ).add_to(m)

    for stn in RIVER_STATIONS:
        stn_color = "red" if stn["status"] == "CRITICAL" else ("orange" if stn["status"] == "WARNING" else "green")
        folium.Marker(
            location=[stn["lat"], stn["lon"]],
            popup=f"<b>{stn['station']} ({stn['river']})</b><br>Current: {stn['current']}m | Danger: {stn['danger']}m<br>Status: <b>{stn['status']}</b>",
            icon=folium.Icon(color=stn_color, icon="info-sign")
        ).add_to(m)

    st_folium(m, width=1100, height=580)

# -----------------------------------------------------------------------------
# AI RISK PREDICTOR & SHAP
# -----------------------------------------------------------------------------
elif page == "🧠 AI Risk Predictor & SHAP":
    st.subheader("🧠 AI Flood Risk Calculator & Feature Attributions")

    col_in, col_out = st.columns([5, 7])
    with col_in:
        st.markdown("##### 🎛️ Input Event Telemetry")
        model_name = st.selectbox("Select Model Architecture", list(ml_engine.models.keys()))

        inputs = {}
        for meta in FEATURE_META:
            inputs[meta["key"]] = st.slider(
                meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=meta["default"],
                step=meta["step"],
                help=meta["help"]
            )

    with col_out:
        st.markdown("##### 📊 Prediction & SHAP Waterfalls")
        row = [inputs[meta["key"]] for meta in FEATURE_META]
        pred = ml_engine.predict(row, model_name=model_name)

        score = pred["risk_score"]
        level = pred["risk_level"]
        color_map = {"Low": "green", "Medium": "orange", "High": "red"}

        st.markdown(f"### Risk Score: :{color_map[level]}[{score}% ({level.upper()} RISK)]")
        st.write(f"**Predicted Severity:** {pred['prediction']} | **Model:** {pred['model_used']}")

        recommendation_map = {
            "Low": "Routine monitoring. Standard operational gauge readings.",
            "Medium": "Notify regional disaster management authority. Increase monitoring frequency to hourly intervals.",
            "High": "CRITICAL EMERGENCY ALERT: Dispatch notification to SDMA/DDMA, mobilize NDRF rescue squads, and initiate floodplain evacuation."
        }
        st.info(f"📋 **Action Protocol:** {recommendation_map[level]}")

        st.markdown("##### 🔍 SHAP Local Feature Impact Waterfall")
        attr_df = pd.DataFrame(pred["attributions"])
        if not attr_df.empty:
            fig_attr = px.bar(
                attr_df,
                x="impact_pct",
                y="feature",
                color="direction",
                orientation="h",
                color_discrete_map={"increases": "#ef4444", "reduces": "#10b981"},
                template="plotly_dark",
                labels={"impact_pct": "Impact (%)", "feature": "Feature"}
            )
            fig_attr.update_layout(height=320, yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_attr, width="stretch")

        report_html = generate_risk_report_html(pred, inputs)
        st.download_button("🖨️ Download Official Assessment Report", data=report_html, file_name="flood_risk_report.html", mime="text/html")

# -----------------------------------------------------------------------------
# CLIMATE STRESS SIMULATOR
# -----------------------------------------------------------------------------
elif page == "🎛️ Climate Stress Simulator":
    st.subheader("🎛️ Climate Scenario Stress Simulator")

    c1, c2 = st.columns([5, 7])
    with c1:
        st.markdown("##### Climate Perturbation Factors")
        d_mult = st.slider("Peak River Discharge Multiplier", 0.5, 2.5, 1.3, 0.1)
        v_mult = st.slider("Flood Volume Multiplier", 0.5, 3.0, 1.2, 0.1)
        dur_add = st.slider("Extended Storm Duration (+Days)", 0, 15, 3, 1)

    with c2:
        st.markdown("##### Baseline vs. Stressed Risk Comparison")
        base_row = [1, 1175, 2207, 2, 1, 2, 7, 2]
        modifiers = {"discharge_mult": d_mult, "volume_mult": v_mult, "duration_add": dur_add}
        sim_res = ml_engine.simulate_scenario(base_row, modifiers)

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Baseline Risk", f"{sim_res['baseline']['risk_score']}%", sim_res['baseline']['risk_level'])
        sc2.metric("Stressed Risk", f"{sim_res['stressed']['risk_score']}%", sim_res['stressed']['risk_level'])
        sc3.metric("Delta Surge", f"+{sim_res['delta_risk']}%", "Net Risk Delta")

        curve_df = pd.DataFrame(sim_res["discharge_curve"])
        fig_sens = px.line(curve_df, x="discharge", y="risk_score", title="Discharge Threshold Sensitivity Curve", labels={"discharge": "Peak Discharge (cumec)", "risk_score": "Risk Score (%)"}, color_discrete_sequence=["#ef4444"], template="plotly_dark")
        fig_sens.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sens, width="stretch")

# -----------------------------------------------------------------------------
# EMERGENCY ALERTS & PLAYBOOKS
# -----------------------------------------------------------------------------
elif page == "🚨 Emergency Alert & Playbook":
    st.subheader("🚨 Live River Telemetry Alerts & Response Dispatch")

    for stn in RIVER_STATIONS:
        with st.expander(f"{'🔴' if stn['status']=='CRITICAL' else '🟡'} {stn['station']} ({stn['river']} River) — Status: {stn['status']}"):
            st.write(f"**State:** {stn['state']} | **Current Level:** {stn['current']}m | **Danger Level:** {stn['danger']}m")
            if stn['status'] == "CRITICAL":
                st.error("🚨 **RED ALERT PROTOCOL:** Mandatory floodplain evacuation within 2km radius. 2 NDRF teams dispatched with 12 motorboats.")
            else:
                st.warning("⚠️ **WARNING PROTOCOL:** Standby SDRF units alerted. Hourly telemetry gauge monitoring activated.")

# -----------------------------------------------------------------------------
# MULTI-MODEL ML BENCHMARKS
# -----------------------------------------------------------------------------
elif page == "🧪 Multi-Model ML Benchmarks":
    st.subheader("🧪 Multi-Model Machine Learning Benchmarks")

    bm_data = []
    for name, bm in ml_engine.benchmarks.items():
        bm_data.append({
            "Model": name,
            "Accuracy (%)": bm["accuracy"],
            "Precision (%)": bm["precision"],
            "Recall (%)": bm["recall"],
            "F1-Score (%)": bm["f1_score"],
            "ROC-AUC": bm["roc_auc"]
        })
    st.table(pd.DataFrame(bm_data))

# -----------------------------------------------------------------------------
# DEVELOPER API SPECS
# -----------------------------------------------------------------------------
elif page == "⚡ Developer API Specs":
    st.subheader("⚡ OpenAPI REST Developer Specifications")
    st.code("""
GET  /api/location-search?q=<place>  # Search location & get location flood risk
POST /api/predict                    # Single AI prediction & SHAP attributions
POST /api/predict/batch              # Batch prediction array processing
POST /api/simulate                   # Climate scenario stress test & sensitivity curve
GET  /api/geospatial                # GeoJSON state points & river station telemetry
GET  /api/alerts/active              # Live active emergency alerts stream
GET  /api/health                    # System health check
    """, language="bash")
