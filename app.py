"""
FloodRiskAI — Extraordinary Disaster Management & AI Risk Platform
DataQuest 2026 · Disaster Management Analytics (Level 1 + Level 2 + Level 3)

Flask Application Server handling:
- Command Center Analytics Dashboard
- Interactive GIS Mapping API & View
- Multi-Model AI Predictions & SHAP Attributions
- Climate Scenario Stress-Test Simulator
- Real-time Early Warning Alert Center & Response Playbook
- Multi-Model Benchmarking Suite
- Developer API Hub & Playground
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

from ml_engine import ml_engine, FEATURE_META, FEATURE_COLS
from report_generator import generate_risk_report_html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)

# State Coordinates (Centroids for GIS mapping)
STATE_COORDS = {
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
    "Assam": {"lat": 26.2006, "lon": 92.9376},
    "Kerala": {"lat": 10.8505, "lon": 76.2711},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
    "Himachal Pradesh": {"lat": 31.1048, "lon": 77.1734},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
    "Bihar": {"lat": 25.0961, "lon": 85.3131},
    "Odisha": {"lat": 20.9517, "lon": 85.0985},
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
    "Gujarat": {"lat": 22.2587, "lon": 71.1924},
    "Punjab": {"lat": 31.1471, "lon": 75.3412},
    "Uttarakhand": {"lat": 30.0668, "lon": 79.0193},
    "Telangana": {"lat": 18.1124, "lon": 79.0193},
    "Jharkhand": {"lat": 23.6102, "lon": 85.2799},
    "Jammu and Kashmir": {"lat": 33.7782, "lon": 76.5762},
    "Chhattisgarh": {"lat": 21.2787, "lon": 81.8661}
}

# Major River Telemetry Stations
RIVER_STATIONS = [
    {
        "id": "STN-BRAHMAPUTRA-01",
        "station": "Guwahati Gauge Station",
        "basin": "Brahmaputra Basin",
        "state": "Assam",
        "river": "Brahmaputra",
        "lat": 26.1445,
        "lon": 91.7362,
        "danger_level": 49.68,
        "current_level": 50.45,
        "discharge": 14200,
        "status": "CRITICAL",
        "trend": "rising",
        "updated": "Live (Just now)"
    },
    {
        "id": "STN-GANGA-02",
        "station": "Patna Main Gauge",
        "basin": "Ganga Basin",
        "state": "Bihar",
        "river": "Ganga",
        "lat": 25.5941,
        "lon": 85.1376,
        "danger_level": 50.00,
        "current_level": 49.20,
        "discharge": 8900,
        "status": "WARNING",
        "trend": "rising",
        "updated": "Live (Just now)"
    },
    {
        "id": "STN-MAHANADI-03",
        "station": "Mahanadi Delta Station",
        "basin": "Mahanadi Basin",
        "state": "Odisha",
        "river": "Mahanadi",
        "lat": 20.4625,
        "lon": 85.8828,
        "danger_level": 26.50,
        "current_level": 27.15,
        "discharge": 11300,
        "status": "CRITICAL",
        "trend": "steady",
        "updated": "Live (Just now)"
    },
    {
        "id": "STN-GODAVARI-04",
        "station": "Rajahmundry Barrage",
        "basin": "Godavari Basin",
        "state": "Andhra Pradesh",
        "river": "Godavari",
        "lat": 17.0005,
        "lon": 81.8040,
        "danger_level": 14.50,
        "current_level": 13.90,
        "discharge": 6400,
        "status": "WARNING",
        "trend": "receding",
        "updated": "Live (Just now)"
    },
    {
        "id": "STN-PERIYAR-05",
        "station": "Idukki Reservoir Gauge",
        "basin": "Periyar Basin",
        "state": "Kerala",
        "river": "Periyar",
        "lat": 9.8497,
        "lon": 76.9813,
        "danger_level": 38.00,
        "current_level": 34.80,
        "discharge": 3200,
        "status": "NORMAL",
        "trend": "steady",
        "updated": "Live (Just now)"
    },
    {
        "id": "STN-YAMUNA-06",
        "station": "Old Delhi Railway Bridge",
        "basin": "Yamuna Basin",
        "state": "Uttar Pradesh",
        "river": "Yamuna",
        "lat": 28.6139,
        "lon": 77.2090,
        "danger_level": 205.33,
        "current_level": 204.60,
        "discharge": 4100,
        "status": "NORMAL",
        "trend": "receding",
        "updated": "Live (Just now)"
    }
]


def load_dashboard_data():
    flood_path = os.path.join(DATA_DIR, "floodevents_indofloods.csv")
    inv_path = os.path.join(DATA_DIR, "India_Flood_Inventory_v3.csv")

    df_flood = pd.read_csv(flood_path)
    df_flood["Start Date"] = pd.to_datetime(
        df_flood["Start Date"], format="%d-%m-%Y", errors="coerce"
    )
    df_flood.dropna(subset=["Start Date"], inplace=True)
    for col in ["Peak Discharge Q (cumec)", "Flood Volume (cumec)"]:
        df_flood[col] = df_flood[col].fillna(df_flood[col].median())
    df_flood["Year"] = df_flood["Start Date"].dt.year

    df_inv = pd.read_csv(inv_path)
    df_inv["Main Cause"] = df_inv["Main Cause"].fillna("Unknown")

    stats = {}
    stats["total_flood_events"] = int(len(df_flood))
    stats["total_inventory_events"] = int(len(df_inv))
    stats["severe_pct"] = round(
        float((df_flood["Flood Type"] == "Severe Flood").mean() * 100), 1
    )

    year_counts = df_flood["Year"].value_counts().sort_index()
    stats["years"] = [int(y) for y in year_counts.index.tolist()]
    stats["year_counts"] = [int(v) for v in year_counts.values.tolist()]

    type_counts = df_flood["Flood Type"].value_counts()
    stats["flood_type_labels"] = type_counts.index.tolist()
    stats["flood_type_counts"] = [int(v) for v in type_counts.values.tolist()]

    state_counts = (
        df_inv["State"].astype(str).str.split(",").explode().str.strip()
        .value_counts().head(10)
    )
    stats["state_labels"] = state_counts.index.tolist()
    stats["state_counts"] = [int(v) for v in state_counts.values.tolist()]

    cause_counts = (
        df_inv["Main Cause"].astype(str).str.lower().str.strip()
        .value_counts().head(8)
    )
    stats["cause_labels"] = [c.title() for c in cause_counts.index.tolist()]
    stats["cause_counts"] = [int(v) for v in cause_counts.values.tolist()]

    return stats


try:
    DASHBOARD_STATS = load_dashboard_data()
    dashboard_error = None
except Exception as exc:
    DASHBOARD_STATS = {}
    dashboard_error = str(exc)


# --- Page Routes ---

@app.route("/")
def index():
    return render_template(
        "index.html",
        stats=DASHBOARD_STATS,
        stats_json=json.dumps(DASHBOARD_STATS),
        dashboard_error=dashboard_error,
        active_alerts=RIVER_STATIONS
    )


@app.route("/gis")
def gis_page():
    return render_template(
        "gis.html",
        stations=RIVER_STATIONS,
        stats=DASHBOARD_STATS
    )


@app.route("/predict")
def predict_page():
    return render_template(
        "predict.html",
        features=FEATURE_META,
        models=list(ml_engine.models.keys()) if ml_engine.is_loaded else ["Random Forest"],
        month_names=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        dow_names=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )


@app.route("/simulator")
def simulator_page():
    return render_template(
        "simulator.html",
        features=FEATURE_META,
        models=list(ml_engine.models.keys()) if ml_engine.is_loaded else ["Random Forest"]
    )


@app.route("/alerts")
def alerts_page():
    return render_template(
        "alerts.html",
        stations=RIVER_STATIONS
    )


@app.route("/models")
def models_page():
    benchmarks = ml_engine.benchmarks if ml_engine.is_loaded else {}
    return render_template(
        "models.html",
        benchmarks=benchmarks,
        benchmarks_json=json.dumps(benchmarks)
    )


@app.route("/api-docs")
def api_docs_page():
    return render_template("api_docs.html")


@app.route("/about")
def about():
    return render_template("about.html")


# --- REST API Endpoints ---

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ml_engine.is_loaded:
        return jsonify({"error": "ML Engine not loaded"}), 500

    payload = request.get_json(force=True, silent=True) or {}
    model_name = payload.get("model_name", "Random Forest")

    try:
        row = []
        for meta in FEATURE_META:
            val = float(payload.get(meta["key"], meta["default"]))
            row.append(val)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input — all feature fields must be numeric."}), 400

    result = ml_engine.predict(row, model_name=model_name)
    
    recommendation_map = {
        "Low": "Routine monitoring. Standard gauge recording at normal operational intervals.",
        "Medium": "Notify regional disaster management office. Increase telemetry monitoring frequency to hourly readings.",
        "High": "CRITICAL EMERGENCY ALERT: Dispatch notification to State/District Disaster Management Authorities (SDMA/DDMA). Mobilize NDRF rescue squads, initiate floodplain evacuation protocols, and ready emergency shelters."
    }
    result["recommendation"] = recommendation_map.get(result["risk_level"], "Monitor closely.")
    
    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    if not ml_engine.is_loaded:
        return jsonify({"error": "ML Engine not loaded"}), 500

    payload = request.get_json(force=True, silent=True) or {}
    records = payload.get("records", [])
    model_name = payload.get("model_name", "Random Forest")

    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "Please provide a non-empty array of records under 'records' key."}), 400

    results = []
    for idx, rec in enumerate(records):
        try:
            row = [float(rec.get(meta["key"], meta["default"])) for meta in FEATURE_META]
            pred = ml_engine.predict(row, model_name=model_name)
            pred["record_id"] = rec.get("id", idx + 1)
            results.append(pred)
        except Exception as e:
            results.append({"record_id": idx + 1, "error": str(e)})

    return jsonify({"total": len(results), "predictions": results})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    if not ml_engine.is_loaded:
        return jsonify({"error": "ML Engine not loaded"}), 500

    payload = request.get_json(force=True, silent=True) or {}
    model_name = payload.get("model_name", "Random Forest")

    try:
        base_row = []
        for meta in FEATURE_META:
            val = float(payload.get(meta["key"], meta["default"]))
            base_row.append(val)

        modifiers = {
            "discharge_mult": float(payload.get("discharge_mult", 1.0)),
            "volume_mult": float(payload.get("volume_mult", 1.0)),
            "duration_add": float(payload.get("duration_add", 0.0)),
            "peak_occurrences_add": float(payload.get("peak_occurrences_add", 0.0))
        }

        sim_result = ml_engine.simulate_scenario(base_row, modifiers, model_name=model_name)
        return jsonify(sim_result)
    except Exception as e:
        return jsonify({"error": f"Simulation error: {str(e)}"}), 400


@app.route("/api/geospatial")
def api_geospatial():
    try:
        inv_path = os.path.join(DATA_DIR, "India_Flood_Inventory_v3.csv")
        df_inv = pd.read_csv(inv_path)
        
        state_agg = (
            df_inv["State"].astype(str).str.split(",").explode().str.strip()
            .value_counts().to_dict()
        )

        geojson_features = []
        for state_name, count in state_agg.items():
            if state_name in STATE_COORDS:
                coords = STATE_COORDS[state_name]
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [coords["lon"], coords["lat"]]
                    },
                    "properties": {
                        "state": state_name,
                        "historical_events": count,
                        "risk_category": "High Risk" if count > 500 else ("Medium Risk" if count > 200 else "Moderate Risk")
                    }
                })

        return jsonify({
            "type": "FeatureCollection",
            "features": geojson_features,
            "stations": RIVER_STATIONS
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/active")
def api_alerts_active():
    critical_count = sum(1 for s in RIVER_STATIONS if s["status"] == "CRITICAL")
    warning_count = sum(1 for s in RIVER_STATIONS if s["status"] == "WARNING")
    
    return jsonify({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_stations": len(RIVER_STATIONS),
        "critical_alerts": critical_count,
        "warning_alerts": warning_count,
        "stations": RIVER_STATIONS
    })


@app.route("/api/models/benchmark")
def api_models_benchmark():
    if not ml_engine.is_loaded:
        return jsonify({"error": "ML Engine not loaded"}), 500
    return jsonify(ml_engine.benchmarks)


@app.route("/api/export-report", methods=["POST"])
def api_export_report():
    payload = request.get_json(force=True, silent=True) or {}
    pred_data = payload.get("prediction", {})
    input_features = payload.get("inputs", {})
    station_name = payload.get("station_name", "Indo-Floods Hydrological Gauge Station #402")

    html_doc = generate_risk_report_html(pred_data, input_features, station_name=station_name)
    return Response(html_doc, mimetype="text/html")


@app.route("/presentation-pdf")
def download_presentation_pdf():
    pdf_path = os.path.join(BASE_DIR, "FloodRiskAI_Platform_Presentation.pdf")
    from flask import send_file
    return send_file(pdf_path, as_attachment=True, download_name="FloodRiskAI_Platform_Presentation.pdf")


@app.route("/api/location-search")
def api_location_search():
    query = request.args.get("q", "").strip()
    from location_engine import search_locations
    results = search_locations(query)
    return jsonify({"query": query, "count": len(results), "locations": results})


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ml_engine_loaded": ml_engine.is_loaded,
        "dashboard_loaded": bool(DASHBOARD_STATS),
        "available_models": list(ml_engine.models.keys()) if ml_engine.is_loaded else []
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
