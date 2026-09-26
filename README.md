# FloodRiskAI — Flood Risk Prediction Platform

DataQuest 2026 · Disaster Management Analytics · **Level 2: Prediction**

A small Flask web application built on top of the `Flood_Risk_Prediction_DataQuest2026.ipynb`
notebook. It provides:

- **Dashboard** (`/`) — Level 1 analytics: flood events by year, flood-type balance, top states,
  and leading causes of flooding, computed live from the two datasets.
- **Predict Risk** (`/predict`) — Level 2 prediction: enter a flood event's hydrological
  characteristics and get a live **Flood Risk Score** (Low/Medium/High), powered by the trained
  Random Forest model, plus the top contributing factors and a suggested action.
- **About** (`/about`) — project write-up: problem statement, datasets, model, and limitations.

## Project structure

```
flood_risk_platform/
├── app.py                     # Flask application (routes + API)
├── requirements.txt
├── README.md
├── model/
│   ├── flood_severity_rf_model.pkl   # trained Random Forest (from the notebook)
│   └── flood_severity_scaler.pkl     # matching StandardScaler
├── data/
│   ├── floodevents_indofloods.csv    # gauge-level flood events (for dashboard + model features)
│   └── India_Flood_Inventory_v3.csv  # state/district flood inventory (for dashboard)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── about.html
└── static/
    └── css/style.css
```

## Setup & run

Requires Python 3.9+.

```bash
cd flood_risk_platform
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## How prediction works

1. The eight sliders on the Predict page map to the same features used in the notebook: number of
   peak flood-level occurrences, peak discharge, flood volume, event duration, time to peak,
   recession time, start month, and start day-of-week.
2. On submit, the browser calls `POST /api/predict` with those values as JSON.
3. The Flask backend scales the input with the saved `StandardScaler` and calls
   `model.predict_proba()` on the saved Random Forest.
4. The probability of the "Severe Flood" class becomes the **Flood Risk Score**, bucketed into
   Low (<33%), Medium (33–66%) and High (≥66%) risk, each mapped to a suggested action.

## Regenerating the model (optional)

If you want to retrain the model instead of using the bundled `.pkl` files, run the modelling
cells in `Flood_Risk_Prediction_DataQuest2026.ipynb` (Section 5) and copy the newly saved
`flood_severity_rf_model.pkl` and `flood_severity_scaler.pkl` into `model/`.

## Notes & limitations

- Predictions are based purely on historical flood-event statistics (no live rainfall, river-gauge,
  soil-moisture or elevation data was available for this build) — see the notebook and the viva
  question bank for full discussion.
- This is a hackathon prototype: for production use, add authentication, input validation hardening,
  HTTPS, and model monitoring before exposing it publicly.
