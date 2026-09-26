"""
FloodRiskAI - Machine Learning & Analytics Engine
Handles multi-model training, evaluation, SHAP-like feature attributions,
and scenario sensitivity analysis.
Models included: Random Forest, Support Vector Machine (SVM), Logistic Regression.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURE_COLS = [
    "Num Peak FL",
    "Peak Discharge Q (cumec)",
    "Flood Volume (cumec)",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Start_Month",
    "Start_DayOfWeek",
]

FEATURE_META = [
    {"key": "num_peak_fl", "col": "Num Peak FL", "label": "Number of Peak Flood-Level Occurrences",
     "help": "How many times the water level peaked during the event.",
     "min": 1, "max": 10, "step": 1, "default": 1, "unit": "peaks"},
    {"key": "peak_discharge", "col": "Peak Discharge Q (cumec)", "label": "Peak Discharge (cumec)",
     "help": "Maximum river discharge recorded, in cubic metres per second.",
     "min": 0, "max": 10000, "step": 50, "default": 1175, "unit": "cumec"},
    {"key": "flood_volume", "col": "Flood Volume (cumec)", "label": "Flood Volume (cumec)",
     "help": "Total volume of water associated with the event.",
     "min": 0, "max": 30000, "step": 100, "default": 2207, "unit": "cumec"},
    {"key": "event_duration", "col": "Event Duration (days)", "label": "Event Duration (days)",
     "help": "Total number of days the flood event lasted.",
     "min": 1, "max": 60, "step": 1, "default": 2, "unit": "days"},
    {"key": "time_to_peak", "col": "Time to Peak (days)", "label": "Time to Peak (days)",
     "help": "Days from the start of the event to its peak level.",
     "min": 1, "max": 30, "step": 1, "default": 1, "unit": "days"},
    {"key": "recession_time", "col": "Recession Time (day)", "label": "Recession Time (days)",
     "help": "Days taken for water levels to recede after the peak.",
     "min": 1, "max": 45, "step": 1, "default": 2, "unit": "days"},
    {"key": "start_month", "col": "Start_Month", "label": "Start Month",
     "help": "Calendar month the flood event began (1 = Jan ... 12 = Dec).",
     "min": 1, "max": 12, "step": 1, "default": 7, "unit": "month"},
    {"key": "start_dow", "col": "Start_DayOfWeek", "label": "Start Day of Week",
     "help": "Day of week the flood began (0 = Monday ... 6 = Sunday).",
     "min": 0, "max": 6, "step": 1, "default": 2, "unit": "dow"},
]


class MLEngine:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_means = {}
        self.feature_stds = {}
        self.benchmarks = {}
        self.is_loaded = False
        self._load_or_train()

    def _prepare_data(self):
        flood_path = os.path.join(DATA_DIR, "floodevents_indofloods.csv")
        df = pd.read_csv(flood_path)
        df["Start Date"] = pd.to_datetime(df["Start Date"], format="%d-%m-%Y", errors="coerce")
        df.dropna(subset=["Start Date"], inplace=True)
        
        df["Start_Month"] = df["Start Date"].dt.month
        df["Start_DayOfWeek"] = df["Start Date"].dt.dayofweek

        for col in ["Peak Discharge Q (cumec)", "Flood Volume (cumec)"]:
            df[col] = df[col].fillna(df[col].median())

        for col in ["Num Peak FL", "Event Duration (days)", "Time to Peak (days)", "Recession Time (day)"]:
            df[col] = df[col].fillna(df[col].median())

        y = (df["Flood Type"] == "Severe Flood").astype(int)
        X = df[FEATURE_COLS]

        return X, y

    def _load_or_train(self):
        try:
            X, y = self._prepare_data()
            
            # Compute feature means and stds for attributions
            for col in FEATURE_COLS:
                self.feature_means[col] = float(X[col].mean())
                self.feature_stds[col] = float(X[col].std()) if X[col].std() > 0 else 1.0

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            rf_path = os.path.join(MODEL_DIR, "flood_severity_rf_model.pkl")
            scaler_path = os.path.join(MODEL_DIR, "flood_severity_scaler.pkl")

            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)
                self.models["Random Forest"] = rf_model
            else:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                self.models["Random Forest"] = rf_model

            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train Support Vector Machine (SVM)
            svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            svm_model.fit(X_train_scaled, y_train)
            self.models["Support Vector Machine (SVM)"] = svm_model

            # Train Logistic Regression
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            self.models["Logistic Regression"] = lr_model

            # Compute benchmarks
            rf_importances = getattr(self.models["Random Forest"], "feature_importances_", None)

            for name, mdl in self.models.items():
                y_pred = mdl.predict(X_test_scaled)
                y_proba = mdl.predict_proba(X_test_scaled)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba)
                cm = confusion_matrix(y_test, y_pred).tolist()

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                step = max(1, len(fpr) // 30)
                roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} 
                              for f, t in zip(fpr[::step], tpr[::step])]

                importances = getattr(mdl, "feature_importances_", None)
                if importances is None and hasattr(mdl, "coef_"):
                    importances = np.abs(mdl.coef_[0])
                    importances = importances / np.sum(importances)
                elif importances is None:
                    # Fallback to Random Forest importances for SVM
                    importances = rf_importances

                feat_imp = []
                if importances is not None:
                    feat_imp = [{"feature": f, "importance": round(float(i), 4)} 
                                for f, i in sorted(zip(FEATURE_COLS, importances), key=lambda p: p[1], reverse=True)]

                self.benchmarks[name] = {
                    "accuracy": round(float(acc) * 100, 1),
                    "precision": round(float(prec) * 100, 1),
                    "recall": round(float(rec) * 100, 1),
                    "f1_score": round(float(f1) * 100, 1),
                    "roc_auc": round(float(auc), 3),
                    "confusion_matrix": cm,
                    "roc_curve": roc_points,
                    "feature_importances": feat_imp,
                }

            self.is_loaded = True
        except Exception as e:
            print(f"MLEngine initialization error: {e}")
            self.is_loaded = False

    def predict(self, feature_values, model_name="Random Forest"):
        if not self.is_loaded:
            raise RuntimeError("ML Engine is not loaded.")

        target_model = self.models.get(model_name, self.models.get("Random Forest"))

        row = [float(v) for v in feature_values]
        df_input = pd.DataFrame([row], columns=FEATURE_COLS)
        X_scaled = self.scaler.transform(df_input)

        proba = float(target_model.predict_proba(X_scaled)[0, 1])
        risk_score = round(proba * 100, 1)

        if proba < 0.33:
            risk_level, risk_color = "Low", "success"
        elif proba < 0.66:
            risk_level, risk_color = "Medium", "warning"
        else:
            risk_level, risk_color = "High", "danger"

        prediction = "Severe Flood" if proba >= 0.5 else "Moderate / Normal Flood"

        # Feature Attribution (SHAP-like decomposition)
        importances = getattr(target_model, "feature_importances_", None)
        if importances is None and hasattr(target_model, "coef_"):
            importances = np.abs(target_model.coef_[0])
            importances = importances / np.sum(importances)
        elif importances is None:
            importances = getattr(self.models["Random Forest"], "feature_importances_", None)

        attributions = []
        if importances is not None:
            raw_contribs = []
            for col, val, imp in zip(FEATURE_COLS, row, importances):
                mean = float(self.feature_means[col])
                std = float(self.feature_stds[col])
                z_score = float((val - mean) / std)
                contrib = float(z_score * float(imp) * 10.0)
                raw_contribs.append((col, val, contrib, float(imp)))

            total_abs = sum(abs(c[2]) for c in raw_contribs) or 1.0
            for col, val, contrib, imp in raw_contribs:
                pct = round(float((contrib / total_abs) * (proba * 100)), 1)
                direction = "increases" if contrib >= 0 else "reduces"
                attributions.append({
                    "feature": col,
                    "value": float(val),
                    "contribution": round(float(contrib), 2),
                    "impact_pct": abs(float(pct)),
                    "direction": direction,
                    "importance": round(float(imp), 3)
                })

            attributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "prediction": prediction,
            "model_used": model_name,
            "attributions": attributions,
            "top_factors": attributions[:3] if attributions else []
        }

    def simulate_scenario(self, base_values, modifiers, model_name="Random Forest"):
        target_model = self.models.get(model_name, self.models.get("Random Forest"))
        
        base_pred = self.predict(base_values, model_name=model_name)

        discharge_mult = float(modifiers.get("discharge_mult", 1.0))
        volume_mult = float(modifiers.get("volume_mult", 1.0))
        duration_add = float(modifiers.get("duration_add", 0.0))
        peak_occurrences_add = float(modifiers.get("peak_occurrences_add", 0.0))

        stressed_values = list(base_values)
        stressed_values[0] = min(10.0, max(1.0, stressed_values[0] + peak_occurrences_add))
        stressed_values[1] = max(0.0, stressed_values[1] * discharge_mult)
        stressed_values[2] = max(0.0, stressed_values[2] * volume_mult)
        stressed_values[3] = min(60.0, max(1.0, stressed_values[3] + duration_add))

        stressed_pred = self.predict(stressed_values, model_name=model_name)

        discharge_curve = []
        base_q = base_values[1]
        test_steps = np.linspace(max(100, base_q * 0.2), base_q * 2.5, 15)
        
        for q_val in test_steps:
            temp_row = list(base_values)
            temp_row[1] = q_val
            if base_q > 0:
                temp_row[2] = base_values[2] * (q_val / base_q)
            res = self.predict(temp_row, model_name=model_name)
            discharge_curve.append({
                "discharge": round(float(q_val), 1),
                "risk_score": res["risk_score"]
            })

        return {
            "baseline": base_pred,
            "stressed": stressed_pred,
            "discharge_curve": discharge_curve,
            "delta_risk": round(stressed_pred["risk_score"] - base_pred["risk_score"], 1)
        }


# Global instance
ml_engine = MLEngine()
