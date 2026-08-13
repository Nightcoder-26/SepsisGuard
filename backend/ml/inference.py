# -*- coding: utf-8 -*-
"""
ML Inference Module (Phase 10)
Pure business logic for ML model inference, risk classification, vital sign checking,
and SHAP explanation generation. Independent of Flask request handling.
"""

from backend.ml.model_loader import get_model, get_scaler, get_metadata, get_explainer
from model.explainability import explain_prediction

def safe_float(val, fallback):
    if val is None:
        return float(fallback)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(fallback)

def get_risk_level(prob):
    if prob < 27.0:
        return {"level": "Low",    "color": "#10b981", "msg": "Stable. Continue standard monitoring."}
    elif prob < 50.0:
        return {"level": "Medium", "color": "#f59e0b", "msg": "WARNING: Elevated sepsis risk above operating threshold (0.27)."}
    else:
        return {"level": "High",   "color": "#ef4444", "msg": "CRITICAL ALERT: High sepsis risk. Immediate clinical assessment required."}

def run_ml_pipeline(pid, vitals, generate_ai=False):
    """
    Executes the complete ML prediction pipeline:
    1. Extracts 20 features matching trained XGBoost model schema.
    2. Applies scaling if scaler artifact is present.
    3. Computes probability & assigns risk classification.
    4. Evaluates vital sign threshold triggers and SIRS criteria.
    5. Computes model-faithful SHAP feature attributions.
    """
    model = get_model()
    scaler = get_scaler()
    metadata = get_metadata()
    explainer = get_explainer()

    if model is None:
        return None

    try:
        feature_cols = metadata.get("features", [
            'Heart_Rate', 'Oxygen_Level', 'Temperature', 'Blood_Pressure',
            'Mean_Arterial_Pressure', 'Resp_Rate', 'Age', 'Infection_Marker',
            'Glucose', 'Creatinine', 'Platelets', 'Heart_Rate_isnan',
            'Temperature_isnan', 'Blood_Pressure_isnan', 'Resp_Rate_isnan',
            'Infection_Marker_isnan', 'Heart_Rate_trend_6h', 'Resp_Rate_trend_6h',
            'Mean_Arterial_Pressure_trend_6h', 'ICU_Length_of_Stay'
        ])

        bp_val = safe_float(vitals.get('Blood_Pressure'), 120.0)
        map_val = safe_float(vitals.get('Mean_Arterial_Pressure'), bp_val * 0.67 + 30)
        
        feature_dict = {
            'Heart_Rate': safe_float(vitals.get('Heart_Rate'), 80.0),
            'Oxygen_Level': safe_float(vitals.get('Oxygen_Level'), 98.0),
            'Temperature': safe_float(vitals.get('Temperature'), 37.0),
            'Blood_Pressure': bp_val,
            'Mean_Arterial_Pressure': map_val,
            'Resp_Rate': safe_float(vitals.get('Resp_Rate'), 16.0),
            'Age': safe_float(vitals.get('Age'), 65.0),
            'Infection_Marker': safe_float(vitals.get('Infection_Marker'), 0.5),
            'Glucose': safe_float(vitals.get('Glucose'), 120.0),
            'Creatinine': safe_float(vitals.get('Creatinine'), 1.0),
            'Platelets': safe_float(vitals.get('Platelets'), 200.0),
            'Heart_Rate_isnan': 0,
            'Temperature_isnan': 0,
            'Blood_Pressure_isnan': 0,
            'Resp_Rate_isnan': 0,
            'Infection_Marker_isnan': 0,
            'Heart_Rate_trend_6h': safe_float(vitals.get('Heart_Rate_trend_6h'), 0.0),
            'Resp_Rate_trend_6h': safe_float(vitals.get('Resp_Rate_trend_6h'), 0.0),
            'Mean_Arterial_Pressure_trend_6h': safe_float(vitals.get('Mean_Arterial_Pressure_trend_6h'), 0.0),
            'ICU_Length_of_Stay': safe_float(vitals.get('ICU_Length_of_Stay'), 1.0)
        }

        features_vector = [feature_dict[col] for col in feature_cols]

        # XGBoost is a tree-based model trained on raw unscaled features.
        X_in = [features_vector]

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_in)[0][1]) * 100
        else:
            prob = float(model.predict(X_in)[0]) * 100

        ri = get_risk_level(prob)

        hr_val   = feature_dict['Heart_Rate']
        temp_val = feature_dict['Temperature']
        o2_val   = feature_dict['Oxygen_Level']
        rr_val   = feature_dict['Resp_Rate']
        inf_val  = feature_dict['Infection_Marker']

        explanation = []
        if hr_val > 100:    explanation.append("Tachycardia")
        if hr_val < 60:     explanation.append("Bradycardia")
        if temp_val > 38.0: explanation.append("Fever / Hyperthermia")
        if temp_val < 36.0: explanation.append("Hypothermia")
        if bp_val < 90:     explanation.append("Hypotension")
        if o2_val < 94:     explanation.append("Hypoxia")
        if rr_val > 20:     explanation.append("Tachypnea")
        if inf_val > 0.5:   explanation.append("Elevated Infection Marker")

        sirs_criteria = {
            "temp_met": bool(temp_val < 36.0 or temp_val > 38.0),
            "hr_met": bool(hr_val > 90.0),
            "rr_met": bool(rr_val > 20.0),
            "wbc_met": bool(inf_val > 0.5),
        }
        sirs = sum(sirs_criteria.values())

        qsofa_criteria = {
            "rr_met": bool(rr_val >= 22.0),
            "sbp_met": bool(bp_val <= 100.0),
        }
        qsofa = sum(qsofa_criteria.values())

        shap_explanation = explain_prediction(model, explainer, feature_dict, feature_cols, top_k=5)
        alert_level = "CRITICAL" if prob >= 50.0 else ("WARNING" if prob >= 27.0 else "STABLE")

        ai_synthesis = ""
        if generate_ai:
            from backend.services.copilot import generate_gemini_synthesis
            ai_synthesis = generate_gemini_synthesis(vitals, prob, ri, explanation, shap_explanation)

        return {
            "risk_score":       round(prob, 1),
            "risk_level":       ri['level'],
            "risk_color":       ri['color'],
            "message":          ri['msg'],
            "explanation":      explanation or ["All vital signs within normal ranges."],
            "sirs_score":       sirs,
            "sirs_criteria":    sirs_criteria,
            "qsofa_score":      qsofa,
            "qsofa_criteria":   qsofa_criteria,
            "qsofa_note":       "Partial — mentation unavailable",
            "shap_explanation": shap_explanation,
            "contributions":    shap_explanation,
            "ai_synthesis":     ai_synthesis,
            "alert_level":      alert_level,
            "disclaimer":       "Demo system. Trained on synthetic data. Not validated for clinical use. Not a substitute for clinical judgment.",
        }
    except Exception as e:
        print(f"[ML Error] {e}")
        return None
