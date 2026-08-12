# -*- coding: utf-8 -*-
"""
SepsisGuard Real Model-Faithful Explainable AI System (Phase 6)
Uses shap.TreeExplainer derived from the final deployed model.
Replaces fake contribution heuristics with genuine signed SHAP attributions.
"""

import os
import sys
import json
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_DISPLAY_MAP = {
    'Heart_Rate': 'Heart Rate',
    'Oxygen_Level': 'SpO₂',
    'Temperature': 'Temperature',
    'Blood_Pressure': 'Systolic Blood Pressure',
    'Mean_Arterial_Pressure': 'Mean Arterial Pressure',
    'Resp_Rate': 'Respiratory Rate',
    'Age': 'Age',
    'Infection_Marker': 'White Blood Cell Count',
    'Glucose': 'Glucose',
    'Creatinine': 'Creatinine',
    'Platelets': 'Platelets',
    'Heart_Rate_isnan': 'Heart Rate Missing',
    'Temperature_isnan': 'Temperature Missing',
    'Blood_Pressure_isnan': 'Blood Pressure Missing',
    'Resp_Rate_isnan': 'Resp Rate Missing',
    'Infection_Marker_isnan': 'WBC Missing',
    'Heart_Rate_trend_6h': 'Heart Rate (6h Trend)',
    'Resp_Rate_trend_6h': 'Resp Rate (6h Trend)',
    'Mean_Arterial_Pressure_trend_6h': 'MAP (6h Trend)',
    'ICU_Length_of_Stay': 'ICU Length of Stay'
}

UNITS_MAP = {
    'Heart_Rate': 'bpm',
    'Oxygen_Level': '%',
    'Temperature': '°C',
    'Blood_Pressure': 'mmHg',
    'Mean_Arterial_Pressure': 'mmHg',
    'Resp_Rate': 'bpm',
    'Age': 'yrs',
    'Infection_Marker': '10³/µL',
    'Glucose': 'mg/dL',
    'Creatinine': 'mg/dL',
    'Platelets': '10³/µL',
    'Heart_Rate_trend_6h': 'bpm',
    'Resp_Rate_trend_6h': 'bpm',
    'Mean_Arterial_Pressure_trend_6h': 'mmHg',
    'ICU_Length_of_Stay': 'hrs'
}

def create_explainer(model):
    """
    Creates and returns a shap.TreeExplainer ONCE for the given tree-based model.
    Must be reused across inference requests.
    """
    try:
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception as e:
        print(f"[SHAP Init Error] {e}")
        return None

def explain_prediction(model, explainer, feature_dict_or_vector, feature_cols, top_k=5):
    """
    Generates genuine model-faithful SHAP attributions for a single sample prediction.
    
    Returns structured explanation dictionary with:
    - method: "SHAP TreeExplainer"
    - base_value: SHAP expected value (baseline model margin)
    - features: list of top_k signed feature attributions sorted by |SHAP|
    """
    if explainer is None or model is None:
        return {
            "method": "SHAP TreeExplainer",
            "available": False,
            "message": "Model explanation is temporarily unavailable."
        }

    try:
        # Build feature DataFrame aligned strictly with feature_cols
        if isinstance(feature_dict_or_vector, dict):
            row_dict = {col: feature_dict_or_vector.get(col, 0.0) for col in feature_cols}
            df_sample = pd.DataFrame([row_dict], columns=feature_cols)
        elif isinstance(feature_dict_or_vector, (list, np.ndarray)):
            df_sample = pd.DataFrame([feature_dict_or_vector], columns=feature_cols)
        elif isinstance(feature_dict_or_vector, pd.DataFrame):
            df_sample = feature_dict_or_vector[feature_cols].iloc[[0]]
        else:
            raise ValueError("Unsupported feature input type for SHAP explanation")

        # Obtain SHAP values
        explanation_obj = explainer(df_sample)
        shap_vals = explanation_obj.values[0] # (n_features,) or (n_features, n_classes)
        if len(shap_vals.shape) == 2 and shap_vals.shape[1] > 1:
            shap_vals = shap_vals[:, 1]
        
        if hasattr(explainer, "expected_value"):
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1] if len(base_val) > 1 else base_val[0]
            base_val = float(base_val)
        else:
            base_val = 0.0

        # Sort features by absolute SHAP attribution
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[::-1][:top_k]

        formatted_features = []
        for idx in top_indices:
            raw_feat = feature_cols[idx]
            display_name = FEATURE_DISPLAY_MAP.get(raw_feat, raw_feat)
            unit = UNITS_MAP.get(raw_feat, "")
            input_val = float(df_sample.iloc[0, idx])
            shap_v = float(shap_vals[idx])
            direction = "increases_risk" if shap_v > 0 else "decreases_risk"

            val_str = f"{input_val:.1f} {unit}".strip() if unit else f"{input_val:.2f}"
            sign_str = f"+{shap_v:.3f}" if shap_v > 0 else f"{shap_v:.3f}"
            action_word = "elevated" if shap_v > 0 else "reduced"
            formatted_text = f"{display_name} ({val_str}) {action_word} model risk estimate ({sign_str})."

            formatted_features.append({
                "feature": raw_feat,
                "display_name": display_name,
                "value": input_val,
                "unit": unit,
                "shap_value": round(shap_v, 4),
                "direction": direction,
                "formatted_text": formatted_text
            })

        return {
            "method": "SHAP TreeExplainer",
            "available": True,
            "base_value": round(base_val, 4),
            "features": formatted_features
        }

    except Exception as e:
        print(f"[SHAP Explanation Error] {e}")
        return {
            "method": "SHAP TreeExplainer",
            "available": False,
            "message": "Model explanation is temporarily unavailable."
        }

def generate_global_shap_summary(model, explainer, X_test, output_dir=None):
    """
    Generates and saves global SHAP summary plot (beeswarm) on held-out test data.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT

    if explainer is None:
        explainer = create_explainer(model)

    print("[*] Calculating global test-set SHAP summary values...")
    explanation = explainer(X_test)

    # Rename columns in explanation for readable plot labels
    X_renamed = X_test.rename(columns=FEATURE_DISPLAY_MAP)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explanation.values, X_renamed, show=False)
    plt.title("SepsisGuard Global SHAP Feature Attributions (Test Set)", fontsize=12, pad=15)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "shap_summary.png")
    model_out_path = os.path.join(PROJECT_ROOT, "model", "shap_summary.png")

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(model_out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Global SHAP summary plot saved to: {out_path}")
    return out_path
