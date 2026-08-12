# -*- coding: utf-8 -*-
"""
SepsisGuard Feature Sensitivity Analysis Engine (Phase 11)
Systematically evaluates frozen model prediction response to controlled vital sign perturbations.

IMPORTANT DISCLAIMER:
This analysis evaluates model response to controlled input perturbations and does not establish causal clinical relationships.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
VALIDATION_DIR = os.path.join(PROJECT_ROOT, "validation")
os.makedirs(VALIDATION_DIR, exist_ok=True)

MODEL_FROZEN = True

# Standard Baseline Patient Profile
BASELINE_PATIENT = {
    'Heart_Rate': 80.0,
    'Oxygen_Level': 98.0,
    'Temperature': 37.0,
    'Blood_Pressure': 120.0,
    'Mean_Arterial_Pressure': 85.0,
    'Resp_Rate': 16.0,
    'Age': 65.0,
    'Infection_Marker': 0.5,
    'Glucose': 120.0,
    'Creatinine': 1.0,
    'Platelets': 200.0,
    'Heart_Rate_isnan': 0,
    'Temperature_isnan': 0,
    'Blood_Pressure_isnan': 0,
    'Resp_Rate_isnan': 0,
    'Infection_Marker_isnan': 0,
    'Heart_Rate_trend_6h': 0.0,
    'Resp_Rate_trend_6h': 0.0,
    'Mean_Arterial_Pressure_trend_6h': 0.0,
    'ICU_Length_of_Stay': 12.0
}

# Clinical Perturbation Ranges for Key Vital Signs & Labs
PERTURBATION_SPECS = {
    'Heart_Rate': {'min': 40.0, 'max': 180.0, 'steps': 15, 'unit': 'bpm'},
    'Resp_Rate': {'min': 8.0, 'max': 45.0, 'steps': 15, 'unit': 'bpm'},
    'Temperature': {'min': 34.0, 'max': 41.0, 'steps': 15, 'unit': '°C'},
    'Blood_Pressure': {'min': 60.0, 'max': 180.0, 'steps': 15, 'unit': 'mmHg'},
    'Infection_Marker': {'min': 0.0, 'max': 25.0, 'steps': 15, 'unit': '10^3/uL'},
    'Glucose': {'min': 40.0, 'max': 400.0, 'steps': 15, 'unit': 'mg/dL'},
    'Creatinine': {'min': 0.4, 'max': 8.0, 'steps': 15, 'unit': 'mg/dL'},
}

def run_sensitivity_analysis():
    print("=" * 70)
    print("SEPSISGUARD PHASE 11 - FEATURE SENSITIVITY ANALYSIS ENGINE")
    print("=" * 70)
    print(f"[*] MODEL_FROZEN: {MODEL_FROZEN}")

    metadata_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("metadata_v2_") and f.endswith(".json")])
    if not metadata_files:
        raise FileNotFoundError("No metadata found in model/")

    with open(os.path.join(MODEL_DIR, metadata_files[-1]), 'r') as f:
        meta_data = json.load(f)

    version_id = meta_data["model_version"]
    threshold = float(meta_data.get("selected_threshold", 0.27))
    model = joblib.load(os.path.join(MODEL_DIR, f"model_{version_id}.joblib"))
    feature_cols = meta_data.get("features", [])

    # Compute baseline prediction
    base_df = pd.DataFrame([BASELINE_PATIENT])[feature_cols]
    base_prob = float(model.predict_proba(base_df)[0][1])
    print(f"[*] Baseline Patient Sepsis Risk Probability: {base_prob * 100:.2f}%")

    results = {
        "model_version": version_id,
        "baseline_probability": round(base_prob, 4),
        "disclaimer": "This analysis evaluates model response to controlled input perturbations and does not establish causal clinical relationships.",
        "perturbations": {}
    }

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, (feature, spec) in enumerate(PERTURBATION_SPECS.items()):
        vals = np.linspace(spec['min'], spec['max'], spec['steps'])
        probs = []
        deltas = []

        for v in vals:
            p_dict = BASELINE_PATIENT.copy()
            p_dict[feature] = v
            # If MAP, update MAP proportionally if BP changes
            if feature == 'Blood_Pressure':
                p_dict['Mean_Arterial_Pressure'] = v * 0.67 + 30
            p_df = pd.DataFrame([p_dict])[feature_cols]
            prob = float(model.predict_proba(p_df)[0][1])
            probs.append(round(prob, 4))
            deltas.append(round(prob - base_prob, 4))

        results["perturbations"][feature] = {
            "unit": spec["unit"],
            "values": [round(float(v), 2) for v in vals],
            "probabilities": probs,
            "probability_deltas": deltas
        }

        ax = axes_flat[idx]
        ax.plot(vals, [p * 100 for p in probs], marker='o', color='#2563eb', linewidth=2)
        ax.axhline(base_prob * 100, color='#6b7280', linestyle='--', label=f'Baseline ({base_prob*100:.1f}%)')
        ax.axhline(threshold * 100, color='#ef4444', linestyle=':', label=f'Threshold ({threshold*100:.0f}%)')
        ax.set_title(f"Sensitivity: {feature}", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"{feature} ({spec['unit']})", fontsize=9)
        ax.set_ylabel("Predicted Sepsis Risk (%)", fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for j in range(len(PERTURBATION_SPECS), len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle("SepsisGuard v3.0 — Systematic Controlled Feature Sensitivity Analysis\n(Evaluates model risk output response under controlled input perturbations)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_img = os.path.join(VALIDATION_DIR, "sensitivity_analysis.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"[OK] Saved sensitivity analysis plot: {out_img}")

    out_json = os.path.join(VALIDATION_DIR, "sensitivity_analysis.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved sensitivity analysis JSON: {out_json}")

    return results

if __name__ == '__main__':
    run_sensitivity_analysis()
