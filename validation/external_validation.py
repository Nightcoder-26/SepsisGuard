# -*- coding: utf-8 -*-
"""
SepsisGuard External Validation Framework (Phase 11)
Evaluates the frozen production model (v2_2026-08-12, threshold 0.27) on independent
external clinical datasets without retraining or scaler re-fitting.
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
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.features import add_derived_features
from model.evaluate import calculate_medical_metrics

# 1. Freeze Model Assurance
MODEL_FROZEN = True

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
VALIDATION_DIR = os.path.join(PROJECT_ROOT, "validation")
os.makedirs(VALIDATION_DIR, exist_ok=True)

def run_external_validation(external_csv_path=None):
    """
    Executes external validation for a frozen production model artifact.
    If no external dataset path is supplied or found, outputs BLOCKED status report.
    """
    print("=" * 70)
    print("SEPSISGUARD PHASE 11 - EXTERNAL VALIDATION FRAMEWORK")
    print("=" * 70)
    print(f"[*] MODEL_FROZEN: {MODEL_FROZEN}")

    # Check for versioned Phase 3/4 model artifacts
    metadata_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("metadata_v2_") and f.endswith(".json")])
    if not metadata_files:
        raise FileNotFoundError("No versioned metadata artifact found in model/")

    latest_meta_path = os.path.join(MODEL_DIR, metadata_files[-1])
    with open(latest_meta_path, 'r') as f:
        meta_data = json.load(f)

    version_id = meta_data["model_version"]
    threshold = float(meta_data.get("selected_threshold", 0.27))
    model_path = os.path.join(MODEL_DIR, f"model_{version_id}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{version_id}.joblib")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    print(f"[*] Loaded Frozen Model Version: {version_id}")
    print(f"[*] Operating Threshold:        {threshold}")

    if external_csv_path is None or not os.path.exists(external_csv_path):
        print("\n[BLOCKED] External Validation: BLOCKED — independent dataset currently unavailable.")
        print("[*] Set A and Set B were combined in Phase 2 development splits; no external hospital cohort is available.")
        
        report_data = {
            "status": "BLOCKED",
            "reason": "Independent external hospital dataset unavailable.",
            "model_version": version_id,
            "threshold": threshold,
            "message": "Framework is ready. Pass an external dataset path to run evaluation."
        }
        
        with open(os.path.join(VALIDATION_DIR, "external_metrics.json"), "w") as f:
            json.dump(report_data, f, indent=2)

        md_content = (
            "# External Validation Report\n\n"
            "**EXTERNAL VALIDATION STATUS**: **`BLOCKED — independent dataset unavailable`**\n\n"
            "PhysioNet Set A and Set B were included in Phase 2 development splits (`train.csv`, `val.csv`, `test.csv`). "
            "To prevent data leakage, no external validation was performed on development data copies.\n\n"
            "The `external_validation.py` framework is implemented and ready to evaluate future independent cohorts.\n"
        )
        with open(os.path.join(VALIDATION_DIR, "external_validation_report.md"), "w") as f:
            f.write(md_content)

        print("[OK] Written status report to validation/external_validation_report.md")
        return report_data

    # Load and evaluate external dataset
    print(f"[*] Loading external dataset: {external_csv_path}")
    ext_df = pd.read_csv(external_csv_path)

    # Feature column verification
    feature_cols = meta_data.get("features", [])
    missing_cols = [c for c in feature_cols if c not in ext_df.columns]

    if missing_cols:
        print(f"[FAIL] MISSING REQUIRED FEATURE(S) IN EXTERNAL DATASET: {missing_cols}")
        raise ValueError(f"External dataset lacks required features: {missing_cols}")

    X_ext = ext_df[feature_cols]
    y_ext = ext_df['Sepsis_Risk'].values

    # Model inference (XGBoost receives feature dataframe directly)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_ext)[:, 1]
    else:
        y_prob = model.predict(X_ext)

    # Metric evaluation
    metrics = calculate_medical_metrics(y_ext, y_prob, threshold=threshold)
    metrics["model_version"] = version_id
    metrics["status"] = "COMPLETE"

    with open(os.path.join(VALIDATION_DIR, "external_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] External validation metrics calculated successfully.")
    print(f"  Sensitivity: {metrics['sensitivity_recall']*100:.2f}%")
    print(f"  Specificity: {metrics['specificity']*100:.2f}%")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")

    return metrics

if __name__ == '__main__':
    run_external_validation()
