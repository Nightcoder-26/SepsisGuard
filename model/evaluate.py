# -*- coding: utf-8 -*-
"""
SepsisGuard Standalone Evaluation Framework (Phase 4)
Evaluates the final Phase 3 model on the untouched held-out test dataset.
Computes medical-ML metrics (Sensitivity, Specificity, PPV, NPV, ROC-AUC, PR-AUC, Brier score, Calibration).
Generates visualization plots and exports metrics_report.json / metrics_report.md.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.features import add_derived_features
from model.explainability import generate_global_shap_summary

# Paths
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ROOT_DIR = PROJECT_ROOT

PATIENT_ID_COL = 'Patient_ID'
TARGET_COL = 'Sepsis_Risk'

def calculate_medical_metrics(y_true, y_prob, threshold):
    """
    Computes complete medical-ML evaluation metrics for a given binary threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total = len(y_true)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / total
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    row_prevalence = (tp + fn) / total

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        },
        "operating_threshold": float(threshold),
        "row_prevalence": float(row_prevalence),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "ppv_precision": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier)
    }

def run_standalone_evaluation():
    print("=" * 70)
    print("SEPSISGUARD PHASE 4 — STANDALONE MODEL EVALUATION FRAMEWORK")
    print("=" * 70)

    # 1. Locate Phase 3 Artifacts
    metadata_files = sorted(glob.glob(os.path.join(MODEL_DIR, "metadata_v2_*.json")))
    if not metadata_files:
        print("PHASE 4 BLOCKED")
        print(f"No versioned metadata artifact found in {MODEL_DIR}. Run Phase 3 first.")
        sys.exit(1)

    latest_metadata_path = metadata_files[-1]
    with open(latest_metadata_path, 'r') as f:
        metadata = json.load(f)

    version_id = metadata["model_version"]
    model_artifact_path = os.path.join(MODEL_DIR, f"model_{version_id}.joblib")

    if not os.path.exists(model_artifact_path):
        print("PHASE 4 BLOCKED")
        print(f"Model artifact not found at {model_artifact_path}.")
        sys.exit(1)

    model = joblib.load(model_artifact_path)
    print(f"[*] Loaded model artifact: {model_artifact_path}")
    print(f"[*] Loaded metadata:       {latest_metadata_path}")
    print(f"[*] Selected Model Type:   {metadata['model_type']}")
    print(f"[*] Selected Threshold:    {metadata['selected_threshold']:.2f}")

    # 2. Load Datasets & Verify Test Set Integrity
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = os.path.join(PROCESSED_DIR, "val.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test.csv")

    if not os.path.exists(test_path):
        print("PHASE 4 BLOCKED")
        print(f"Test dataset missing at {test_path}.")
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test_raw  = pd.read_csv(test_path)

    train_pids = set(df_train[PATIENT_ID_COL].unique())
    val_pids   = set(df_val[PATIENT_ID_COL].unique())
    test_pids  = set(df_test_raw[PATIENT_ID_COL].unique())

    overlap_train_test = train_pids & test_pids
    overlap_val_test   = val_pids & test_pids

    if len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
        print("CRITICAL EVALUATION FAILURE: TEST SET PATIENT LEAKAGE DETECTED!")
        print(f"Train/Test overlap: {len(overlap_train_test)} | Val/Test overlap: {len(overlap_val_test)}")
        sys.exit(1)

    print(f"[OK] Test set patient isolation verified: 0 patient overlap across splits ({len(test_pids)} test patients).")

    # 3. Preprocess Test Data (Causal Derived Features)
    df_test = add_derived_features(df_test_raw)
    feature_cols = metadata["features"]

    X_test = df_test[feature_cols]
    y_test = df_test[TARGET_COL].values
    patient_ids = df_test[PATIENT_ID_COL]

    total_test_rows = len(df_test)
    total_test_patients = len(test_pids)

    # Calculate Patient-level prevalence (any positive row for patient)
    patient_sepsis = df_test.groupby(PATIENT_ID_COL)[TARGET_COL].max()
    pos_patients = int((patient_sepsis == 1).sum())
    patient_prevalence = pos_patients / float(total_test_patients)

    row_pos = int((y_test == 1).sum())
    row_neg = int((y_test == 0).sum())
    row_prevalence = row_pos / float(total_test_rows)

    print(f"[*] Test observations: {total_test_rows} rows across {total_test_patients} patients.")
    print(f"[*] Row-level prevalence:     {row_prevalence * 100:.2f}% ({row_pos} pos / {row_neg} neg)")
    print(f"[*] Patient-level prevalence: {patient_prevalence * 100:.2f}% ({pos_patients} pos / {total_test_patients - pos_patients} neg)")

    # 4. Generate Continuous Probability Predictions on Test Set
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    operating_threshold = metadata["selected_threshold"]
    metrics = calculate_medical_metrics(y_test, y_prob, operating_threshold)

    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION RESULTS (Threshold = {:.2f})".format(operating_threshold))
    print("=" * 70)
    print(f"  Confusion Matrix: TN={cm['TN']} | FP={cm['FP']} | FN={cm['FN']} | TP={cm['TP']}")
    print(f"  Sensitivity / Recall: {metrics['sensitivity_recall'] * 100:.2f}% (Actual Sepsis Cases Detected)")
    print(f"  Specificity:          {metrics['specificity'] * 100:.2f}% (Non-Sepsis Cases Identified)")
    print(f"  PPV / Precision:      {metrics['ppv_precision'] * 100:.2f}% (Prevalence: {row_prevalence * 100:.2f}%)")
    print(f"  NPV:                  {metrics['npv'] * 100:.2f}%")
    print(f"  Accuracy:             {metrics['accuracy'] * 100:.2f}%")
    print(f"  F1-Score:             {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:               {metrics['pr_auc']:.4f}")
    print(f"  Brier Score:          {metrics['brier_score']:.4f}")
    print("=" * 70)

    # 5. Calibration Curve Analysis
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
    
    # Assess calibration status
    # Mean absolute calibration error
    cal_error = np.mean(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0
    if cal_error < 0.05:
        calibration_status = "GOOD"
        cal_explain = f"Probabilities are well-calibrated (Mean Absolute Calibration Error = {cal_error:.4f})."
    elif cal_error < 0.15:
        calibration_status = "NEEDS IMPROVEMENT"
        cal_explain = f"Moderate over/under-confidence observed (Mean Absolute Calibration Error = {cal_error:.4f})."
    else:
        calibration_status = "POOR"
        cal_explain = f"Significant probability distortion (Mean Absolute Calibration Error = {cal_error:.4f})."

    print(f"[*] Calibration Status: {calibration_status} ({cal_explain})")

    # 6. PhysioNet Utility Score Decision
    physionet_utility = {
        "score": None,
        "calculated": False,
        "reason": "PhysioNet utility score not calculated because the current preprocessed evaluation dataset representation does not provide the required temporal sepsis onset time annotations."
    }

    # 7. Generate Visualization Plots
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_matrix = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Sepsis', 'Sepsis'],
                yticklabels=['Non-Sepsis', 'Sepsis'])
    plt.title(f"Confusion Matrix (Test Set @ Threshold {operating_threshold:.2f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(ROOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot 2: ROC and Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color='#1f77b4', lw=2, label=f"XGBoost (AUC = {metrics['roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Chance (AUC = 0.50)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(rec_arr, prec_arr, color='#ff7f0e', lw=2, label=f"XGBoost (PR-AUC = {metrics['pr_auc']:.4f})")
    ax2.axhline(y=row_prevalence, color='navy', lw=1.5, linestyle='--', label=f'Baseline Prevalence ({row_prevalence*100:.1f}%)')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision (PPV)')
    ax2.set_title('Precision-Recall Curve (PR)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_plot_path = os.path.join(ROOT_DIR, "roc_pr_curves.png")
    plt.savefig(curves_plot_path, dpi=300)
    plt.savefig(os.path.join(MODEL_DIR, "roc_pr_curves.png"), dpi=300)
    plt.close()

    # Plot 3: Calibration Curve
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "s-", color='#2ca02c', label=f"XGBoost (Brier = {metrics['brier_score']:.4f})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title("Probability Calibration Reliability Diagram")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cal_plot_path = os.path.join(ROOT_DIR, "calibration_curve.png")
    plt.savefig(cal_plot_path, dpi=300)
    plt.savefig(os.path.join(MODEL_DIR, "calibration_curve.png"), dpi=300)
    plt.close()

    # Plot 4: Threshold Analysis Curve
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    sens_list, spec_list, prec_list, npv_list, f1_list = [], [], [], [], []

    for t in thresholds_grid:
        m = calculate_medical_metrics(y_test, y_prob, t)
        sens_list.append(m["sensitivity_recall"])
        spec_list.append(m["specificity"])
        prec_list.append(m["ppv_precision"])
        npv_list.append(m["npv"])
        f1_list.append(m["f1_score"])

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds_grid, sens_list, label='Sensitivity (Recall)', color='#d62728', lw=2)
    plt.plot(thresholds_grid, spec_list, label='Specificity', color='#1f77b4', lw=2)
    plt.plot(thresholds_grid, prec_list, label='Precision (PPV)', color='#ff7f0e', lw=2)
    plt.plot(thresholds_grid, f1_list, label='F1-Score', color='#9467bd', lw=1.5, linestyle='--')
    plt.axvline(x=operating_threshold, color='black', linestyle=':', lw=2,
                label=f'Selected Phase 3 Threshold ({operating_threshold:.2f})')
    plt.xlabel('Operating Threshold')
    plt.ylabel('Metric Score')
    plt.title('Medical Metric Trade-off Analysis Across Thresholds')
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    thresh_plot_path = os.path.join(ROOT_DIR, "threshold_analysis.png")
    plt.savefig(thresh_plot_path, dpi=300)
    plt.savefig(os.path.join(MODEL_DIR, "threshold_analysis.png"), dpi=300)
    plt.close()

    # Plot 5: Global SHAP Summary Plot
    shap_plot_path = generate_global_shap_summary(model, None, X_test, output_dir=ROOT_DIR)

    print(f"[OK] Generated visualization plots:")
    print(f"  - {cm_plot_path}")
    print(f"  - {curves_plot_path}")
    print(f"  - {cal_plot_path}")
    print(f"  - {thresh_plot_path}")

    # 8. Export Structured Reports
    full_report = {
        "model_version": version_id,
        "model_type": metadata["model_type"],
        "dataset_name": "PhysioNet 2019 Sepsis Challenge (Held-out Test Split)",
        "test_counts": {
            "test_patients": total_test_patients,
            "test_observations": total_test_rows,
            "positive_rows": row_pos,
            "negative_rows": row_neg,
            "row_prevalence_pct": round(row_prevalence * 100, 2),
            "patient_prevalence_pct": round(patient_prevalence * 100, 2)
        },
        "selected_threshold_provenance": "Validation set optimization for target sensitivity >= 0.85 (Phase 3)",
        "operating_threshold": operating_threshold,
        "metrics": metrics,
        "calibration": {
            "status": calibration_status,
            "brier_score": metrics["brier_score"],
            "explanation": cal_explain
        },
        "physionet_utility_score": physionet_utility,
        "test_set_integrity": "PASS (0 patient overlap across train/val/test)"
    }

    report_json_path = os.path.join(ROOT_DIR, "metrics_report.json")
    with open(report_json_path, 'w') as f:
        json.dump(full_report, f, indent=2)

    # Markdown Report
    report_md_content = f"""# SepsisGuard Model Evaluation Report (Phase 4)

**Model Version**: `{version_id}`  
**Model Type**: `{metadata['model_type']}`  
**Evaluation Dataset**: PhysioNet 2019 Sepsis Challenge (Held-out Test Set)  
**Test Set Integrity**: Verified (0 patient overlap across train/val/test)

---

## 1. Test Dataset Characteristics

- **Test Patients**: {total_test_patients}
- **Test Observations (Rows)**: {total_test_rows}
- **Positive Sepsis Rows**: {row_pos} ({row_prevalence*100:.2f}%)
- **Negative Sepsis Rows**: {row_neg} ({(1-row_prevalence)*100:.2f}%)
- **Patient-Level Prevalence**: {patient_prevalence*100:.2f}% ({pos_patients} positive / {total_test_patients} total patients)

---

## 2. Primary Medical Performance (Threshold = {operating_threshold:.2f})

> **Note**: Operating threshold was selected strictly on validation data in Phase 3 ($Sensitivity \\ge 85\\%$). The test set remained untouched until evaluation.

| Metric | Score | Medical Interpretation |
| :--- | :--- | :--- |
| **Sensitivity / Recall** | **{metrics['sensitivity_recall']*100:.2f}%** | Proportion of actual sepsis cases detected by the model |
| **Specificity** | **{metrics['specificity']*100:.2f}%** | Proportion of non-sepsis cases correctly identified |
| **PPV / Precision** | **{metrics['ppv_precision']*100:.2f}%** | Positive Predictive Value (Prevalence-dependent: {row_prevalence*100:.2f}%) |
| **NPV** | **{metrics['npv']*100:.2f}%** | Negative Predictive Value |
| **Accuracy** | **{metrics['accuracy']*100:.2f}%** | Total correct prediction proportion (Secondary metric) |
| **F1-Score** | **{metrics['f1_score']:.4f}** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **{metrics['roc_auc']:.4f}** | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | **{metrics['pr_auc']:.4f}** | Area under Precision-Recall curve (Primary ML metric) |
| **Brier Score** | **{metrics['brier_score']:.4f}** | Probabilistic loss (Lower is better probabilistic accuracy) |

---

## 3. Confusion Matrix Raw Counts

- **True Positives (TP)**: {cm['TP']}
- **True Negatives (TN)**: {cm['TN']}
- **False Positives (FP)**: {cm['FP']}
- **False Negatives (FN)**: {cm['FN']}

---

## 4. Calibration & Utility Analysis

- **Calibration Status**: `{calibration_status}`
- **Brier Score**: `{metrics['brier_score']:.4f}`
- **Explanation**: {cal_explain}
- **PhysioNet Utility Score**: Not calculated because preprocessed row-level evaluation data does not include official temporal sepsis onset time annotations.

---

## 5. Artifact Summary

- **Confusion Matrix Plot**: `confusion_matrix.png`
- **ROC / PR Curves**: `roc_pr_curves.png`
- **Calibration Diagram**: `calibration_curve.png`
- **Threshold Analysis**: `threshold_analysis.png`
- **JSON Report**: `metrics_report.json`
"""

    report_md_path = os.path.join(ROOT_DIR, "metrics_report.md")
    with open(report_md_path, 'w') as f:
        f.write(report_md_content)

    print(f"[OK] Saved structured reports:")
    print(f"  - {report_json_path}")
    print(f"  - {report_md_path}")

    return full_report

if __name__ == "__main__":
    run_standalone_evaluation()
