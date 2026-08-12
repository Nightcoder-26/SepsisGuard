# SepsisGuard Model Evaluation Report (Phase 4)

**Model Version**: `v2_2026-08-12`  
**Model Type**: `XGBoost`  
**Evaluation Dataset**: PhysioNet 2019 Sepsis Challenge (Held-out Test Set)  
**Test Set Integrity**: Verified (0 patient overlap across train/val/test)

---

## 1. Test Dataset Characteristics

- **Test Patients**: 244
- **Test Observations (Rows)**: 9962
- **Positive Sepsis Rows**: 163 (1.64%)
- **Negative Sepsis Rows**: 9799 (98.36%)
- **Patient-Level Prevalence**: 6.97% (17 positive / 244 total patients)

---

## 2. Primary Medical Performance (Threshold = 0.27)

> **Note**: Operating threshold was selected strictly on validation data in Phase 3 ($Sensitivity \ge 85\%$). The test set remained untouched until evaluation.

| Metric | Score | Medical Interpretation |
| :--- | :--- | :--- |
| **Sensitivity / Recall** | **96.93%** | Proportion of actual sepsis cases detected by the model |
| **Specificity** | **32.31%** | Proportion of non-sepsis cases correctly identified |
| **PPV / Precision** | **2.33%** | Positive Predictive Value (Prevalence-dependent: 1.64%) |
| **NPV** | **99.84%** | Negative Predictive Value |
| **Accuracy** | **33.37%** | Total correct prediction proportion (Secondary metric) |
| **F1-Score** | **0.0454** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **0.8250** | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | **0.1191** | Area under Precision-Recall curve (Primary ML metric) |
| **Brier Score** | **0.1524** | Probabilistic loss (Lower is better probabilistic accuracy) |

---

## 3. Confusion Matrix Raw Counts

- **True Positives (TP)**: 158
- **True Negatives (TN)**: 3166
- **False Positives (FP)**: 6633
- **False Negatives (FN)**: 5

---

## 4. Calibration & Utility Analysis

- **Calibration Status**: `POOR`
- **Brier Score**: `0.1524`
- **Explanation**: Significant probability distortion (Mean Absolute Calibration Error = 0.4036).
- **PhysioNet Utility Score**: Not calculated because preprocessed row-level evaluation data does not include official temporal sepsis onset time annotations.

---

## 5. Artifact Summary

- **Confusion Matrix Plot**: `confusion_matrix.png`
- **ROC / PR Curves**: `roc_pr_curves.png`
- **Calibration Diagram**: `calibration_curve.png`
- **Threshold Analysis**: `threshold_analysis.png`
- **JSON Report**: `metrics_report.json`
