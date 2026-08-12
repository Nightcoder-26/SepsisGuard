# SepsisGuard v3.0 — Model Card

## 1. Model Overview
- **Model Name**: SepsisGuard XGBoost Risk Classifier
- **Model Version**: `v2_2026-08-12`
- **Model Type**: XGBoost Gradient Boosted Decision Trees (`XGBClassifier`)
- **Operating Decision Threshold**: $0.27$ (Optimized for high sensitivity in ICU triage)
- **Artifact File**: `model/model_v2_2026-08-12.joblib` (SHA-256: `59a5dfe7f6f3...`)
- **Release Date**: August 2026

---

## 2. Intended Use & Out-of-Scope Use

### Intended Use
- Technical demonstration of real-time ICU telemetry processing.
- Academic research prototyping for machine learning early-warning algorithms.
- Educational reference for leakage-free clinical preprocessing, SHAP explainability, and medical metric evaluation.

### Out-of-Scope Use
- **NOT** intended for clinical diagnosis of sepsis.
- **NOT** intended for issuing medication, dosing, or treatment orders.
- **NOT** intended for real-patient ICU monitoring or autonomous medical triage.
- **NOT** intended for deployment in real clinical care before multi-center prospective validation and regulatory approval.

---

## 3. Training Data & Preprocessing

- **Dataset**: PhysioNet 2019 Challenge Clinical ICU Dataset (PhysioNet CC-BY 4.0).
- **Dataset Reference**: Detailed dataset provenance in [data/DATASET_CARD.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/data/DATASET_CARD.md).
- **Split Strategy**: Patient-level group split ($80\%$ train, $10\%$ val, $10\%$ test). 0 patient overlap across splits.
- **Input Features (20 Features)**:
  1. Base Vital Signs & Labs (11): `Heart_Rate`, `Oxygen_Level`, `Temperature`, `Blood_Pressure`, `Mean_Arterial_Pressure`, `Resp_Rate`, `Age`, `Infection_Marker`, `Glucose`, `Creatinine`, `Platelets`.
  2. Missingness Indicators (5): `Heart_Rate_isnan`, `Temperature_isnan`, `Blood_Pressure_isnan`, `Resp_Rate_isnan`, `Infection_Marker_isnan`.
  3. Causal Rolling Trends (3): `Heart_Rate_trend_6h`, `Resp_Rate_trend_6h`, `Mean_Arterial_Pressure_trend_6h`.
  4. Temporal Context (1): `ICU_Length_of_Stay`.

---

## 4. Training Methodology & Hyperparameters
- **Cross-Validation**: 5-fold patient-level Stratified Group K-Fold cross-validation.
- **Hyperparameters**: `max_depth=4`, `learning_rate=0.05`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Threshold Selection**: Operating threshold $0.27$ selected on validation set to achieve $\ge 90\%$ sensitivity target.

---

## 5. Quantitative Performance Evaluation

Evaluated on untouched held-out test set (`data/processed/test.csv`, 9,962 rows across 244 patients):

| Metric | Measured Test Result | Interpretation |
| :--- | ---: | :--- |
| **Sensitivity / Recall** | **96.93%** | Detects $96.9\%$ of actual sepsis events at threshold $0.27$ |
| **Specificity** | **32.31%** | Identifies non-sepsis ICU hours |
| **Positive Predictive Value (PPV)** | **2.33%** | Low prevalence ($1.64\%$) context |
| **Negative Predictive Value (NPV)** | **99.84%** | $< 0.2\%$ false negative rate |
| **ROC-AUC** | **0.8250** | Strong discrimination across risk spectrum |
| **PR-AUC** | **0.1171** | Precision-Recall trade-off |
| **Brier Score** | **0.1524** | Probabilistic calibration error |

---

## 6. Clinical Baseline Comparison (Phase 5)
- **SIRS Baseline**: Clinical SIRS rule score ($\ge 2$ criteria) achieved ROC-AUC `0.5821`.
- **ML Advantage**: SepsisGuard XGBoost achieved ROC-AUC `0.8250` ($+0.2429$ improvement over SIRS, DeLong test $p < 0.001$).

---

## 7. Explainability & SHAP Feature Attributions
- **SHAP Engine**: Model-faithful `shap.TreeExplainer` generating exact feature risk contributions per prediction.
- **Top Risk Drivers**: Respiratory Rate, Temperature, Heart Rate, and 6-hour vital sign trends.
- **Non-Causal Notice**: SHAP attributions represent statistical model feature importances and do NOT establish clinical causality.

---

## 8. External Validation & Subgroup Performance

- **External Validation Status**: **`BLOCKED — independent dataset unavailable`** (PhysioNet Set A and Set B were merged in Phase 2 development splits; detailed in [validation/EXTERNAL_DATASET_CARD.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/validation/EXTERNAL_DATASET_CARD.md)).
- **Subgroup Analysis**: Subgroup metrics with Wilson 95% CIs available in [validation/GENERALIZATION_REPORT.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/validation/GENERALIZATION_REPORT.md).

---

## 9. Safety Considerations & Canonical Disclaimer

> **⚠️ CANONICAL NON-CLINICAL DISCLAIMER**
> 
> SepsisGuard v3.0 is a technical demonstration and research prototype.
> It has **NOT** been clinically validated for real-patient decision making, diagnosis, or treatment.
> It is **NOT** FDA approved, CE marked, or certified as Software as a Medical Device (SaMD).
> Independent clinical verification by qualified medical professionals is strictly required.
