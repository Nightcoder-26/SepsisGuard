# SepsisGuard v3.0 — Research-Grade Generalization & Subgroup Evaluation Report

## 1. Objective
This report evaluates the generalization capability, demographic subgroup equity, and vital sign feature sensitivity of the frozen SepsisGuard v3.0 production model (`v2_2026-08-12`, threshold `0.27`, XGBoost classifier) across held-out longitudinal ICU telemetry records.

---

## 2. Production Model & Preprocessing Specifications
- **Model Version**: `v2_2026-08-12` (XGBClassifier)
- **Model File**: `model/model_v2_2026-08-12.joblib` (SHA-256: `59a5dfe7f6f3...`)
- **Scaler Artifact**: `model/scaler_v2_2026-08-12.joblib`
- **Operating Threshold**: $0.27$
- **Feature Set (20 features)**: `Heart_Rate`, `Oxygen_Level`, `Temperature`, `Blood_Pressure`, `Mean_Arterial_Pressure`, `Resp_Rate`, `Age`, `Infection_Marker`, `Glucose`, `Creatinine`, `Platelets`, `Heart_Rate_isnan`, `Temperature_isnan`, `Blood_Pressure_isnan`, `Resp_Rate_isnan`, `Infection_Marker_isnan`, `Heart_Rate_trend_6h`, `Resp_Rate_trend_6h`, `Mean_Arterial_Pressure_trend_6h`, `ICU_Length_of_Stay`.
- **Model Frozen Flag**: `MODEL_FROZEN = True` enforced across all evaluation scripts.

---

## 3. Development Dataset
- **Dataset**: PhysioNet 2019 Challenge Sepsis Dataset
- **Patient Split**: $80\%$ Training (`train.csv`, 1,944 patients), $10\%$ Validation (`val.csv`, 243 patients), $10\%$ Held-Out Test (`test.csv`, 244 patients).
- **Test Set Size**: 9,962 longitudinal hourly observations across 244 untouched patients (0 patient overlap across splits).

---

## 4. External Dataset & Independence Assessment

> **EXTERNAL VALIDATION STATUS**: **`BLOCKED — independent dataset unavailable`**

- **Independence Audit Result**: PhysioNet 2019 Set A and Set B records were merged in Phase 2 development splits (`train.csv`, `val.csv`, `test.csv`; 764 Set B patients are in `train.csv`).
- **Data Leakage Ruling**: Neither Set A nor Set B can be treated as an unexposed external validation dataset. Re-evaluating on Set B would constitute temporal and population data leakage. No external third-party hospital dataset (e.g. MIMIC-IV, eICU) is currently committed to this repository.
- **Framework Status**: Standalone validation framework (`validation/external_validation.py`) is fully implemented and ready to evaluate future independent external cohorts.

---

## 5. Development vs External Performance Comparison

| Metric | Development Test Set | External Validation Cohort |
| :--- | ---: | ---: |
| **Sensitivity / Recall** | **96.93%** (158 / 163) | *BLOCKED — Data Unavailable* |
| **Specificity** | **32.31%** (3,166 / 9,799) | *BLOCKED — Data Unavailable* |
| **Positive Predictive Value (PPV)** | **2.33%** | *BLOCKED — Data Unavailable* |
| **Negative Predictive Value (NPV)** | **99.84%** | *BLOCKED — Data Unavailable* |
| **F1 Score** | **0.0455** | *BLOCKED — Data Unavailable* |
| **ROC-AUC** | **0.8250** | *BLOCKED — Data Unavailable* |
| **PR-AUC** | **0.1171** | *BLOCKED — Data Unavailable* |
| **Brier Score** | **0.1524** | *BLOCKED — Data Unavailable* |

---

## 6. Calibration Under Distribution Shift
- **Development Test Set Brier Score**: `0.1524`
- **External Calibration Status**: *BLOCKED — independent dataset unavailable*.

---

## 7. Subgroup Analysis — Age

| Subgroup | N Patients | N Observations | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | ROC-AUC (95% CI) | Brier Score |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- | ---: |
| **`<40 years`** | 33 | 1,509 | 1.92% | **96.55%** [82.8%, 99.4%] | **37.23%** [34.8%, 39.7%] | **0.8105** [0.730, 0.887] | 0.1558 |
| **`40–64 years`** | 107 | 3,995 | 1.68% | **98.51%** [92.0%, 99.7%] | **44.57%** [43.0%, 46.1%] | **0.8892** [0.856, 0.920] | 0.1342 |
| **`65–79 years`** | 72 | 3,190 | 1.79% | **100.0%** [93.7%, 100%] | **22.60%** [21.2%, 24.1%] | **0.8686** [0.832, 0.902] | 0.1706 |
| **`80+ years`** | 32 | 1,268 | 0.79% | **70.00%** [39.7%, 89.2%] | **12.56%** [10.8%, 14.5%] | **0.4479** [0.264, 0.635] | 0.1751 |

---

## 8. Subgroup Analysis — Sex

| Subgroup | N Patients | N Observations | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | ROC-AUC (95% CI) | Brier Score |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- | ---: |
| **Female** | 103 | 3,925 | 0.71% | **96.43%** [82.3%, 99.4%] | **33.08%** [31.6%, 34.6%] | **0.8180** [0.742, 0.889] | 0.1508 |
| **Male** | 141 | 6,037 | 2.24% | **97.04%** [92.6%, 98.8%] | **31.80%** [30.6%, 33.0%] | **0.8178** [0.781, 0.852] | 0.1534 |

---

## 9. Subgroup Analysis — ICU Context (Length of Stay)

| Subgroup Window | N Patients | N Observations | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | ROC-AUC (95% CI) | Brier Score |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- | ---: |
| **Early Stay ($\le 24$h)** | 244 | 5,458 | 0.97% | **90.57%** [79.8%, 95.9%] | **34.14%** [32.9%, 35.4%] | **0.6969** [0.630, 0.763] | 0.1518 |
| **Later Stay ($>24$h)** | 184 | 4,504 | 2.44% | **100.0%** [96.6%, 100%] | **30.14%** [28.8%, 31.5%] | **0.8682** [0.838, 0.898] | 0.1531 |

---

## 10. Temporal Analysis
- **Status**: `TEMPORAL VALIDATION BLOCKED — suitable temporal date/timestamp information unavailable in unanchored ICU telemetry files.`

---

## 11. Systematic Feature Sensitivity Analysis
- **Engine**: `validation/sensitivity_analysis.py`
- **Output Artifacts**: `validation/sensitivity_analysis.json`, `validation/sensitivity_analysis.png`
- **Disclaimer**: *This analysis evaluates model response to controlled input perturbations and does not establish causal clinical relationships.*
- **Findings**: Model probability monotonically increases with worsening vital signs (e.g. HR increasing from 40 to 180 bpm increases predicted probability from $15.2\%$ to $68.4\%$; Temperature increasing from 37.0°C to 41.0°C increases risk probability to $74.2\%$).

---

## 12. Important Research Findings
1. **Demographic Equity**: Sensitivity remains high ($\ge 96\%$) across both Female and Male cohorts, and across `<40`, `40–64`, and `65–79` age groups.
2. **Elderly Cohort Sensitivity Drop**: In the `80+ years` cohort ($N=10$ positives), Sensitivity drops to 70.0% and Specificity to 12.56% with wide confidence intervals $[39.7\% - 89.2\%]$, reflecting high vital sign variability and missingness in elderly ICU patients.
3. **Temporal Signal Accumulation**: Early ICU stay ($\le 24$ hours) exhibits lower ROC-AUC (0.6969) compared to later ICU stay ($>24$ hours, ROC-AUC 0.8682), demonstrating that predictive signal accumulates over length of ICU stay.

---

## 13. Limitations
1. **Independent External Cohort Blocked**: PhysioNet Set A and Set B were combined in Phase 2 development splits; no external hospital dataset was available in the repository.
2. **Small Positive Sample Size in 80+ Age Cohort**: $N=10$ positive cases in the 80+ group widens confidence intervals.
3. **Retrospective Evaluation**: All metrics are derived from retrospective ICU telemetry streams.

---

## 14. What This Phase Does NOT Establish (Clinical Validation Boundary)

> **IMPORTANT NON-CLINICAL DISCLAIMER**
> 
> This research evaluation does **NOT** establish:
> - Prospective clinical effectiveness or safety in real hospital workflows.
> - Improved patient clinical outcomes or reduced ICU mortality.
> - Reduced time to antibiotic administration.
> - FDA, CE, or regulatory software-as-a-medical-device (SaMD) clearance.
> - Suitability for autonomous clinical decision-making.
> 
> Full clinical validation requires prospective, multi-center randomized controlled trials (RCTs).

---

## 15. Conclusion & Research-Grade Generalization Assessment
- **Research-Grade Generalization**: **INSUFFICIENT EVIDENCE FOR EXTERNAL HOSPITAL GENERALIZATION** (due to external dataset availability blocker).
- **Subgroup Analysis**: **COMPLETE** on 244-patient held-out test set with Wilson 95% CIs and 1,000-bootstrap ROC-AUC CIs.

---

## 16. Runnable Commands

```bash
# Run External Validation Framework (returns BLOCKED status if no CSV passed)
python validation/external_validation.py

# Run Subgroup Performance Analysis
python validation/subgroup_analysis.py

# Run Systematic Feature Sensitivity Analysis
python validation/sensitivity_analysis.py
```
