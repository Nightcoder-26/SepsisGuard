# External Validation Dataset Card & Independence Audit

## 1. External Dataset Status & Audit Result

> **EXTERNAL VALIDATION STATUS**: **`BLOCKED — independent dataset unavailable`**

### Dataset Independence Audit
- **Development Data Composition**: During Phase 2 preprocessing, PhysioNet 2019 Challenge dataset files from both **Set A** (Hospital System A) and **Set B** (Hospital System B) were combined and split at the patient level into `train.csv` ($80\%$), `val.csv` ($10\%$), and `test.csv` ($10\%$).
- **Overlap Audit Result**: Out of 978 patients in `data/raw/setB`, **764 patient records** are present in `data/processed/train.csv`. Set B records participated directly in preprocessing fitting, feature selection, and XGBoost model training.
- **Methodological Ruling**: Neither Set A nor Set B can be treated as an unexposed external validation dataset. Re-evaluating on Set B would constitute temporal and population data leakage. No external third-party hospital dataset (e.g. MIMIC-IV, eICU) is currently committed to this repository.
- **Strict Anti-Fabrication Directive**: In accordance with medical ML validation guidelines, no external results have been manufactured, fabricated, or generated from training data copies.

---

## 2. External Dataset Requirements (For Future Independent Cohorts)

When an independent clinical cohort (e.g., MIMIC-IV Sepsis Cohort or prospective hospital ICU dataset) becomes available, it must meet the following schema and preprocessing requirements:

### Required Input Variables & Feature Mapping

| Production Model Feature | Expected Raw Variable | Units | Conversion / Transformation |
| :--- | :--- | :--- | :--- |
| `Heart_Rate` | `HR` | bpm | Direct |
| `Oxygen_Level` | `O2Sat` / `SpO2` | % | Direct |
| `Temperature` | `Temp` | °C | Convert °F to °C if needed: $(T_{^\circ\text{F}} - 32) \times 5/9$ |
| `Blood_Pressure` | `SBP` | mmHg | Direct |
| `Mean_Arterial_Pressure` | `MAP` | mmHg | Direct (or calc: $\text{DBP} + \frac{1}{3}(\text{SBP} - \text{DBP})$) |
| `Resp_Rate` | `Resp` | bpm | Direct |
| `Age` | `Age` | years | Direct |
| `Infection_Marker` | `WBC` | $10^3/\mu\text{L}$ | Direct |
| `Glucose` | `Glucose` | mg/dL | Direct |
| `Creatinine` | `Creatinine` | mg/dL | Direct |
| `Platelets` | `Platelets` | $10^3/\mu\text{L}$ | Direct |
| `ICU_Length_of_Stay` | `ICULOS` | hours | Direct |
| `Sepsis_Risk` | `SepsisLabel` | binary | Sepsis-3 criteria ($0 = \text{Non-Sepsis}, 1 = \text{Sepsis}$) |

---

## 3. Strict Preprocessing & Model Freezing Rules

When applying external validation:
1. **Model Frozen**: Use model artifact `model/model_v2_2026-08-12.joblib` without retraining or hyperparameter tuning (`MODEL_FROZEN = True`).
2. **Preprocessing Frozen**: Transform external features using existing scaler artifact `model/scaler_v2_2026-08-12.joblib`. **DO NOT FIT A NEW SCALER OR IMPUTER ON EXTERNAL DATA**.
3. **Threshold Frozen**: Evaluate predictions at the fixed operating threshold $0.27$. **DO NOT OPTIMIZE A NEW THRESHOLD FOR EXTERNAL DATA**.
4. **Temporal Causality**: Compute 6-hour rolling trend features causally using only past observations ($t' \le t$).

---

## 4. Limitations & Non-Clinical Disclaimer
- **Research Prototype Scope**: External validation frameworks are designed for academic evaluation of ML model generalization across hospital populations.
- **No Regulatory Approval**: The framework and models have not received FDA or CE clinical certification for active patient triage.
