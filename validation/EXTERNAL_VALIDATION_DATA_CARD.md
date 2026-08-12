# External Validation Data Card & Independence Assessment

## 1. Executive Summary & Independence Audit

> **EXTERNAL VALIDATION STATUS**: **`BLOCKED — independent dataset unavailable`**

- **Dataset Provenance**: PhysioNet 2019 Challenge dataset files from **Set A** (Hospital System A) and **Set B** (Hospital System B) were combined during Phase 2 preprocessing and partitioned into $80\%$ training (`train.csv`), $10\%$ validation (`val.csv`), and $10\%$ test (`test.csv`) at the patient level.
- **Independence Audit Result**: Out of 978 patients in `data/raw/setB`, **764 patient records** are present in `data/processed/train.csv`. Set B records participated directly in preprocessing fitting, feature selection, and XGBoost model training.
- **Methodological Ruling**: Neither Set A nor Set B can be treated as an unexposed external validation dataset. Re-evaluating on Set B would constitute temporal and population data leakage. No external third-party hospital dataset (e.g. MIMIC-IV, eICU) is currently committed to this repository.
- **Strict Anti-Fabrication Directive**: In accordance with medical ML validation guidelines, no external metrics have been manufactured or generated from development dataset copies.

---

## 2. External Dataset Schema Requirements

Future independent validation cohorts must meet the following variable mappings and unit requirements:

| Production Feature | Source Column | Source Unit | Conversion Rule |
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

## 3. Production Pipeline Application Rules
1. **Model Frozen**: Use model artifact `model/model_v2_2026-08-12.joblib` without retraining or hyperparameter tuning (`MODEL_FROZEN = True`).
2. **Preprocessing Frozen**: External features must use existing preprocessing artifacts without re-fitting scalers or imputers on external data.
3. **Threshold Frozen**: Evaluate predictions at the fixed operating threshold $0.27$. **DO NOT OPTIMIZE A NEW THRESHOLD FOR EXTERNAL DATA**.
