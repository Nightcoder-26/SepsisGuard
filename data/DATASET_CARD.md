# Dataset Card: PhysioNet / Computing in Cardiology Challenge 2019 Sepsis Dataset

## 1. Dataset Summary & Provenance
- **Dataset Name**: Early Prediction of Sepsis from Clinical Data (PhysioNet/Computing in Cardiology Challenge 2019)
- **Source**: PhysioNet (PhysioNet.org)
- **Official DOI / Location**: [https://doi.org/10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857) | [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)
- **License / Terms of Use**: Creative Commons Attribution 4.0 International Public License (CC-BY 4.0).
- **Intended Purpose**: Research and development of early warning models for sepsis identification using ICU time-series physiological data.

## 2. Dataset Structure & Schema
The dataset consists of longitudinal hourly ICU measurements recorded from adult ICU patients across two tertiary hospital systems (Set A and Set B).

### Identifiers & Temporal Variables
- **`Patient_ID`**: Unique patient record identifier (e.g. `p000001`, `p100001`).
- **`ICULOS`**: ICU Length of Stay (hours since ICU admission). Primary chronological index.
- **`HospAdmTime`**: Hospital admission time prior to ICU admission (hours).

### Target Variable
- **`SepsisLabel`** (`Sepsis_Risk`): Binary clinical target (0 = Non-Sepsis, 1 = Sepsis).
  - Defined clinically according to Sepsis-3 criteria (SOFA score increase $\ge 2$ with suspected infection window).
  - For sepsis patients, `SepsisLabel = 1` occurs 6 hours prior to clinical diagnosis time $t_{\text{sepsis}}$ up to end of record.

### Key Physiological Features & Data Dictionary
| Project Feature | PhysioNet Column | Unit | Description |
| :--- | :--- | :--- | :--- |
| `Heart_Rate` | `HR` | BPM | Heart rate |
| `Oxygen_Level` | `O2Sat` | % | Pulse oximetry SpO2 |
| `Temperature` | `Temp` | °C | Body temperature |
| `Blood_Pressure` | `SBP` | mmHg | Systolic blood pressure |
| `Mean_Arterial_Pressure` | `MAP` | mmHg | Mean arterial pressure |
| `Resp_Rate` | `Resp` | BPM | Respiratory rate |
| `Infection_Marker` | `WBC` | $10^3 / \mu\text{L}$ | White blood cell count (primary infection indicator) |
| `Age` | `Age` | Years | Patient age (range 18–100) |
| `Glucose` | `Glucose` | mg/dL | Blood glucose concentration |
| `Creatinine` | `Creatinine` | mg/dL | Serum creatinine |
| `Platelets` | `Platelets` | $10^3 / \mu\text{L}$ | Platelet count |

## 3. Data Processing & Splitting Methodology

### Patient-Level Group-Aware Splitting
To prevent data leakage, dataset splitting is performed strictly at the **patient level** using `Patient_ID` grouping:
- **Train Set**: 80% of unique patients.
- **Validation Set**: 10% of unique patients.
- **Test Set**: 10% of unique patients (held out).

> **Enforced Assertion**: $\text{set}(\text{train\_patient\_ids}) \cap \text{set}(\text{test\_patient\_ids}) = \emptyset$. No patient records overlap across splits.

### Missingness Analysis & Imputation Strategy
- **Missingness Audit**: Real ICU data contains significant missingness due to intermittent lab testing and non-continuous vital monitoring.
- **Missingness Indicators**: Added binary indicator features (e.g. `HR_isnan`) to preserve clinical sampling frequency signals.
- **Leakage-Free Imputation**: Imputers (median imputation) and scalers are fitted **STRICTLY ON THE TRAINING SPLIT ONLY**, then applied to validation and test sets.

### Temporal Leakage Prevention
- Observations are ordered chronologically by `ICULOS` per patient.
- Feature calculations at time $t$ evaluate only measurements available at or before time $t$ ($t' \le t$). No future measurements ($t' > t$) are accessed.

## 4. Limitations & Non-Clinical Disclaimer
- **Prototype Scope**: This dataset is processed for research prototype demonstration.
- **No Clinical Certification**: The preprocessed dataset and derived models have not received FDA, CE, or regulatory approval for active clinical diagnosis or treatment.
- **Decision Support**: All outputs are decision-support artifacts requiring independent verification by licensed medical professionals.
