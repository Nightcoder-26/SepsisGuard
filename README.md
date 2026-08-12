# SepsisGuard AI — Central Hospital Telemetry Server v3.0

> **Real-Time ICU Intelligence Ecosystem & Decision-Support Prototype**

---

## ⚠️ Limitations & Intended Use

> **CANONICAL DISCLAIMER**
> 
> **SepsisGuard v3.0 is a technical demonstration and research prototype.**
> **This system has not been clinically validated for real-patient decision making.**
> 
> - **Intended Use**: Technical demonstration, machine learning research prototyping, educational evaluation of medical ML pipelines.
> - **Out-of-Scope Use**: Clinical diagnosis of sepsis, issuing medication/treatment orders, autonomous patient triage, or deployment on real ICU patients before prospective clinical trials and regulatory clearance.
> - **Clinical Judgment Required**: All outputs represent statistical model probabilities requiring independent clinical verification by qualified medical professionals.

---

## Overview

SepsisGuard AI is an end-to-end machine learning decision-support architecture for early sepsis detection in intensive care unit (ICU) environments:

- **ML Inference Engine**: Versioned XGBoost risk classifier (`v2_2026-08-12`, operating threshold $0.27$) predicting sepsis probability from 20 clinical features.
- **PhysioNet 2019 Dataset**: Trained and evaluated on leakage-free patient splits from the PhysioNet 2019 Challenge dataset ($40,336$ patients, $1.4\text{M}$ clinical hours).
- **Model-Faithful SHAP Explainability**: Genuine `shap.TreeExplainer` feature attributions identifying top risk drivers per prediction (e.g. Temperature, Respiratory Rate, 6h vital sign trends).
- **Clinical Rule Baselines**: Real-time SIRS and partial qSOFA score evaluation displayed alongside ML risk predictions.
- **AI Copilot & Narrative Synthesis**: Gemini LLM clinical observation synthesis with strict non-clinical safety boundaries (no treatment directives or dosing orders).
- **Simulated ICU Telemetry Stream**: Real-time WebSocket streaming (`icu_unit_a`) delivering 6 simulated patient beds to interactive ICU dashboard cards. *(Note: Telemetry streams represent simulated demo data).*

---

## Evaluation Results

Detailed evaluation reports are available in [TEST_REPORT.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/TEST_REPORT.md), [clinical_baseline_comparison.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/clinical_baseline_comparison.md), and [validation/GENERALIZATION_REPORT.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/validation/GENERALIZATION_REPORT.md).

### Held-Out Test Set Metrics (Phase 4 — Threshold = 0.27)

Evaluated on untouched held-out test set (`data/processed/test.csv`, 9,962 longitudinal rows across 244 patients):

| Evaluation Metric | Measured Result | Clinical Significance |
| :--- | ---: | :--- |
| **Sensitivity / Recall** | **96.93%** | Detects $96.9\%$ of actual sepsis events at threshold $0.27$ |
| **Specificity** | **32.31%** | Identifies non-sepsis ICU hours |
| **Positive Predictive Value (PPV)** | **2.33%** | Low prevalence ($1.64\%$) context |
| **Negative Predictive Value (NPV)** | **99.84%** | $< 0.2\%$ false negative rate |
| **F1 Score** | **0.0455** | Harmonic mean of PPV and Recall |
| **ROC-AUC** | **0.8250** | Discrimination across risk spectrum |
| **PR-AUC** | **0.1171** | Precision-Recall curve area |
| **Brier Score** | **0.1524** | Probabilistic calibration error |

### Clinical Baseline Comparison (Phase 5)
- **SIRS Baseline**: ROC-AUC `0.5821`
- **SepsisGuard XGBoost**: ROC-AUC `0.8250` ($+0.2429$ improvement over SIRS, DeLong test $p < 0.001$)

### External Validation Status (Phase 11)
- **Status**: **`BLOCKED — independent dataset unavailable`**
- **Audit Result**: PhysioNet Set A and Set B were merged during Phase 2 development splits; no external hospital cohort was available in the repository. Detailed in [validation/EXTERNAL_DATASET_CARD.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/validation/EXTERNAL_DATASET_CARD.md).

---

## Quick Start & Installation

### 1. Local Setup

```bash
# Clone repository
git clone https://github.com/SepsisGuard/SepsisGuard-AI.git
cd SepsisGuard-AI

# Create virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Run SepsisGuard Telemetry Server
python backend/app.py
```

### 2. Docker Setup

```bash
# Build & launch container
docker-compose up -d --build

# Verify health status
curl http://localhost:5000/health
```

---

## Documentation Directory

- [ARCHITECTURE.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/ARCHITECTURE.md) — Modular architecture, dependency flow, and Socket.IO room scoping.
- [DEPLOYMENT.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/DEPLOYMENT.md) — Deployment, WSGI configuration, Docker, and health monitoring.
- [model/MODEL_CARD.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/model/MODEL_CARD.md) — Research-grade model card for model `v2_2026-08-12`.
- [data/DATASET_CARD.md](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/data/DATASET_CARD.md) — PhysioNet 2019 dataset card & preprocessing specification.
- [openapi.yaml](file:///c:/Users/Lenovo/Desktop/SepsisGuard/Sepsis-detection/openapi.yaml) — OpenAPI 3.0 specification for REST API endpoints.

---

## License
Distributed under the MIT License. PhysioNet 2019 data used under CC-BY 4.0.
