# SepsisGuard AI v3.0 — Modular System Architecture & Specification

This document details the modular backend architecture, component boundaries, dependency flow, and security layers of SepsisGuard v3.0.

---

## 1. System Overview & Dependency Direction

SepsisGuard follows a strict unidirectional dependency hierarchy where API routes consume machine learning, validation, and domain services, while pure ML modules remain 100% decoupled from Flask HTTP request contexts and web server logic.

### Dependency Flow Diagram

```text
Browser UI (Dashboard / Patient Cards)
  │
  ├── REST Requests ─────────► api/predict.py
  │                             │
  │                             ├──► validation/schemas.py (Pydantic VitalsInput)
  │                             ├──► ml/inference.py (Pure ML Prediction Engine)
  │                             │     ├──► ml/model_loader.py (Versioned Model Singleton)
  │                             │     ├──► model/explainability.py (SHAP TreeExplainer)
  │                             │     └──► model/clinical_baselines.py (SIRS Rules)
  │                             └──► services/copilot.py (Gemini LLM Narrative)
  │
  ├── REST Requests ─────────► api/patients.py (Health Check & Patient Registry Query)
  │
  └── Socket.IO Telemetry ──► api/sockets.py
                                │
                                └──► services/telemetry.py (ICU Simulation Stream)
```

> **Strict Architectural Boundary Rule**: Pure ML modules (`backend/ml/*`) MUST NOT import API routes (`backend/api/*`) or access Flask request objects (`request`, `jsonify`, `session`).

---

## 2. Directory Structure & Module Responsibilities

```text
backend/
├── app.py                     # Application Factory (create_app) & Entry Point
├── config.py                  # Environment Configuration & Secret Management
├── validation/
│   ├── __init__.py
│   └── schemas.py             # Pydantic Input Schemas & JSON Sanitization
├── ml/
│   ├── __init__.py
│   ├── model_loader.py        # Single Model/Scaler/Metadata Loading Singleton
│   ├── inference.py           # Pure Business Logic for Prediction & Risk Scoring
│   ├── explainability.py      # Model-Faithful SHAP Feature Attribution
│   └── clinical_baselines.py  # SIRS & Partial qSOFA Clinical Rule Evaluators
├── api/
│   ├── __init__.py
│   ├── predict.py             # Blueprint for POST /predict Route & API Auth
│   ├── patients.py            # Blueprint for GET /patients, /health, Static Files
│   └── sockets.py             # Socket.IO Event Handlers & Room Scoping ('icu_unit_a')
└── services/
    ├── __init__.py
    ├── telemetry.py           # In-Memory Patient Registry & Vital Simulation Engine
    └── copilot.py             # Gemini LLM Clinical Copilot & Fallback Narratives
```

---

## 3. Detailed Component Specifications

### 3.1 Application Factory (`backend/app.py`)
- **Responsibility**: Initializes Flask application instance, configures CORS allow-lists (`FRONTEND_ORIGIN`), attaches Flask-Limiter rate limiting, registers API Blueprints (`predict_bp`, `patients_bp`), and binds real-time Socket.IO event handlers.
- **Line Count**: Reduced from 667 lines down to 78 lines.

### 3.2 Configuration (`backend/config.py`)
- **Responsibility**: Centralized environment variable reading (`FLASK_SECRET_KEY`, `API_KEY`, `GEMINI_API_KEY`, `PREDICT_RATE_LIMIT`) and absolute filesystem paths (`PROJECT_ROOT`, `MODEL_DIR`, `FRONTEND_DIR`).

### 3.3 Model Loader Singleton (`backend/ml/model_loader.py`)
- **Responsibility**: Scans `model/` for versioned artifacts (`metadata_v2_*.json`, `model_v2_*.joblib`, `scaler_v2_*.joblib`), loads XGBoost model into memory ONCE at startup, and initializes `shap.TreeExplainer`.
- **Lifecycle**: Exposes thread-safe getters (`get_model()`, `get_scaler()`, `get_metadata()`, `get_explainer()`). Prevents repeated model reloading on per-request paths.

### 3.4 ML Inference Engine (`backend/ml/inference.py`)
- **Responsibility**: Pure ML pipeline execution (`run_ml_pipeline`). Converts vital sign inputs to 20 model features, computes probability, assigns risk classification (Low: $<30\%$, Medium: $30-70\%$, High: $>70\%$), evaluates SIRS score, and generates SHAP explanations. Can be imported and tested independently of Flask.

### 3.5 Validation Schemas (`backend/validation/schemas.py`)
- **Responsibility**: Pydantic `VitalsInput` schema enforcing clinical boundaries (e.g. Heart Rate: $20.0-300.0$, Oxygen Level: $50.0-100.0$, Temperature: $30.0-45.0$). Returns HTTP 422 for out-of-bound inputs.

### 3.6 API Layer (`backend/api/`)
- **`predict.py`**: Handles `POST /predict`, checks `X-API-Key` authentication via `require_api_key`, applies rate limiting (60/min), validates Pydantic schema, and returns JSON payload.
- **`patients.py`**: Serves `GET /patients` (API key protected), `GET /health` system status, and dashboard static files (`dashboard.html`, `patient.html`).
- **`sockets.py`**: Manages real-time Socket.IO authentication (`auth.token`), scope authorization (`join_room('icu_unit_a')`), telemetry broadcasts, and AI Copilot query events.

### 3.7 Services Layer (`backend/services/`)
- **`telemetry.py`**: Manages demo patient registry (`PATIENTS`, `patient_state`) and vital simulation loop (`simulate_vitals`).
  > **Demo Data Notice**: All patient profiles, vital streams, and telemetry events in `telemetry.py` represent simulated demonstration data.
- **`copilot.py`**: Handles Gemini LLM prompt construction with strict non-clinical safety restrictions (no treatment orders or dosing directives) and local fallback synthesis.

---

## 4. Verification & Testing

Pure ML modules (`inference.py`, `explainability.py`, `clinical_baselines.py`) can be imported and executed standalone without Flask web server initialization:

```bash
python -c "from backend.ml.inference import run_ml_pipeline; print(run_ml_pipeline('P001', {'Heart_Rate': 110, 'Oxygen_Level': 92, 'Temperature': 38.5, 'Blood_Pressure': 85, 'Resp_Rate': 24, 'Age': 68}))"
```

To run full project test suite (53 tests across all modules):
```bash
pytest
```
