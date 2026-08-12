# SepsisGuard AI v3.0 — Reproducibility & Deployment Manifest

This document provides a comprehensive step-by-step reproducibility guide for SepsisGuard v3.0, enabling researchers and software engineering teams to reconstruct the environment, obtain datasets, run feature extraction, train models, evaluate held-out performance, and launch the application backend from scratch.

---

## 1. System Requirements & Tested Environment

- **Operating System**: Windows 10/11 / Linux (Ubuntu 22.04 LTS) / macOS 13+
- **Python Version**: `3.11.9` (Supported: `3.10` – `3.11`)
- **Package Manager**: `pip` (v24.0+)
- **Primary Dependencies**:
  - `xgboost==2.0.3`
  - `shap==0.45.0`
  - `scikit-learn==1.4.1.post1`
  - `pandas==2.2.1`
  - `numpy==1.26.4`
  - `flask==3.0.2`
  - `flask-socketio==5.3.6`
  - `pydantic==2.6.4`
  - `flask-limiter==3.7.0`

---

## 2. Environment Setup & Dependency Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/SepsisGuard.git
cd SepsisGuard/Sepsis-detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
```

Activate environment:
- **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
- **Linux / macOS**: `source .venv/bin/activate`

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(For exact bit-for-bit environment locks, install via `pip install -r requirements.lock.txt`)*

---

## 3. Environment Configuration

### Step 1: Create Environment File
```bash
cp backend/.env.example backend/.env
```

### Step 2: Configure Environment Variables
Open `backend/.env` and set secure environment secrets:

```env
GEMINI_API_KEY=your_optional_gemini_api_key
FLASK_SECRET_KEY=3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b8f9a2b4c6d8e1f3a5c7b9d2e4f6a8c0b
API_KEY=sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b
FRONTEND_ORIGIN=http://localhost:5000,http://127.0.0.1:5000
PREDICT_RATE_LIMIT=60 per minute
```

To generate cryptographically strong secret keys, run:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 4. Dataset Acquisition & Provenance

> [!IMPORTANT]
> **Raw Clinical Data Notice**:
> Raw patient records are NOT committed to this repository in accordance with PhysioNet data governance policies.

### Dataset Source
- **Name**: Early Prediction of Sepsis from Clinical Data (PhysioNet/Computing in Cardiology Challenge 2019)
- **DOI**: [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)

### Downloading Dataset
1. Download the PhysioNet 2019 challenge dataset (`training_setA.zip` / `training_setB.zip`) from PhysioNet.org.
2. Extract `.psv` files into `data/raw/` (e.g. `data/raw/p000001.psv`).

### Cryptographic Checksums (SHA-256)
Verified preprocessed dataset splits located in `data/processed/`:

| Split | File | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Train** | `data/processed/train.csv` | `22f169f41749e31cf6d8f37eeab3996b513893b1bebcccd5c6e889c332bc91b2` |
| **Validation** | `data/processed/val.csv` | `059e5606b89c8a7befccd17a62b9ebf2db73ef6ebd7bd1e90f881540833b8b1b` |
| **Test** | `data/processed/test.csv` | `dd4b90c506d08580d4b7814851f1ca7d73ea3df1218c04d55090c4a236c9a107` |

---

## 5. Execution Workflow

### Step 1: Preprocessing & Feature Engineering
Extracts derived clinical features, missingness indicators, 6-hour causal rolling trends, and creates patient-isolated GroupKFold splits (`train`, `val`, `test`):
```bash
python data/preprocess.py
```

### Step 2: Model Selection & Training Pipeline
Trains baseline models (Logistic Regression, Random Forest, XGBoost) using 5-fold `GroupKFold` cross-validation with fixed seed (`random_state=42`), selects XGBoost, and saves model artifacts:
```bash
python model/train_model.py
```

Generated artifacts:
- `model/model_v2_2026-08-12.joblib`
- `model/scaler_v2_2026-08-12.joblib`
- `model/metadata_v2_2026-08-12.json`
- `model/model_card_v2_2026-08-12.json`

### Step 3: Standalone Evaluation Framework
Evaluates the final selected model on the untouched held-out test set (`data/processed/test.csv`):
```bash
python model/evaluate.py
```

Generates metrics, calibration curves, threshold analysis, and global SHAP summary plot (`shap_summary.png`).

### Step 4: Run Application Server

#### Local Development Server
```bash
python backend/app.py
```

#### Production Server (Gunicorn + Eventlet)
```bash
gunicorn -k eventlet -w 1 backend.app:app --bind 0.0.0.0:5000
```

Access the dashboard UI at: `http://localhost:5000`

---

## 6. Verification & Automated Smoke Testing

To verify project health, dependency imports, model artifact loading, SHAP explainer initialization, and pipeline integrity, run:
```bash
python scripts/smoke_test.py
```

Expected output:
```
======================================================================
SEPSISGUARD v3.0 - REPRODUCIBILITY & SMOKE TEST SUITE
======================================================================
[OK] Core dependencies imported successfully.
[OK] Environment variables verified.
[OK] Loaded model artifact: model_v2_2026-08-12.joblib
[OK] Loaded model metadata: model_v2_2026-08-12
[OK] SHAP TreeExplainer initialized and additivity verified.
[OK] Inference pipeline test prediction successful.
[OK] All smoke test checks PASSED.
```

To run full project test suite (43 tests):
```bash
python -m unittest discover tests
```
