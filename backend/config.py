# -*- coding: utf-8 -*-
"""
SepsisGuard Configuration Module (Phase 10 / Phase 12)
Centralized environment configuration, path definitions, security secrets, and logging setup.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sepsisguard")

# Path Definitions
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Security & Secrets Configuration
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "sepsisguard_secret_key_8f9a2b4c6d8e1f3a5c7b9d2e4f6a8c0b")
API_KEY = os.environ.get("API_KEY", "sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# CORS Allow-List Configuration
raw_origins = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5000,http://127.0.0.1:5000")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

# Rate Limiting Configuration
PREDICT_RATE_LIMIT = os.environ.get("PREDICT_RATE_LIMIT", "60 per minute")
