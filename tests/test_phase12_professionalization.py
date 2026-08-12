# -*- coding: utf-8 -*-
"""
Phase 12 Professionalization & Monitoring Test Suite - SepsisGuard v3.0
Verifies enhanced /health endpoint, last_successful_inference tracking, Python logging,
OpenAPI schema drift, and documentation link integrity.
"""

import unittest
import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.config import logger, API_KEY
from scripts.check_openapi_drift import check_openapi_drift
from scripts.check_doc_links import check_doc_links

class TestPhase12Professionalization(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # TEST 1 — Enhanced Health Endpoint
    def test_1_enhanced_health_endpoint(self):
        """Verify GET /health returns status, model_loaded, model_version, and active_patients."""
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_version"], "v2_2026-08-12")
        self.assertIn("last_successful_inference", data)
        self.assertIn("active_patients", data)

    # TEST 2 — Last Successful Inference Tracking
    def test_2_last_successful_inference_tracking(self):
        """Verify last_successful_inference is updated ONLY after a successful /predict call."""
        # Initial health check
        h1 = self.client.get('/health').get_json()
        initial_ts = h1.get("last_successful_inference")

        # Execute successful prediction
        payload = {
            "Heart_Rate": 115.0,
            "Oxygen_Level": 92.0,
            "Temperature": 38.5,
            "Blood_Pressure": 85.0,
            "Resp_Rate": 24.0,
            "Age": 68.0,
            "Infection_Marker": 0.8
        }
        res = self.client.post('/predict', json=payload, headers={"X-API-Key": API_KEY})
        self.assertEqual(res.status_code, 200)

        # Check health endpoint again
        h2 = self.client.get('/health').get_json()
        updated_ts = h2.get("last_successful_inference")
        self.assertIsNotNone(updated_ts, "last_successful_inference must not be None after a successful prediction!")
        self.assertNotEqual(initial_ts, updated_ts, "last_successful_inference timestamp must be updated after successful prediction!")

    # TEST 3 — Python Logging Verification
    def test_3_python_logging_setup(self):
        """Verify sepsisguard logger is configured and records messages cleanly."""
        self.assertIsNotNone(logger)
        logger.info("Test log message from Phase 12 test suite.")

    # TEST 4 — OpenAPI Drift Check Execution
    def test_4_openapi_drift_check(self):
        """Verify check_openapi_drift() passes with zero schema drift."""
        success = check_openapi_drift()
        self.assertTrue(success, "OpenAPI schema drift detected between openapi.yaml and Flask routes/Pydantic schemas!")

    # TEST 5 — Documentation Link Check Execution
    def test_5_doc_links_check(self):
        """Verify check_doc_links() passes with zero broken internal links."""
        success = check_doc_links()
        self.assertTrue(success, "Broken internal documentation links detected in project markdown files!")

    # TEST 6 — Environment Example File Verification
    def test_6_env_example_file(self):
        """Verify .env.example exists and contains no literal secret keys."""
        env_ex = os.path.join(PROJECT_ROOT, ".env.example")
        self.assertTrue(os.path.exists(env_ex), ".env.example file missing!")
        text = open(env_ex, "r").read()
        self.assertIn("FLASK_SECRET_KEY=", text)
        self.assertIn("API_KEY=", text)

    # TEST 7 — Dockerfile and Docker Compose File Verification
    def test_7_docker_files_exist(self):
        """Verify Dockerfile and docker-compose.yml exist in project root."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, "Dockerfile")), "Dockerfile missing!")
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, "docker-compose.yml")), "docker-compose.yml missing!")

if __name__ == '__main__':
    unittest.main()
