# -*- coding: utf-8 -*-
"""
Tests for Clinician-Facing Sepsis Risk Assessment Workflow (Phase 12 Extension)
Verifies API contracts, qSOFA, SHAP grounding, prompt safety rules, error handling,
and absence of diagnostic or treatment directives.
"""

import os
import sys
import unittest
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.ml.inference import run_ml_pipeline
from backend.services.copilot import generate_gemini_synthesis

API_KEY = "sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

class TestClinicianWorkflow(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_1_run_risk_assessment_api_returns_required_fields(self):
        """Verify POST /predict returns risk_score, risk_level, SHAP, SIRS, qSOFA, and disclaimer."""
        payload = {
            "Heart_Rate": 118.0,
            "Oxygen_Level": 88.0,
            "Temperature": 39.5,
            "Blood_Pressure": 82.0,
            "Resp_Rate": 28.0,
            "Infection_Marker": 0.9,
            "Age": 78.0,
            "generate_synthesis": True
        }
        res = self.client.post("/predict", headers=HEADERS, data=json.dumps(payload))
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Core assessment fields
        self.assertIn("risk_score", data)
        self.assertIn("risk_level", data)
        self.assertIn("risk_color", data)
        self.assertIn("alert_level", data)
        self.assertIn("disclaimer", data)

        # SHAP & Clinical Baselines
        self.assertIn("shap_explanation", data)
        self.assertIn("sirs_score", data)
        self.assertIn("qsofa_score", data)
        self.assertIn("qsofa_note", data)
        self.assertEqual(data["qsofa_note"], "Partial — mentation unavailable")

        # Values check
        self.assertGreaterEqual(data["sirs_score"], 0)
        self.assertLessEqual(data["sirs_score"], 4)
        self.assertGreaterEqual(data["qsofa_score"], 0)
        self.assertLessEqual(data["qsofa_score"], 2)

    def test_2_qsofa_partial_calculation(self):
        """Verify qSOFA partial score equals sum of RR >= 22 and BP <= 100."""
        # High RR (28) + Low SBP (82) => qSOFA = 2
        res1 = run_ml_pipeline("P003", {
            "Heart_Rate": 118, "Oxygen_Level": 88, "Temperature": 39.5,
            "Blood_Pressure": 82, "Resp_Rate": 28, "Infection_Marker": 0.9, "Age": 78
        })
        self.assertEqual(res1["qsofa_score"], 2)
        self.assertEqual(res1["qsofa_note"], "Partial — mentation unavailable")

        # Normal RR (14) + Normal SBP (120) => qSOFA = 0
        res2 = run_ml_pipeline("P001", {
            "Heart_Rate": 72, "Oxygen_Level": 99, "Temperature": 36.8,
            "Blood_Pressure": 120, "Resp_Rate": 14, "Infection_Marker": 0.2, "Age": 67
        })
        self.assertEqual(res2["qsofa_score"], 0)

    def test_3_ai_synthesis_no_treatment_directives(self):
        """Verify AI synthesis narrative contains NO treatment recommendations, antibiotics, or diagnostic claims."""
        vitals = {"Heart_Rate": 118, "Oxygen_Level": 88, "Temperature": 39.5, "Blood_Pressure": 82, "Resp_Rate": 28, "Infection_Marker": 0.9}
        ri = {"level": "High", "color": "#ef4444", "msg": "CRITICAL"}
        explanation = ["Tachycardia", "Hypotension", "Tachypnea"]
        shap_exp = {
            "features": [
                {"display_name": "Respiratory Rate", "direction": "increases_risk"},
                {"display_name": "Systolic Blood Pressure", "direction": "increases_risk"}
            ]
        }
        synthesis = generate_gemini_synthesis(vitals, 88.0, ri, explanation, shap_exp)

        # Negative checks for non-clinical compliance
        synthesis_lower = synthesis.lower()
        forbidden = ["administer", "prescribe", "antibiotic", "fluid bolus", "vasopressor", "diagnose", "definitely has"]
        for term in forbidden:
            self.assertNotIn(term, synthesis_lower, f"Forbidden non-clinical directive '{term}' found in AI synthesis!")

    def test_4_error_handling_401_unauthorized(self):
        """Verify 401 response when X-API-Key is missing or invalid."""
        payload = {"Heart_Rate": 80.0, "Oxygen_Level": 98.0, "Temperature": 37.0, "Blood_Pressure": 120.0, "Resp_Rate": 16.0}
        res = self.client.post("/predict", headers={"X-API-Key": "invalid_key"}, data=json.dumps(payload))
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertIn("error", data)

    def test_5_error_handling_422_validation(self):
        """Verify 422 response when input vitals are out of physiological range or missing."""
        payload = {"Heart_Rate": 999.0, "Oxygen_Level": 98.0, "Temperature": 37.0, "Blood_Pressure": 120.0, "Resp_Rate": 16.0}
        res = self.client.post("/predict", headers=HEADERS, data=json.dumps(payload))
        self.assertEqual(res.status_code, 422)
        data = res.get_json()
        self.assertIn("error", data)

    def test_6_no_fake_survival_or_onset_fields(self):
        """Verify response contains NO removed fake metrics such as survival rate or onset time."""
        payload = {"Heart_Rate": 80.0, "Oxygen_Level": 98.0, "Temperature": 37.0, "Blood_Pressure": 120.0, "Resp_Rate": 16.0}
        res = self.client.post("/predict", headers=HEADERS, data=json.dumps(payload))
        data = res.get_json()
        forbidden_fields = ["survival_rate", "onset_time", "fake_confidence", "cure_probability"]
        for f in forbidden_fields:
            self.assertNotIn(f, data, f"Forbidden fake field '{f}' returned in API response!")

if __name__ == "__main__":
    unittest.main()
