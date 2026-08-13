# -*- coding: utf-8 -*-
"""
Tests for Final UI/UX Professionalization & Clinician Decision-Support Positioning
Verifies non-clinical forbidden terms are absent, required professional terms are present,
metadata fields match model card, and criteria dictionaries are returned.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ml.inference import run_ml_pipeline

class TestUIProfessionalization(unittest.TestCase):
    def test_1_forbidden_non_clinical_terms_absent_from_frontend(self):
        """Verify forbidden consumer terms are absent from frontend HTML and JS UI copy."""
        frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
        forbidden = [
            "AI Doctor",
            "Diagnose Sepsis",
            "Sepsis Confirmed",
            "Guaranteed Detection",
            "Treatment Recommendation"
        ]
        
        target_files = ["dashboard.html", "patient.html", "index.html", "dashboard.js", "patient.js", "script.js"]
        for fname in target_files:
            fpath = os.path.join(frontend_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            for term in forbidden:
                self.assertNotIn(term, content, f"Forbidden non-clinical term '{term}' found in frontend/{fname}!")

    def test_2_required_professional_terms_present(self):
        """Verify professional decision-support terms are present in patient.html UI copy."""
        patient_html = os.path.join(PROJECT_ROOT, "frontend", "patient.html")
        with open(patient_html, "r", encoding="utf-8") as f:
            content = f.read()

        required_terms = [
            "Run Risk Assessment",
            "Why did the model produce this result?",
            "SHAP",
            "CLINICAL REFERENCE SCORES",
            "SIMULATED TELEMETRY"
        ]
        for term in required_terms:
            self.assertIn(term, content, f"Required professional term '{term}' missing from patient.html!")

    def test_3_ml_inference_returns_sirs_and_qsofa_criteria_checklists(self):
        """Verify run_ml_pipeline returns sirs_criteria and qsofa_criteria dictionaries."""
        res = run_ml_pipeline("P003", {
            "Heart_Rate": 118, "Oxygen_Level": 88, "Temperature": 39.5,
            "Blood_Pressure": 82, "Resp_Rate": 28, "Infection_Marker": 0.9, "Age": 78
        })
        self.assertIn("sirs_criteria", res)
        self.assertIn("qsofa_criteria", res)

        sirs_c = res["sirs_criteria"]
        self.assertTrue(sirs_c["temp_met"])
        self.assertTrue(sirs_c["hr_met"])
        self.assertTrue(sirs_c["rr_met"])
        self.assertTrue(sirs_c["wbc_met"])

        qsofa_c = res["qsofa_criteria"]
        self.assertTrue(qsofa_c["rr_met"])
        self.assertTrue(qsofa_c["sbp_met"])

    def test_4_about_model_metadata_in_dashboard(self):
        """Verify model version v2_2026-08-12 and threshold 0.27 are displayed in dashboard.html."""
        dashboard_html = os.path.join(PROJECT_ROOT, "frontend", "dashboard.html")
        with open(dashboard_html, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("v2_2026-08-12", content)
        self.assertIn("0.27", content)
        self.assertIn("SHAP", content)

if __name__ == "__main__":
    unittest.main()
