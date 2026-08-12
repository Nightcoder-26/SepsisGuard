# -*- coding: utf-8 -*-
"""
Phase 1 Unit Tests - SepsisGuard AI Medical Safety
"""
import unittest
import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from app import run_ml_pipeline, _local_synthesis, app

class TestPhase1MedicalSafety(unittest.TestCase):

    def setUp(self):
        self.sample_vitals = {
            'Heart_Rate': 110,
            'Temperature': 38.5,
            'Blood_Pressure': 85,
            'Resp_Rate': 24,
            'Oxygen_Level': 92,
            'Age': 60,
            'Infection_Marker': 0.75
        }

    def test_prediction_output_fields(self):
        """Verify prediction response contains valid fields and lacks fabricated medical outputs."""
        result = run_ml_pipeline("test_patient", self.sample_vitals, generate_ai=False)
        self.assertIsNotNone(result, "ML pipeline result should not be None")
        
        # Must contain legitimate fields
        required_fields = ["risk_score", "risk_level", "message", "explanation", "sirs_score", "disclaimer"]
        for field in required_fields:
            self.assertIn(field, result, f"Result missing required field '{field}'")

        # Must NOT contain fabricated fields
        forbidden_fields = ["survival_rate", "confidence", "onset_time"]
        for field in forbidden_fields:
            self.assertNotIn(field, result, f"Result contains forbidden fabricated field '{field}'")

        # Verify disclaimer message content
        self.assertIn("Demo system", result["disclaimer"])
        self.assertIn("Not validated for clinical use", result["disclaimer"])

    def test_local_synthesis_non_directive_language(self):
        """Verify _local_synthesis uses observational language and avoids treatment directives."""
        banned_phrases = [
            "initiate antibiotics",
            "start antibiotics",
            "administer antibiotics",
            "give antibiotics",
            "prescribe",
            "administer",
            "medication should be started",
            "critically urgent"
        ]

        # Test high risk synthesis
        high_risk_text = _local_synthesis({"level": "High"}, ["Tachycardia", "Hypotension"], prob=85.0).lower()
        for phrase in banned_phrases:
            self.assertNotIn(phrase, high_risk_text, f"High risk synthesis contains directive phrase '{phrase}'")

        # Test medium risk synthesis
        med_risk_text = _local_synthesis({"level": "Medium"}, ["Tachycardia"], prob=45.0).lower()
        for phrase in banned_phrases:
            self.assertNotIn(phrase, med_risk_text, f"Medium risk synthesis contains directive phrase '{phrase}'")

        # Test low risk synthesis
        low_risk_text = _local_synthesis({"level": "Low"}, [], prob=10.0).lower()
        for phrase in banned_phrases:
            self.assertNotIn(phrase, low_risk_text, f"Low risk synthesis contains directive phrase '{phrase}'")

    def test_flask_predict_endpoint(self):
        """Verify /predict HTTP endpoint structure and disclaimer."""
        import backend.app as app_mod
        client = app_mod.app.test_client()
        headers = {"X-API-Key": app_mod.API_KEY}
        response = client.post('/predict', json=self.sample_vitals, headers=headers)
        self.assertEqual(response.status_code, 200, f"Predict failed with status {response.status_code}: {response.get_data(as_text=True)}")
        data = response.get_json()
        
        self.assertIn('risk_score', data)
        self.assertIn('disclaimer', data)
        self.assertNotIn('survival_rate', data)
        self.assertNotIn('confidence', data)
        self.assertNotIn('onset_time', data)

if __name__ == '__main__':
    unittest.main()
