# -*- coding: utf-8 -*-
"""
Phase 9 Integration & End-to-End Test Suite - SepsisGuard v3.0
Covers end-to-end REST prediction pipeline, Socket.IO auth and room scoping,
mocked Gemini Copilot queries, temporal leakage prevention, NEWS2 unavailability,
and deterministic metrics calculations.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["FLASK_SECRET_KEY"] = "e2e_test_secret_key_8f9a2b4c6d8e1f3a5c"
os.environ["API_KEY"] = "sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b"
os.environ["FRONTEND_ORIGIN"] = "http://localhost:5000"

from backend.app import app, socketio, API_KEY
from data.features import add_derived_features
from model.clinical_baselines import compute_sirs, compute_qsofa_partial

class TestEndToEndIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.api_key = API_KEY
        cls.valid_vitals = {
            "Heart_Rate": 118.0,
            "Oxygen_Level": 92.0,
            "Temperature": 38.8,
            "Blood_Pressure": 85.0,
            "Resp_Rate": 24.0,
            "Age": 68.0,
            "Infection_Marker": 0.85
        }

    # 1. E2E REST API PIPELINE TEST
    def test_e2e_rest_predict_pipeline(self):
        """Verify full REST predict flow: Auth -> Validation -> XGBoost -> SHAP -> Response."""
        headers = {"X-API-Key": self.api_key}
        response = self.client.post('/predict', json=self.valid_vitals, headers=headers)
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn("risk_score", data)
        self.assertIn("risk_level", data)
        self.assertIn("explanation", data)
        self.assertIn("shap_explanation", data)
        self.assertIn("sirs_score", data)
        self.assertIn("disclaimer", data)
        
        # Verify SHAP structure
        shap_info = data["shap_explanation"]
        self.assertTrue(shap_info.get("available", False))
        self.assertEqual(shap_info.get("method"), "SHAP TreeExplainer")
        self.assertGreater(len(shap_info.get("features", [])), 0)

    # 2. MOCKED AI COPILOT SOCKET QUERY TEST
    @patch("requests.post")
    def test_e2e_copilot_socket_query_mocked(self, mock_post):
        """Verify Socket.IO Copilot query using mocked Gemini API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Patient presents tachycardia and hypotension requiring close monitoring."}]}}]
        }
        mock_post.return_value = mock_response

        sio_client = socketio.test_client(app, auth={"token": self.api_key})
        self.assertTrue(sio_client.is_connected())

        sio_client.emit("copilot_query", {"question": "What is the sepsis risk status?", "pid": "P001"})
        received = sio_client.get_received()
        
        copilot_events = [evt for evt in received if evt["name"] == "copilot_response"]
        self.assertEqual(len(copilot_events), 1)
        resp_data = copilot_events[0]["args"][0]
        self.assertIn("answer", resp_data)
        self.assertIn("tachycardia", resp_data["answer"].lower())
        sio_client.disconnect()

    # 3. TEMPORAL LEAKAGE PREVENTION REGRESSION TEST
    def test_temporal_leakage_prevention(self):
        """Verify that modifying future observations (t' > t) does NOT alter features at time t."""
        raw_df = pd.DataFrame([
            {"Patient_ID": "P100", "ICU_Length_of_Stay": 1.0, "Heart_Rate": 75.0, "Oxygen_Level": 98.0, "Temperature": 37.0, "Blood_Pressure": 120.0, "Mean_Arterial_Pressure": 85.0, "Resp_Rate": 16.0, "Infection_Marker": 0.5, "Age": 60.0},
            {"Patient_ID": "P100", "ICU_Length_of_Stay": 2.0, "Heart_Rate": 80.0, "Oxygen_Level": 98.0, "Temperature": 37.0, "Blood_Pressure": 118.0, "Mean_Arterial_Pressure": 82.0, "Resp_Rate": 18.0, "Infection_Marker": 0.5, "Age": 60.0},
            {"Patient_ID": "P100", "ICU_Length_of_Stay": 3.0, "Heart_Rate": 85.0, "Oxygen_Level": 98.0, "Temperature": 37.0, "Blood_Pressure": 115.0, "Mean_Arterial_Pressure": 80.0, "Resp_Rate": 20.0, "Infection_Marker": 0.5, "Age": 60.0},
        ])

        feat_original = add_derived_features(raw_df.copy())
        val_t2_orig = feat_original.iloc[1]["Heart_Rate_trend_6h"]

        # Modify FUTURE observation at ICU_Length_of_Stay=3
        raw_df_modified = raw_df.copy()
        raw_df_modified.loc[2, "Heart_Rate"] = 200.0  # Massive spike at t=3

        feat_modified = add_derived_features(raw_df_modified)
        val_t2_mod = feat_modified.iloc[1]["Heart_Rate_trend_6h"]

        self.assertEqual(val_t2_orig, val_t2_mod, "Temporal leakage detected! Future measurement (t=3) altered past feature at (t=2)!")

    # 4. NEWS2 UNSET REGRESSION TEST
    def test_news2_unavailable_regression(self):
        """Verify NEWS2 is not implemented using fake/invented values when oxygen support or consciousness is missing."""
        import model.clinical_baselines as cb
        self.assertFalse(hasattr(cb, "compute_news2_score"), "NEWS2 score function was introduced without required clinical variables!")

    # 5. DETERMINISTIC EVALUATION METRICS TEST
    def test_deterministic_evaluation_metrics_manual(self):
        """Test metric functions against a manually calculated confusion matrix: TP=8, TN=12, FP=3, FN=2."""
        from model.evaluate import calculate_medical_metrics
        y_true = np.array([1]*10 + [0]*15)
        y_pred_prob = np.array([0.9]*8 + [0.1]*2 + [0.9]*3 + [0.1]*12) # TP=8, FN=2, FP=3, TN=12 at threshold 0.5

        metrics = calculate_medical_metrics(y_true, y_pred_prob, threshold=0.5)

        tp, fn, fp, tn = 8, 2, 3, 12
        expected_sens = tp / (tp + fn) # 0.80
        expected_spec = tn / (tn + fp) # 0.80
        expected_ppv  = tp / (tp + fp) # 8/11 = 0.72727...
        expected_npv  = tn / (tn + fn) # 12/14 = 0.85714...

        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["TP"], 8)
        self.assertEqual(cm["TN"], 12)
        self.assertEqual(cm["FP"], 3)
        self.assertEqual(cm["FN"], 2)
        self.assertAlmostEqual(metrics["sensitivity_recall"], expected_sens, places=4)
        self.assertAlmostEqual(metrics["specificity"], expected_spec, places=4)
        self.assertAlmostEqual(metrics["ppv_precision"], expected_ppv, places=4)
        self.assertAlmostEqual(metrics["npv"], expected_npv, places=4)

if __name__ == '__main__':
    unittest.main()
