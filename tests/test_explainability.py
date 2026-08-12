# -*- coding: utf-8 -*-
"""
Phase 6 Unit Tests - SepsisGuard Real Model-Faithful Explainable AI (SHAP)
Tests SHAP additivity, model faithfulness on a single-feature dummy model,
feature order alignment, signed direction mapping, and removal of old heuristics.
"""

import unittest
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.features import add_derived_features
from model.explainability import create_explainer, explain_prediction, FEATURE_DISPLAY_MAP

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

class TestPhase6Explainability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load Phase 3/4 final model, metadata, and test split."""
        metadata_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("metadata_v2_") and f.endswith(".json")]
        if metadata_files:
            cls.meta_path = os.path.join(MODEL_DIR, metadata_files[0])
            with open(cls.meta_path, 'r') as f:
                cls.metadata = json.load(f)
            version_id = cls.metadata["model_version"]
            cls.model_path = os.path.join(MODEL_DIR, f"model_{version_id}.joblib")
            cls.model = joblib.load(cls.model_path)
            cls.feature_cols = cls.metadata["features"]
        else:
            cls.model = None
            cls.metadata = None
            cls.feature_cols = []

        cls.explainer = create_explainer(cls.model) if cls.model is not None else None

        df_test_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
        cls.df_test = add_derived_features(df_test_raw)

    def test_1_shap_additivity(self):
        """
        TEST 1: Verify SHAP additivity: base_value + sum(shap_values) == model raw margin output.
        Assert abs(reconstructed - expected) < 1e-4.
        """
        if self.model is None or self.explainer is None:
            self.skipTest("Phase 3/4 model not loaded")

        X_sample = self.df_test[self.feature_cols].iloc[:10]
        explanation = self.explainer(X_sample)

        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]

        expected_margins = self.model.predict(X_sample, output_margin=True)
        reconstructed_margins = base_val + explanation.values.sum(axis=1)

        diffs = np.abs(expected_margins - reconstructed_margins)
        for i, diff in enumerate(diffs):
            self.assertLess(diff, 1e-4, f"Additivity check failed for sample {i}: diff={diff}")

    def test_2_model_faithfulness_dummy_model(self):
        """
        TEST 2: Verify model faithfulness on a simple deterministic dummy model where ONLY Heart_Rate drives predictions.
        Confirms SHAP attributions reflect actual model logic, catching un-grounded heuristics.
        """
        # Simple training data: 100 samples with 3 features
        np.random.seed(42)
        X_dummy = pd.DataFrame({
            'Heart_Rate': np.random.uniform(50, 140, 100),
            'Temperature': np.random.uniform(35, 40, 100),
            'Resp_Rate': np.random.uniform(10, 30, 100)
        })
        # Target strictly dependent on Heart_Rate > 100
        y_dummy = (X_dummy['Heart_Rate'] > 100).astype(int)

        dummy_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        dummy_tree.fit(X_dummy, y_dummy)

        dummy_explainer = shap.TreeExplainer(dummy_tree)
        
        # Test sample with elevated HR
        sample_high_hr = pd.DataFrame([{'Heart_Rate': 130.0, 'Temperature': 37.0, 'Resp_Rate': 16.0}])
        exp_res = explain_prediction(dummy_tree, dummy_explainer, sample_high_hr, list(X_dummy.columns), top_k=3)

        top_feat = exp_res["features"][0]
        self.assertEqual(top_feat["feature"], "Heart_Rate", "Heart_Rate did not receive top SHAP attribution!")
        self.assertGreater(top_feat["shap_value"], 0, "High HR did not produce positive SHAP attribution!")

        # Changing an unrelated feature (Temperature) should not make it top attribution
        sample_temp_change = pd.DataFrame([{'Heart_Rate': 130.0, 'Temperature': 39.5, 'Resp_Rate': 16.0}])
        exp_res2 = explain_prediction(dummy_tree, dummy_explainer, sample_temp_change, list(X_dummy.columns), top_k=3)
        self.assertEqual(exp_res2["features"][0]["feature"], "Heart_Rate")

    def test_3_feature_order_alignment(self):
        """
        TEST 3: Verify SHAP feature names and feature ordering strictly match model feature_cols.
        Prevents dangerous feature-order mismatches.
        """
        if self.model is None or self.explainer is None:
            self.skipTest("Phase 3/4 model not loaded")

        sample_row = self.df_test[self.feature_cols].iloc[[0]]
        explanation_obj = self.explainer(sample_row)

        self.assertEqual(list(sample_row.columns), self.feature_cols, "Feature columns mismatch model metadata!")
        self.assertEqual(explanation_obj.values.shape[1], len(self.feature_cols), "SHAP feature dimension mismatch!")

    def test_4_signed_contributions(self):
        """
        TEST 4: Verify positive SHAP values map to 'increases_risk' and negative values map to 'decreases_risk'.
        """
        if self.model is None or self.explainer is None:
            self.skipTest("Phase 3/4 model not loaded")

        sample = self.df_test[self.feature_cols].iloc[0].to_dict()
        res = explain_prediction(self.model, self.explainer, sample, self.feature_cols, top_k=5)

        self.assertTrue(res["available"])
        for item in res["features"]:
            if item["shap_value"] > 0:
                self.assertEqual(item["direction"], "increases_risk")
            else:
                self.assertEqual(item["direction"], "decreases_risk")

    def test_5_no_old_heuristic_fallback(self):
        """
        TEST 5: Verify compute_contributions is removed from backend/app.py and SHAP error yields available=False.
        """
        import backend.app as app_mod
        self.assertFalse(hasattr(app_mod, "compute_contributions"), "compute_contributions function still exists in app.py!")

        # Verify safe failure handling when explainer is None
        res = explain_prediction(self.model, None, {}, self.feature_cols)
        self.assertFalse(res["available"])
        self.assertIn("unavailable", res["message"].lower())

if __name__ == '__main__':
    unittest.main()
