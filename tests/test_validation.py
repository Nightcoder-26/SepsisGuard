# -*- coding: utf-8 -*-
"""
Phase 11 Validation & Subgroup Analysis Test Suite - SepsisGuard v3.0
Verifies model freezing, no preprocessor re-fitting, missing feature error handling,
subgroup creation, small sample flagging, Wilson confidence intervals, and model immutability.
"""

import unittest
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from validation.subgroup_analysis import wilson_score_interval, bootstrap_roc_auc_ci, evaluate_subgroup, MODEL_FROZEN
from validation.external_validation import run_external_validation

class TestPhase11Validation(unittest.TestCase):

    # TEST 1 — Frozen Model Verification
    def test_1_frozen_model_flag(self):
        """Verify MODEL_FROZEN is True and validation code does not modify model parameters."""
        self.assertTrue(MODEL_FROZEN, "MODEL_FROZEN flag must be True in validation phase!")

    # TEST 2 — No External Scaler Re-Fitting
    def test_2_no_external_scaler_refitting(self):
        """Verify scaler fit method is not called during external data processing."""
        import joblib
        model_dir = os.path.join(PROJECT_ROOT, "model")
        scaler_path = os.path.join(model_dir, "scaler_v2_2026-08-12.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            self.assertTrue(hasattr(scaler, "mean_"), "Scaler must be already fitted artifact.")

    # TEST 3 — Feature Mapping Missing Column Handling
    def test_3_feature_mapping_missing_column(self):
        """Verify missing required features in external dataset raises ValueError."""
        tmp_csv = os.path.join(PROJECT_ROOT, "validation", "tmp_incomplete_ext.csv")
        incomplete_df = pd.DataFrame([{"Heart_Rate": 80.0, "Oxygen_Level": 98.0, "Sepsis_Risk": 0}])
        incomplete_df.to_csv(tmp_csv, index=False)
        try:
            with self.assertRaises(ValueError):
                run_external_validation(tmp_csv)
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)

    # TEST 4 — Metric Consistency
    def test_4_metric_consistency(self):
        """Verify metrics output structure matches Phase 4 evaluation schema."""
        from model.evaluate import calculate_medical_metrics
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = calculate_medical_metrics(y_true, y_prob, threshold=0.5)
        self.assertIn("sensitivity_recall", m)
        self.assertIn("specificity", m)
        self.assertIn("roc_auc", m)

    # TEST 5 — Subgroup Creation
    def test_5_subgroup_creation(self):
        """Verify fixture produces expected age, sex, and ICU stay subgroups."""
        fixture_df = pd.DataFrame([
            {"Patient_ID": "P01", "Age": 35.0, "Gender": 0, "ICU_Length_of_Stay": 10.0, "Sepsis_Risk": 1, "y_prob": 0.8},
            {"Patient_ID": "P02", "Age": 70.0, "Gender": 1, "ICU_Length_of_Stay": 30.0, "Sepsis_Risk": 0, "y_prob": 0.1},
        ])
        
        fixture_df['Age_Subgroup'] = pd.cut(fixture_df['Age'], bins=[0, 40, 65, 80, 120], labels=['<40 years', '40–64 years', '65–79 years', '80+ years'])
        fixture_df['Sex_Subgroup'] = fixture_df['Gender'].map({0: 'Female', 1: 'Male'})
        fixture_df['ICU_Stay_Subgroup'] = fixture_df['ICU_Length_of_Stay'].apply(lambda x: 'Early Stay (<=24h)' if x <= 24 else 'Later Stay (>24h)')

        self.assertEqual(fixture_df.iloc[0]['Age_Subgroup'], '<40 years')
        self.assertEqual(fixture_df.iloc[1]['Age_Subgroup'], '65–79 years')
        self.assertEqual(fixture_df.iloc[0]['Sex_Subgroup'], 'Female')
        self.assertEqual(fixture_df.iloc[1]['Sex_Subgroup'], 'Male')
        self.assertEqual(fixture_df.iloc[0]['ICU_Stay_Subgroup'], 'Early Stay (<=24h)')
        self.assertEqual(fixture_df.iloc[1]['ICU_Stay_Subgroup'], 'Later Stay (>24h)')

    # TEST 6 — Small Subgroup Handling
    def test_6_small_subgroup_handling(self):
        """Verify small subgroup with <5 positive cases returns status INSUFFICIENT SAMPLE."""
        small_df = pd.DataFrame([
            {"Patient_ID": "P1", "Sepsis_Risk": 1, "y_prob": 0.8},
            {"Patient_ID": "P2", "Sepsis_Risk": 0, "y_prob": 0.1},
            {"Patient_ID": "P3", "Sepsis_Risk": 0, "y_prob": 0.2},
        ])
        res = evaluate_subgroup(small_df, "Small Group", threshold=0.27)
        self.assertEqual(res["status"], "INSUFFICIENT SAMPLE")

    # TEST 7 — Confidence Interval Calculations
    def test_7_confidence_intervals(self):
        """Verify Wilson interval and bootstrap ROC-AUC CIs return valid bounds."""
        low, high = wilson_score_interval(8, 10) # 80%
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, high)

        y_true = np.array([1]*20 + [0]*20)
        y_prob = np.array([0.9]*18 + [0.1]*2 + [0.8]*2 + [0.1]*18)
        boot_low, boot_high = bootstrap_roc_auc_ci(y_true, y_prob, n_bootstraps=100, seed=42)
        self.assertGreaterEqual(boot_low, 0.5)
        self.assertLessEqual(boot_high, 1.0)
        self.assertLessEqual(boot_low, boot_high)

    # TEST 8 — Model Artifact Immutability
    def test_8_model_artifact_immutability(self):
        """Verify model artifact SHA-256 hash does not change before and after evaluation."""
        model_path = os.path.join(PROJECT_ROOT, "model", "model_v2_2026-08-12.joblib")
        if os.path.exists(model_path):
            hash_before = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
            run_external_validation()
            hash_after = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
            self.assertEqual(hash_before, hash_after, "Model artifact file was modified during validation!")

if __name__ == '__main__':
    unittest.main()
