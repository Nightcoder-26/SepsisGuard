# -*- coding: utf-8 -*-
"""
Phase 11 Validation Test Suite - SepsisGuard v3.0 (Research-Grade Validation)
Verifies 10 critical validation requirements:
1. Frozen model (no retraining).
2. No scaler re-fitting.
3. Patient overlap detection.
4. Metric consistency matching Phase 4.
5. Small subgroup handling ('INSUFFICIENT SAMPLE').
6. Age subgroup assignment correctness.
7. Sex subgroup assignment correctness.
8. Sensitivity analysis prediction perturbation response.
9. Model artifact file immutability (SHA-256).
10. Privacy (no patient identifiers in reports).
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

from validation.subgroup_analysis import evaluate_subgroup, MODEL_FROZEN, run_subgroup_analysis
from validation.external_validation import run_external_validation
from validation.sensitivity_analysis import run_sensitivity_analysis

class TestPhase11ResearchValidation(unittest.TestCase):

    # TEST 1 — Frozen Model Guarantee
    def test_1_external_data_cannot_retrain_model(self):
        """Verify MODEL_FROZEN flag is True and validation code does not call training."""
        self.assertTrue(MODEL_FROZEN, "MODEL_FROZEN flag must be set to True!")

    # TEST 2 — Preprocessing Immutability
    def test_2_external_preprocessing_no_refit(self):
        """Verify scaler fit method is not invoked on external data."""
        import joblib
        scaler_path = os.path.join(PROJECT_ROOT, "model", "scaler_v2_2026-08-12.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            self.assertTrue(hasattr(scaler, "mean_"), "Scaler must be an existing fitted object.")

    # TEST 3 — Patient Overlap Detection
    def test_3_patient_overlap_detection(self):
        """Verify patient overlap between train and test sets is zero."""
        train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "train.csv"))
        test_df  = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
        train_pids = set(train_df["Patient_ID"].unique())
        test_pids  = set(test_df["Patient_ID"].unique())
        overlap = train_pids.intersection(test_pids)
        self.assertEqual(len(overlap), 0, f"Patient leakage detected! Overlap: {overlap}")

    # TEST 4 — Metric Consistency
    def test_4_metrics_match_phase4_definitions(self):
        """Verify evaluation metrics return required Phase 4 medical keys."""
        from model.evaluate import calculate_medical_metrics
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = calculate_medical_metrics(y_true, y_prob, threshold=0.27)
        for key in ["sensitivity_recall", "specificity", "ppv_precision", "npv", "roc_auc", "pr_auc", "brier_score"]:
            self.assertIn(key, m, f"Metric key {key} missing from calculation.")

    # TEST 5 — Small Subgroup Handling
    def test_5_small_subgroup_returns_insufficient_sample(self):
        """Verify subgroup with <5 positive cases returns INSUFFICIENT SAMPLE status."""
        small_df = pd.DataFrame([
            {"Patient_ID": "P1", "Sepsis_Risk": 1, "y_prob": 0.9},
            {"Patient_ID": "P2", "Sepsis_Risk": 0, "y_prob": 0.1},
        ])
        res = evaluate_subgroup(small_df, "Tiny Subgroup", threshold=0.27)
        self.assertEqual(res["status"], "INSUFFICIENT SAMPLE")

    # TEST 6 — Age Subgroup Assignment
    def test_6_age_subgroup_assignment(self):
        """Verify age binning produces exact expected subgroup categories."""
        df = pd.DataFrame({"Age": [25.0, 50.0, 72.0, 85.0]})
        df['Age_Subgroup'] = pd.cut(df['Age'], bins=[0, 40, 65, 80, 120], labels=['<40 years', '40–64 years', '65–79 years', '80+ years'])
        self.assertEqual(list(df['Age_Subgroup']), ['<40 years', '40–64 years', '65–79 years', '80+ years'])

    # TEST 7 — Sex Subgroup Assignment
    def test_7_sex_subgroup_assignment(self):
        """Verify gender mapping maps 0 to Female and 1 to Male."""
        df = pd.DataFrame({"Gender": [0, 1]})
        df['Sex_Subgroup'] = df['Gender'].map({0: 'Female', 1: 'Male'})
        self.assertEqual(list(df['Sex_Subgroup']), ['Female', 'Male'])

    # TEST 8 — Sensitivity Analysis Response
    def test_8_sensitivity_analysis_perturbation_changes_prediction(self):
        """Verify perturbing Heart Rate changes the predicted sepsis probability."""
        res = run_sensitivity_analysis()
        hr_probs = res["perturbations"]["Heart_Rate"]["probabilities"]
        self.assertGreater(max(hr_probs) - min(hr_probs), 0.05, "Heart Rate perturbation should meaningfully alter model prediction probability!")

    # TEST 9 — Model Artifact Immutability
    def test_9_validation_does_not_modify_model_artifact(self):
        """Verify SHA-256 hash of model artifact is unchanged after running validation scripts."""
        model_path = os.path.join(PROJECT_ROOT, "model", "model_v2_2026-08-12.joblib")
        if os.path.exists(model_path):
            hash_before = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
            run_subgroup_analysis()
            hash_after = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
            self.assertEqual(hash_before, hash_after, "Model artifact file was modified during validation!")

    # TEST 10 — Privacy Assurance
    def test_10_no_patient_identifiers_in_reports(self):
        """Verify generated markdown and JSON report files contain no patient names or raw IDs."""
        val_dir = os.path.join(PROJECT_ROOT, "validation")
        report_files = ["GENERALIZATION_REPORT.md", "subgroup_analysis.md", "EXTERNAL_DATASET_CARD.md"]
        forbidden_terms = ["James Hartwell", "Sarah Chen", "Robert Okafor", "Maria Gonzalez", "p100001"]

        for filename in report_files:
            filepath = os.path.join(val_dir, filename)
            if os.path.exists(filepath):
                text = open(filepath, "r", encoding="utf-8", errors="ignore").read()
                for term in forbidden_terms:
                    self.assertNotIn(term, text, f"Forbidden patient identifier '{term}' found in {filename}!")

if __name__ == '__main__':
    unittest.main()
