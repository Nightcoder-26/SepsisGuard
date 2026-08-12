# -*- coding: utf-8 -*-
"""
Phase 4 Unit Tests - SepsisGuard Model Evaluation Framework
Includes Golden-Value hand-verifiable metric test, test set integrity assertion,
standalone execution check, and metric reproducibility.
"""

import unittest
import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.evaluate import (
    calculate_medical_metrics, run_standalone_evaluation,
    PROCESSED_DIR, ROOT_DIR, PATIENT_ID_COL
)

class TestPhase4EvaluationFramework(unittest.TestCase):

    def test_1_golden_value_verification(self):
        """
        TEST 1: Golden-Value test using a deterministic hand-verifiable confusion matrix.
        Known parameters:
          TP = 8, TN = 12, FP = 3, FN = 2 (Total = 25)
        Hand Calculations:
          Sensitivity / Recall = 8 / (8 + 2) = 0.8000
          Specificity          = 12 / (12 + 3) = 0.8000
          PPV / Precision      = 8 / (8 + 3) = 0.7272727...
          NPV                  = 12 / (12 + 2) = 0.8571428...
          Accuracy             = (8 + 12) / 25 = 0.8000
          F1 Score             = 2 * (8/11 * 0.8) / (8/11 + 0.8) = 1.6 * (8/11) / (16.8/11) = 12.8 / 16.8 = 0.7619047...
        """
        # Create synthetic ground truth and predicted probabilities
        y_true = np.array([1]*8 + [0]*12 + [0]*3 + [1]*2)
        # Probabilities so that threshold 0.5 yields exact TP=8, TN=12, FP=3, FN=2
        y_prob = np.array([0.9]*8 + [0.1]*12 + [0.9]*3 + [0.1]*2)

        metrics = calculate_medical_metrics(y_true, y_prob, threshold=0.5)
        cm = metrics["confusion_matrix"]

        self.assertEqual(cm["TP"], 8)
        self.assertEqual(cm["TN"], 12)
        self.assertEqual(cm["FP"], 3)
        self.assertEqual(cm["FN"], 2)

        self.assertAlmostEqual(metrics["sensitivity_recall"], 0.8000, places=4)
        self.assertAlmostEqual(metrics["specificity"], 0.8000, places=4)
        self.assertAlmostEqual(metrics["ppv_precision"], 8/11, places=4)
        self.assertAlmostEqual(metrics["npv"], 12/14, places=4)
        self.assertAlmostEqual(metrics["accuracy"], 0.8000, places=4)
        self.assertAlmostEqual(metrics["f1_score"], 0.7619, places=4)

    def test_2_test_set_integrity(self):
        """TEST 2: Verify zero patient leakage between train, val, and held-out test splits."""
        df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
        df_val   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
        df_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

        train_pids = set(df_train[PATIENT_ID_COL].unique())
        val_pids   = set(df_val[PATIENT_ID_COL].unique())
        test_pids  = set(df_test[PATIENT_ID_COL].unique())

        self.assertEqual(len(train_pids & test_pids), 0, "Patient leakage between Train and Test splits!")
        self.assertEqual(len(val_pids & test_pids), 0, "Patient leakage between Val and Test splits!")

    def test_3_standalone_execution_and_artifacts(self):
        """TEST 3: Verify evaluate.py runs standalone and creates all required report and plot files."""
        report = run_standalone_evaluation()

        self.assertIsNotNone(report)
        self.assertIn("metrics", report)
        self.assertIn("operating_threshold", report)
        self.assertIn("calibration", report)

        required_plots = [
            "confusion_matrix.png",
            "roc_pr_curves.png",
            "calibration_curve.png",
            "threshold_analysis.png"
        ]
        for plot_file in required_plots:
            plot_path = os.path.join(ROOT_DIR, plot_file)
            self.assertTrue(os.path.exists(plot_path), f"Expected plot file missing: {plot_file}")

        self.assertTrue(os.path.exists(os.path.join(ROOT_DIR, "metrics_report.json")))
        self.assertTrue(os.path.exists(os.path.join(ROOT_DIR, "metrics_report.md")))

    def test_4_evaluation_reproducibility(self):
        """TEST 4: Verify evaluation metrics are 100% reproducible across repeated runs."""
        r1 = run_standalone_evaluation()
        r2 = run_standalone_evaluation()

        self.assertEqual(r1["metrics"]["confusion_matrix"], r2["metrics"]["confusion_matrix"])
        self.assertAlmostEqual(r1["metrics"]["roc_auc"], r2["metrics"]["roc_auc"], places=6)
        self.assertAlmostEqual(r1["metrics"]["pr_auc"], r2["metrics"]["pr_auc"], places=6)
        self.assertAlmostEqual(r1["metrics"]["brier_score"], r2["metrics"]["brier_score"], places=6)

if __name__ == '__main__':
    unittest.main()
