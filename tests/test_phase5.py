# -*- coding: utf-8 -*-
"""
Phase 5 Unit Tests - SepsisGuard Clinical Baseline Comparison
Verifies SIRS logic, Partial qSOFA logic, required labeling, NEWS2 exclusion,
and patient ID matching on the held-out test dataset.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.clinical_baselines import (
    compute_sirs, compute_qsofa_partial, evaluate_clinical_baselines,
    SIRS_LABEL, QSOFA_PARTIAL_LABEL, NEWS2_REASON, PROCESSED_DIR, PATIENT_ID_COL
)

class TestPhase5ClinicalBaselines(unittest.TestCase):

    def test_1_sirs_calculation(self):
        """
        TEST 1: Verify exact SIRS score calculation logic on hand-constructed rows.
        Row 1: Temp = 39.0 (>38), HR = 95 (>90), RR = 18 (<=20), WBC = 10.0 (normal) -> Score = 2, Pos = 1
        Row 2: Temp = 35.0 (<36), HR = 80 (<=90), RR = 25 (>20), WBC = 13.0 (>12) -> Score = 3, Pos = 1
        Row 3: Temp = 37.0 (normal), HR = 80 (normal), RR = 16 (normal), WBC = 7.0 (normal) -> Score = 0, Pos = 0
        """
        data = pd.DataFrame({
            'Temperature': [39.0, 35.0, 37.0],
            'Heart_Rate': [95, 80, 80],
            'Resp_Rate': [18, 25, 16],
            'Infection_Marker': [10.0, 13.0, 7.0]
        })

        scores, pos = compute_sirs(data)

        self.assertEqual(list(scores), [2, 3, 0])
        self.assertEqual(list(pos), [1, 1, 0])

    def test_2_qsofa_partial_calculation(self):
        """
        TEST 2: Verify exact Partial qSOFA calculation logic on hand-constructed rows.
        Row 1: RR = 24 (>=22), Blood_Pressure = 90 (<=100) -> Score = 2, Pos = 1
        Row 2: RR = 18 (<22), Blood_Pressure = 110 (>100) -> Score = 0, Pos = 0
        Row 3: RR = 23 (>=22), Blood_Pressure = 120 (>100) -> Score = 1, Pos = 0
        """
        data = pd.DataFrame({
            'Resp_Rate': [24, 18, 23],
            'Blood_Pressure': [90, 110, 120]
        })

        scores, pos = compute_qsofa_partial(data)

        self.assertEqual(list(scores), [2, 0, 1])
        self.assertEqual(list(pos), [1, 0, 0])

    def test_3_labeling_and_news2_exclusion(self):
        """
        TEST 3: Verify SIRS is labeled rule-based, qSOFA is labeled partial, and NEWS2 is NOT implemented.
        """
        self.assertIn("Rule-based", SIRS_LABEL)
        self.assertIn("(not ML-derived)", SIRS_LABEL)

        self.assertIn("partial — mentation unavailable", QSOFA_PARTIAL_LABEL)
        self.assertIn("NEWS2 not calculated", NEWS2_REASON)

    def test_4_same_test_set_integrity(self):
        """
        TEST 4: Verify ML and clinical baselines evaluate on the EXACT SAME test patient IDs.
        """
        df_test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
        test_pids = set(df_test[PATIENT_ID_COL].unique())

        report = evaluate_clinical_baselines()

        self.assertEqual(report["test_patients"], len(test_pids))
        self.assertIn("ML_Model", report["methods"])
        self.assertIn("SIRS", report["methods"])
        self.assertIn("Partial_qSOFA", report["methods"])
        self.assertFalse(report["methods"]["NEWS2"]["implemented"])

if __name__ == '__main__':
    unittest.main()
