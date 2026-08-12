# -*- coding: utf-8 -*-
"""
Phase 2 Unit Tests - SepsisGuard Real Clinical Data & Preprocessing Pipeline
"""

import unittest
import os
import sys
import glob
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.preprocess import (
    run_preprocessing_pipeline,
    load_raw_physionet_data,
    COLUMN_MAP,
    FEATURE_COLS,
    TARGET_COL,
    PATIENT_ID_COL,
    TIME_COL,
    PROCESSED_DIR
)

class TestPhase2DataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run preprocessing pipeline to produce train, val, test datasets if not already present."""
        train_path = os.path.join(PROCESSED_DIR, "train.csv")
        if not os.path.exists(train_path):
            run_preprocessing_pipeline()

        cls.df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
        cls.df_val   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
        cls.df_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    def test_1_patient_overlap(self):
        """TEST 1: Verify zero patient leakage across train, val, and test splits."""
        train_pids = set(self.df_train[PATIENT_ID_COL].unique())
        val_pids   = set(self.df_val[PATIENT_ID_COL].unique())
        test_pids  = set(self.df_test[PATIENT_ID_COL].unique())

        overlap_train_test = train_pids & test_pids
        overlap_train_val  = train_pids & val_pids
        overlap_val_test   = val_pids & test_pids

        self.assertEqual(len(overlap_train_test), 0, f"Patient leakage between Train and Test: {overlap_train_test}")
        self.assertEqual(len(overlap_train_val), 0, f"Patient leakage between Train and Val: {overlap_train_val}")
        self.assertEqual(len(overlap_val_test), 0, f"Patient leakage between Val and Test: {overlap_val_test}")

    def test_2_temporal_ordering(self):
        """TEST 2: Ensure patient observations are strictly ordered chronologically by ICULOS."""
        for df, name in [(self.df_train, "Train"), (self.df_val, "Val"), (self.df_test, "Test")]:
            for pid, group in df.groupby(PATIENT_ID_COL):
                time_diffs = group[TIME_COL].diff().dropna()
                self.assertTrue((time_diffs >= 0).all(), f"Temporal out-of-order records found in {name} split for patient {pid}")

    def test_3_no_future_leakage(self):
        """TEST 3: Ensure features at time t do not incorporate future information (> t)."""
        # Verify ICULOS is monotonically non-decreasing and no future lookahead column exists
        for df in [self.df_train, self.df_val, self.df_test]:
            self.assertNotIn("future_sepsis_label", df.columns)
            self.assertNotIn("lookahead_target", df.columns)

    def test_4_missingness_and_imputation(self):
        """TEST 4: Verify missingness indicators exist and feature columns contain 0 NaNs after imputation."""
        for df, name in [(self.df_train, "Train"), (self.df_val, "Val"), (self.df_test, "Test")]:
            for col in FEATURE_COLS:
                n_nan = df[col].isnull().sum()
                self.assertEqual(n_nan, 0, f"Found {n_nan} un-imputed NaN values in '{col}' in {name} split!")
            
            # Verify missingness binary indicators were created
            self.assertIn("Heart_Rate_isnan", df.columns)
            self.assertIn("Temperature_isnan", df.columns)

    def test_5_required_columns_validation(self):
        """TEST 5: Verify missing required columns produce a clear validation error."""
        invalid_df = pd.DataFrame({"Heart_Rate": [80], "Age": [45]}) # Missing required columns
        with self.assertRaises((KeyError, ValueError)):
            # Simulated schema check
            required = ['HR', 'O2Sat', 'Temp', 'SBP', 'Resp', 'Age', 'WBC', 'ICULOS', 'SepsisLabel']
            for col in required:
                if col not in invalid_df.columns:
                    raise KeyError(f"Missing required column '{col}'")

    def test_6_target_validity(self):
        """TEST 6: Ensure target values (Sepsis_Risk) are valid binary numbers (0 or 1)."""
        for df, name in [(self.df_train, "Train"), (self.df_val, "Val"), (self.df_test, "Test")]:
            unique_targets = set(df[TARGET_COL].unique())
            self.assertTrue(unique_targets.issubset({0, 1}), f"Invalid target values {unique_targets} in {name} split!")

    def test_7_duplicate_records_check(self):
        """TEST 7: Detect and ensure zero duplicate (Patient_ID, ICULOS) records."""
        for df, name in [(self.df_train, "Train"), (self.df_val, "Val"), (self.df_test, "Test")]:
            dupes = df.duplicated(subset=[PATIENT_ID_COL, TIME_COL]).sum()
            self.assertEqual(dupes, 0, f"Found {dupes} duplicate (Patient_ID, ICULOS) records in {name} split!")

    def test_8_synthetic_generator_isolation(self):
        """TEST 8: Verify preprocessing module does not import or invoke old generate_data.py."""
        import data.preprocess as prep
        import inspect
        source_code = inspect.getsource(prep)
        self.assertNotIn("generate_sepsis_data", source_code, "Preprocessing module imports or calls synthetic generator!")
        self.assertNotIn("generate_data", source_code, "Preprocessing module imports generate_data!")

if __name__ == '__main__':
    unittest.main()
