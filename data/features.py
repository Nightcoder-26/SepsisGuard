# -*- coding: utf-8 -*-
"""
Clinically Motivated Derived Feature Engineering (Phase 3)
Strictly causal: For prediction at time t, only information from time <= t is used.
"""

import pandas as pd
import numpy as np

DERIVED_FEATURE_COLS = [
    'Heart_Rate_trend_6h',
    'Resp_Rate_trend_6h',
    'Mean_Arterial_Pressure_trend_6h',
    'ICU_Length_of_Stay'
]

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds clinically motivated causal derived features to the input DataFrame:
    1. Heart_Rate_trend_6h: 6-hour rolling trend (HR(t) - HR(t-5 or t_start))
    2. Resp_Rate_trend_6h: 6-hour rolling trend (RR(t) - RR(t-5 or t_start))
    3. Mean_Arterial_Pressure_trend_6h: 6-hour rolling trend (MAP(t) - MAP(t-5 or t_start))
    4. ICU_Length_of_Stay: Time since ICU admission (already in dataset, preserved)
    
    Causality Guarantee:
    Grouped by Patient_ID and sorted by ICU_Length_of_Stay.
    At row index i (time t), calculations ONLY use rows <= i (times <= t).
    No lookahead or future observations are ever accessed.
    """
    df_out = df.copy()

    # Ensure chronological order per patient
    if 'Patient_ID' in df_out.columns and 'ICU_Length_of_Stay' in df_out.columns:
        df_out = df_out.sort_values(by=['Patient_ID', 'ICU_Length_of_Stay']).reset_index(drop=True)

    def _calc_6h_trend(series: pd.Series) -> pd.Series:
        # Shift 5 periods back for 6-hour window (hours t-5 to t)
        # For initial hours (index < 5), compare against initial observation (index 0)
        shifted = series.shift(5)
        first_val = series.iloc[0] if len(series) > 0 else 0.0
        shifted = shifted.fillna(first_val)
        return series - shifted

    if 'Patient_ID' in df_out.columns:
        df_out['Heart_Rate_trend_6h'] = df_out.groupby('Patient_ID')['Heart_Rate'].transform(_calc_6h_trend)
        df_out['Resp_Rate_trend_6h'] = df_out.groupby('Patient_ID')['Resp_Rate'].transform(_calc_6h_trend)
        df_out['Mean_Arterial_Pressure_trend_6h'] = df_out.groupby('Patient_ID')['Mean_Arterial_Pressure'].transform(_calc_6h_trend)
    else:
        df_out['Heart_Rate_trend_6h'] = _calc_6h_trend(df_out['Heart_Rate'])
        df_out['Resp_Rate_trend_6h'] = _calc_6h_trend(df_out['Resp_Rate'])
        df_out['Mean_Arterial_Pressure_trend_6h'] = _calc_6h_trend(df_out['Mean_Arterial_Pressure'])

    return df_out
