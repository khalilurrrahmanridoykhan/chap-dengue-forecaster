import math
import pandas as pd
import numpy as np


def parse_month_from_period(period_str: str) -> int:
    """Extract 1-12 month integer from period string like '2010-04' or '2010-W14'."""
    try:
        if "-" in period_str:
            parts = period_str.split("-")
            val = parts[1]
            if val.startswith("W"):
                week = int(val[1:])
                return min(12, max(1, int(math.ceil(week / 4.33))))
            return int(val)
        return 1
    except Exception:
        return 1


def build_lagged_features(
    df: pd.DataFrame,
    rainfall_lags: list[int] = [1, 2],
    temp_lags: list[int] = [1],
    case_lags: list[int] = [1],
) -> pd.DataFrame:
    """
    Constructs climate-lag and seasonal cyclical features for a single location time-series.
    """
    df_feat = df.copy()

    # Fill NaNs in raw columns with forward/backward fill or reasonable defaults
    if "rainfall" in df_feat.columns:
        df_feat["rainfall"] = df_feat["rainfall"].ffill().bfill().fillna(0.0)
    if "mean_temperature" in df_feat.columns:
        df_feat["mean_temperature"] = df_feat["mean_temperature"].ffill().bfill().fillna(25.0)
    if "disease_cases" in df_feat.columns:
        df_feat["disease_cases"] = df_feat["disease_cases"].fillna(0.0)

    # 1. Weather Lags
    for lag in rainfall_lags:
        df_feat[f"rainfall_lag_{lag}"] = df_feat["rainfall"].shift(lag)

    for lag in temp_lags:
        df_feat[f"temp_lag_{lag}"] = df_feat["mean_temperature"].shift(lag)

    # 2. Autoregressive Disease Lags
    if "disease_cases" in df_feat.columns:
        for lag in case_lags:
            df_feat[f"cases_lag_{lag}"] = df_feat["disease_cases"].shift(lag)

    # 3. Seasonal Cyclical Components (sin/cos encoding of month)
    if "time_period" in df_feat.columns:
        months = df_feat["time_period"].astype(str).apply(parse_month_from_period)
        df_feat["sin_month"] = np.sin(2 * np.pi * months / 12.0)
        df_feat["cos_month"] = np.cos(2 * np.pi * months / 12.0)
    else:
        df_feat["sin_month"] = 0.0
        df_feat["cos_month"] = 0.0

    return df_feat
