import argparse
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.features import parse_month_from_period

FEATURE_COLS = [
    "rainfall",
    "mean_temperature",
    "rainfall_lag_1",
    "rainfall_lag_2",
    "temp_lag_1",
    "cases_lag_1",
    "sin_month",
    "cos_month",
]


def predict(model_path: str, historic_csv: str, future_csv: str, output_csv: str):
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    models_by_location = payload.get("models_by_location", {})
    global_model = payload.get("global_model")

    hist_df = pd.read_csv(historic_csv)
    fut_df = pd.read_csv(future_csv)

    loc_col = "location" if "location" in fut_df.columns else ("orgUnit" if "orgUnit" in fut_df.columns else None)
    if loc_col is None:
        loc_col = "location"
        fut_df["location"] = "all"
        hist_df["location"] = "all"

    output_dfs = []

    for loc in fut_df[loc_col].unique():
        loc_fut = fut_df[fut_df[loc_col] == loc].sort_values("time_period").copy().reset_index(drop=True)
        loc_hist = hist_df[hist_df[loc_col] == loc].sort_values("time_period").copy().reset_index(drop=True)

        model = models_by_location.get(loc, global_model)

        # Baseline fallbacks if historic data is empty
        last_cases = float(loc_hist["disease_cases"].iloc[-1]) if ("disease_cases" in loc_hist.columns and len(loc_hist) > 0) else 0.0
        last_rain = float(loc_hist["rainfall"].iloc[-1]) if ("rainfall" in loc_hist.columns and len(loc_hist) > 0) else 50.0
        prev_rain = float(loc_hist["rainfall"].iloc[-2]) if ("rainfall" in loc_hist.columns and len(loc_hist) > 1) else last_rain
        last_temp = float(loc_hist["mean_temperature"].iloc[-1]) if ("mean_temperature" in loc_hist.columns and len(loc_hist) > 0) else 25.0

        predictions = []

        curr_prev_rain = prev_rain
        curr_last_rain = last_rain
        curr_last_temp = last_temp
        curr_last_cases = last_cases

        for idx, row in loc_fut.iterrows():
            r = float(row.get("rainfall", curr_last_rain))
            t = float(row.get("mean_temperature", curr_last_temp))
            period = str(row.get("time_period", ""))

            m = parse_month_from_period(period)
            sin_m = np.sin(2 * np.pi * m / 12.0)
            cos_m = np.cos(2 * np.pi * m / 12.0)

            feat_vector = np.array([[
                r,
                t,
                curr_last_rain,
                curr_prev_rain,
                curr_last_temp,
                curr_last_cases,
                sin_m,
                cos_m,
            ]])

            pred_val = float(model.predict(feat_vector)[0]) if model is not None else 0.0
            pred_val = max(0.0, pred_val)
            predictions.append(pred_val)

            # Advance state for next step
            curr_prev_rain = curr_last_rain
            curr_last_rain = r
            curr_last_temp = t
            curr_last_cases = pred_val

        loc_fut["sample_0"] = predictions
        output_dfs.append(loc_fut)

    final_df = pd.concat(output_dfs, ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Predictions successfully written to: {output_csv} ({len(final_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using trained dengue forecasting model.")
    parser.add_argument("model", type=str, help="Path to trained model file.")
    parser.add_argument("historic_data", type=str, help="Path to historic CSV data.")
    parser.add_argument("future_data", type=str, help="Path to future climate CSV data.")
    parser.add_argument("out_file", type=str, help="Path to write output predictions CSV.")
    args = parser.parse_args()

    predict(args.model, args.historic_data, args.future_data, args.out_file)
