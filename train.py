import argparse
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features import build_lagged_features
from src.model import DengueForecasterModel

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


def train(train_csv_path: str, model_save_path: str):
    df = pd.read_csv(train_csv_path)

    # Location column check
    loc_col = "location" if "location" in df.columns else ("orgUnit" if "orgUnit" in df.columns else None)

    models_by_location = {}
    all_X = []
    all_y = []

    if loc_col is not None:
        locations = df[loc_col].unique()
    else:
        locations = ["all"]
        df["location"] = "all"
        loc_col = "location"

    for loc in locations:
        loc_df = df[df[loc_col] == loc].sort_values("time_period").reset_index(drop=True)
        if len(loc_df) < 3:
            continue

        feat_df = build_lagged_features(loc_df)
        # Drop the first 2 rows where lag_2 is NaN
        valid_df = feat_df.iloc[2:].reset_index(drop=True)

        X = valid_df[FEATURE_COLS].to_numpy()
        y = valid_df["disease_cases"].to_numpy()

        loc_model = DengueForecasterModel(alpha=1.0)
        loc_model.fit(X, y, feature_names=FEATURE_COLS)
        models_by_location[loc] = loc_model

        all_X.append(X)
        all_y.append(y)

    # Train global fallback model
    global_model = DengueForecasterModel(alpha=2.0)
    if all_X:
        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        global_model.fit(X_all, y_all, feature_names=FEATURE_COLS)

    payload = {
        "models_by_location": models_by_location,
        "global_model": global_model,
        "features": FEATURE_COLS,
    }

    # Ensure parent dir exists
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Successfully trained {len(models_by_location)} regional models and saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train climate-informed dengue forecasting model.")
    parser.add_argument("train_data", type=str, help="Path to input training CSV dataset.")
    parser.add_argument("model", type=str, help="Path to save trained model artifact.")
    args = parser.parse_args()

    train(args.train_data, args.model)
