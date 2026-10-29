import numpy as np
from sklearn.linear_model import Ridge


class DengueForecasterModel:
    """
    Climate-informed epidemic forecasting model.
    Fits in log-space (log1p) with L2 regularization to guarantee non-negative case predictions
    and stable convergence across diverse regional epidemiology.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.regressor = Ridge(alpha=alpha)
        self.feature_names = None
        self.mean_case_fallback = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] = None):
        self.feature_names = feature_names
        y_clean = np.nan_to_num(y, nan=0.0)
        y_clean = np.maximum(0.0, y_clean)
        self.mean_case_fallback = float(np.mean(y_clean))

        # Train on log1p scale: log(y + 1)
        y_log = np.log1p(y_clean)
        X_clean = np.nan_to_num(X, nan=0.0)
        self.regressor.fit(X_clean, y_log)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0.0)
        log_preds = self.regressor.predict(X_clean)
        # Invert from log space: exp(pred) - 1, bounded at 0
        preds = np.expm1(log_preds)
        preds = np.maximum(0.0, preds)
        return preds
