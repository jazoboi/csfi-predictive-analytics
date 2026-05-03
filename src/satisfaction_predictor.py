"""
CSFI satisfaction score prediction model.

XGBoost regressor with hyperparameter tuning and
cross-validated performance evaluation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Model performance metrics."""
    r2: float
    mae: float
    rmse: float
    cv_r2_mean: float
    cv_r2_std: float


class SatisfactionPredictor:
    """XGBoost-based CSFI score predictor.

    Parameters
    ----------
    tune_hyperparams : bool
        Whether to run grid search (default: False for speed).
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(self, tune_hyperparams: bool = False) -> None:
        self._tune = tune_hyperparams
        self._model: xgb.XGBRegressor | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> PredictionMetrics:
        """Train the satisfaction predictor.

        Parameters
        ----------
        X_train, y_train : array-like
            Training features and target CSFI scores.
        X_val, y_val : array-like, optional
            Validation set for early stopping.

        Returns
        -------
        PredictionMetrics
            Cross-validated performance metrics.
        """
        if self._tune:
            self._model = self._grid_search(X_train, y_train)
        else:
            self._model = xgb.XGBRegressor(**self.DEFAULT_PARAMS)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Cross-validation
        cv_scores = cross_val_score(
            self._model, X_train, y_train, cv=5, scoring="r2"
        )

        y_pred = self._model.predict(X_val if X_val is not None else X_train)
        y_true = y_val if y_val is not None else y_train

        return PredictionMetrics(
            r2=round(r2_score(y_true, y_pred), 4),
            mae=round(mean_absolute_error(y_true, y_pred), 4),
            rmse=round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            cv_r2_mean=round(float(cv_scores.mean()), 4),
            cv_r2_std=round(float(cv_scores.std()), 4),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate CSFI score predictions."""
        return self._model.predict(X)

    @property
    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        importance = self._model.feature_importances_
        names = self._model.get_booster().feature_names
        return dict(sorted(zip(names, importance), key=lambda x: x[1], reverse=True))

    def _grid_search(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
        """Hyperparameter tuning via grid search."""
        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 300, 500],
        }
        base = xgb.XGBRegressor(**{k: v for k, v in self.DEFAULT_PARAMS.items()
                                    if k not in param_grid})
        grid = GridSearchCV(base, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid.fit(X, y)
        logger.info("Best params: %s (R²=%.4f)", grid.best_params_, grid.best_score_)
        return grid.best_estimator_
