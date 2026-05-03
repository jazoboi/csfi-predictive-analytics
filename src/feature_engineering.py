"""
Feature engineering for CSFI survey data.

Transforms raw survey responses, operational metrics, and
demographic data into ML-ready feature matrices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Engineered feature matrix with metadata."""
    X: pd.DataFrame
    feature_names: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    scaler: StandardScaler | None = None


class CSFIFeatureEngineer:
    """Transforms raw CSFI data into ML features.

    Handles missing values, categorical encoding, temporal
    features, and interaction terms.
    """

    # Expected raw columns
    SURVEY_COLS = [
        "department_id", "fiscal_year", "service_level_score",
        "response_time_days", "employee_satisfaction", "workload_index",
        "budget_utilization", "staffing_ratio", "org_size",
    ]

    def __init__(self, scale: bool = True) -> None:
        self._scale = scale
        self._scaler = StandardScaler() if scale else None
        self._encoders: dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame) -> FeatureSet:
        """Fit on training data and return engineered features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw survey data with CSFI scores and covariates.

        Returns
        -------
        FeatureSet
            ML-ready feature matrix.
        """
        df = df.copy()

        # Handle missing values
        df = self._impute(df)

        # Temporal features
        df = self._add_temporal_features(df)

        # Interaction terms
        df = self._add_interactions(df)

        # Encode categoricals
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            self._encoders[col] = LabelEncoder()
            df[col] = self._encoders[col].fit_transform(df[col].astype(str))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if self._scale and self._scaler:
            df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols])

        return FeatureSet(
            X=df,
            feature_names=df.columns.tolist(),
            categorical_features=categorical_cols,
            numeric_features=numeric_cols,
            scaler=self._scaler,
        )

    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with domain-appropriate strategies."""
        # Numeric: median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Categorical: mode imputation
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

        return df

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add fiscal year-based temporal features."""
        if "fiscal_year" in df.columns:
            df["year_trend"] = df["fiscal_year"] - df["fiscal_year"].min()
            df["year_squared"] = df["year_trend"] ** 2
        return df

    @staticmethod
    def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-relevant interaction terms."""
        if "service_level_score" in df.columns and "response_time_days" in df.columns:
            df["service_efficiency"] = (
                df["service_level_score"] / (df["response_time_days"] + 1)
            )
        if "budget_utilization" in df.columns and "staffing_ratio" in df.columns:
            df["resource_pressure"] = df["budget_utilization"] * df["staffing_ratio"]
        return df
