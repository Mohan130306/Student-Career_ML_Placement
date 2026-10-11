"""
Placement Prediction & Probability Scoring Inference Engine
AI-Powered Student Career & Placement Prediction System

Provides production-ready inference:
- Loads fitted preprocessor and champion model
- Ingests raw student profile (dictionary or DataFrame)
- Computes domain features and transformations
- Produces placement classification, calibrated probability, and readiness tier
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import argparse
import json
import joblib
import numpy as np
import pandas as pd

from src.data.generate_dataset import (
    COL_STUDENT_ID,
    COL_GENDER,
    COL_PLACEMENT_STATUS,
    COL_PLACED_ROLE,
)
from src.features.build_features import add_domain_features
from src.features.pipeline import get_feature_names_from_preprocessor


TIER_HIGH = "High Placement Readiness"
TIER_MODERATE = "Moderate Placement Readiness"
TIER_AT_RISK = "At-Risk / Immediate Support Recommended"


class PlacementPredictor:
    """
    Inference class encapsulating the trained preprocessing and model pipeline.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/champion_model.joblib",
        preprocessor_path: Union[str, Path] = "models/preprocessor.joblib",
    ):
        self.model_path = Path(model_path).resolve()
        self.preprocessor_path = Path(preprocessor_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor pipeline not found at {self.preprocessor_path}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.feature_names = get_feature_names_from_preprocessor(self.preprocessor)

    def _prepare_dataframe(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Converts dictionary to DataFrame and adds domain features."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("Input must be a dictionary or pandas DataFrame.")

        # Ensure ID or targets are dropped if present
        cols_to_drop = [c for c in [COL_STUDENT_ID, COL_GENDER, COL_PLACEMENT_STATUS, COL_PLACED_ROLE] if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop)

        # Compute domain features
        df_featured = add_domain_features(df_clean)
        return df_featured

    def predict_proba(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> np.ndarray:
        """Returns array of placement probabilities (P(Placed))."""
        df_featured = self._prepare_dataframe(input_data)
        X_trans = self.preprocessor.transform(df_featured)
        X_df = pd.DataFrame(X_trans, columns=self.feature_names)
        return self.model.predict_proba(X_df)[:, 1]

    def predict_single(self, student_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs comprehensive placement evaluation on a single student profile.

        Args:
            student_profile: Dictionary of student features.

        Returns:
            Dict: Prediction, probability percentage, and readiness tier.
        """
        proba = float(self.predict_proba(student_profile)[0])
        prediction = 1 if proba >= 0.50 else 0

        if proba >= 0.70:
            tier = TIER_HIGH
        elif proba >= 0.45:
            tier = TIER_MODERATE
        else:
            tier = TIER_AT_RISK

        return {
            "prediction": prediction,
            "status": "Placement-Ready" if prediction == 1 else "Not Placed (At-Risk)",
            "placement_probability": round(proba, 4),
            "placement_probability_pct": round(proba * 100.0, 1),
            "readiness_tier": tier,
        }

    def predict_batch(self, df_students: pd.DataFrame) -> pd.DataFrame:
        """
        Runs placement scoring on a batch DataFrame of students.

        Args:
            df_students: DataFrame containing multiple student profiles.

        Returns:
            pd.DataFrame: Original DataFrame augmented with prediction columns.
        """
        probs = self.predict_proba(df_students)
        preds = (probs >= 0.50).astype(int)

        tiers = []
        for p in probs:
            if p >= 0.70:
                tiers.append(TIER_HIGH)
            elif p >= 0.45:
                tiers.append(TIER_MODERATE)
            else:
                tiers.append(TIER_AT_RISK)

        out = df_students.copy()
        out["predicted_placement"] = preds
        out["placement_probability"] = np.round(probs, 4)
        out["placement_probability_pct"] = np.round(probs * 100.0, 1)
        out["readiness_tier"] = tiers

        return out


def predict_student_placement(
    student_profile: Dict[str, Any],
    model_path: str = "models/champion_model.joblib",
    preprocessor_path: str = "models/preprocessor.joblib",
) -> Dict[str, Any]:
    """Convenience function for single student prediction."""
    predictor = PlacementPredictor(model_path=model_path, preprocessor_path=preprocessor_path)
    return predictor.predict_single(student_profile)
