"""
Domain Feature Engineering Module
AI-Powered Student Career & Placement Prediction System

Constructs domain-specific composite indicators to enrich machine learning models
and career recommendation logic without inducing target leakage.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.generate_dataset import (
    COL_10TH_PERCENT,
    COL_12TH_PERCENT,
    COL_CGPA,
    COL_HISTORY_OF_ARREARS,
    COL_CURRENT_ARREARS,
    COL_DSA_SCORE,
    COL_JAVA_SCORE,
    COL_PYTHON_SCORE,
    COL_SQL_SCORE,
    COL_WEB_DEV_SCORE,
    COL_APTITUDE_SCORE,
    COL_CODING_SCORE,
    COL_COMMUNICATION_SCORE,
    COL_PROJECTS_COUNT,
    COL_INTERNSHIPS_COUNT,
    COL_CERTIFICATIONS_COUNT,
    COL_LEETCODE_SOLVED,
    COL_GITHUB_PROFILE,
    COL_LINKEDIN_PROFILE,
)

# New engineered column constants
COL_FE_TECHNICAL_AGGREGATE = "fe_technical_aggregate"
COL_FE_CODING_DEXTERITY = "fe_coding_dexterity"
COL_FE_PROFILE_RICHNESS = "fe_profile_richness"
COL_FE_ACADEMIC_CONSISTENCY = "fe_academic_consistency"
COL_FE_ARREAR_BURDEN = "fe_arrear_burden"

ENGINEERED_FEATURE_COLUMNS = [
    COL_FE_TECHNICAL_AGGREGATE,
    COL_FE_CODING_DEXTERITY,
    COL_FE_PROFILE_RICHNESS,
    COL_FE_ACADEMIC_CONSISTENCY,
    COL_FE_ARREAR_BURDEN,
]


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-informed composite features on student profiles.

    Args:
        df: Input DataFrame containing base student attributes.

    Returns:
        pd.DataFrame: A copy of the DataFrame with added composite features.
    """
    out = df.copy()

    # 1. Technical Aggregate Index (0 - 100)
    # Weights: DSA (30%), General Coding (25%), SQL (15%), Languages/Web (30%)
    tech_agg = (
        0.30 * out[COL_DSA_SCORE]
        + 0.25 * out[COL_CODING_SCORE]
        + 0.15 * out[COL_SQL_SCORE]
        + 0.15 * ((out[COL_JAVA_SCORE] + out[COL_PYTHON_SCORE]) / 2.0)
        + 0.15 * out[COL_WEB_DEV_SCORE]
    )
    out[COL_FE_TECHNICAL_AGGREGATE] = np.round(np.clip(tech_agg, 0.0, 100.0), 2)

    # 2. Coding Dexterity Index (0 - 100)
    # Synthesizes competitive coding volume (LeetCode problems) with platform assessment score
    # Normalizing 300+ LeetCode problems to ~100 scale
    leetcode_norm = np.clip((out[COL_LEETCODE_SOLVED] / 300.0) * 100.0, 0.0, 100.0)
    coding_dexterity = 0.60 * out[COL_CODING_SCORE] + 0.40 * leetcode_norm
    out[COL_FE_CODING_DEXTERITY] = np.round(np.clip(coding_dexterity, 0.0, 100.0), 2)

    # 3. Profile Richness Score (0 - 100)
    # Measures student proactivity across internships, projects, certifications, and online presence
    profile_score = (
        np.clip(out[COL_PROJECTS_COUNT] * 12.0, 0.0, 36.0)
        + np.clip(out[COL_INTERNSHIPS_COUNT] * 20.0, 0.0, 40.0)
        + np.clip(out[COL_CERTIFICATIONS_COUNT] * 6.0, 0.0, 18.0)
        + out[COL_GITHUB_PROFILE] * 8.0
        + out[COL_LINKEDIN_PROFILE] * 8.0
    )
    out[COL_FE_PROFILE_RICHNESS] = np.round(np.clip(profile_score, 0.0, 100.0), 2)

    # 4. Academic Consistency Ratio (typically 0.70 - 1.25)
    # Ratio of 12th percentage to 10th percentage: measures trajectory stability
    # Avoid zero-division defensively
    tenth_safe = np.maximum(out[COL_10TH_PERCENT], 1.0)
    consistency = out[COL_12TH_PERCENT] / tenth_safe
    out[COL_FE_ACADEMIC_CONSISTENCY] = np.round(consistency, 3)

    # 5. Arrear Burden Index
    # Non-linear penalty reflecting recruiter filtering criteria
    # Active current arrears carry 3x weight compared to historical resolved arrears
    arrear_burden = out[COL_CURRENT_ARREARS] * 3.0 + out[COL_HISTORY_OF_ARREARS] * 0.75
    out[COL_FE_ARREAR_BURDEN] = np.round(arrear_burden, 2)

    return out


class DomainFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer to add domain features within a Pipeline.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless transformation, returns self)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies domain feature engineering to input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DomainFeatureTransformer requires a pandas DataFrame.")
        return add_domain_features(X)
