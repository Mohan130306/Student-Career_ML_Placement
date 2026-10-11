"""
Preprocessing Pipeline Module
AI-Powered Student Career & Placement Prediction System

Constructs leak-free Scikit-learn ColumnTransformer pipelines combining:
- Robust imputation (median for continuous, mode for categorical)
- OneHotEncoding for nominal attributes (department)
- StandardScaler for numeric features
- Feature names retrieval for Explainable AI (SHAP)
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.generate_dataset import (
    COL_DEPARTMENT,
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
from src.features.build_features import ENGINEERED_FEATURE_COLUMNS


# Categorical feature columns
CATEGORICAL_FEATURES: List[str] = [
    COL_DEPARTMENT,
]

# Base numeric features (excluding engineered)
BASE_NUMERIC_FEATURES: List[str] = [
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
]

# Complete set of numeric features including engineered columns
ALL_NUMERIC_FEATURES: List[str] = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURE_COLUMNS


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Creates an unfitted ColumnTransformer pipeline for student features.

    Returns:
        ColumnTransformer: Preprocessing transformer ready for fitting on X_train.
    """
    # Numeric Pipeline: Median Imputation -> Standard Scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Pipeline: Most frequent Imputation -> One-Hot Encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts the feature names produced by the fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer instance.

    Returns:
        List[str]: Clean list of feature names.
    """
    feature_names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            encoder = trans.named_steps.get("encoder")
            if encoder is not None and hasattr(encoder, "get_feature_names_out"):
                feature_names.extend(encoder.get_feature_names_out(cols).tolist())
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)

    return feature_names
