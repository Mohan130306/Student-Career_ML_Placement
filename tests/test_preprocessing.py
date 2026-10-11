"""
Unit Tests for Data Preprocessing & Feature Engineering Pipeline (Phase 2)
AI-Powered Student Career & Placement Prediction System
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

from src.data.generate_dataset import generate_placement_dataset, COL_PLACEMENT_STATUS
from src.features.build_features import (
    add_domain_features,
    DomainFeatureTransformer,
    ENGINEERED_FEATURE_COLUMNS,
    COL_FE_TECHNICAL_AGGREGATE,
    COL_FE_CODING_DEXTERITY,
    COL_FE_PROFILE_RICHNESS,
    COL_FE_ACADEMIC_CONSISTENCY,
    COL_FE_ARREAR_BURDEN,
)
from src.features.pipeline import (
    create_preprocessing_pipeline,
    get_feature_names_from_preprocessor,
)
from src.features.preprocess import run_preprocessing_pipeline


@pytest.fixture(scope="module")
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing 100 sample raw student records."""
    return generate_placement_dataset(n_samples=100, random_state=42)


def test_add_domain_features_columns(sample_raw_df: pd.DataFrame):
    """Verifies that all 5 engineered features are properly calculated and attached."""
    featured_df = add_domain_features(sample_raw_df)

    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in featured_df.columns, f"Engineered column {col} missing from output."

    assert featured_df[COL_FE_TECHNICAL_AGGREGATE].between(0.0, 100.0).all()
    assert featured_df[COL_FE_CODING_DEXTERITY].between(0.0, 100.0).all()
    assert featured_df[COL_FE_PROFILE_RICHNESS].between(0.0, 100.0).all()
    assert (featured_df[COL_FE_ACADEMIC_CONSISTENCY] > 0.0).all()
    assert (featured_df[COL_FE_ARREAR_BURDEN] >= 0.0).all()


def test_domain_feature_transformer_sklearn_api(sample_raw_df: pd.DataFrame):
    """Verifies DomainFeatureTransformer conforms to sklearn TransformerMixin API."""
    transformer = DomainFeatureTransformer()
    transformer.fit(sample_raw_df)
    transformed_df = transformer.transform(sample_raw_df)

    assert isinstance(transformed_df, pd.DataFrame)
    assert len(transformed_df) == len(sample_raw_df)
    assert COL_FE_TECHNICAL_AGGREGATE in transformed_df.columns


def test_column_transformer_fitting_and_imputation(sample_raw_df: pd.DataFrame):
    """Verifies ColumnTransformer correctly scales, encodes, and imputes missing values."""
    featured_df = add_domain_features(sample_raw_df)

    # Inject deliberate NaNs into a copy to verify imputer robustness
    corrupted_df = featured_df.copy()
    corrupted_df.loc[0, "cgpa"] = np.nan
    corrupted_df.loc[1, "dsa_score"] = np.nan
    corrupted_df.loc[2, "department"] = np.nan

    preprocessor = create_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(corrupted_df)
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # 24 numeric features + 5 one-hot departments = 29
    assert transformed.shape[1] == len(feature_names)
    assert np.isnan(transformed).sum() == 0, "Transformed data must not contain NaNs after imputation."


def test_run_preprocessing_pipeline_end_to_end(tmp_path: Path):
    """Verifies complete end-to-end preprocessing run and artifact generation in temporary directory."""
    raw_csv = tmp_path / "raw.csv"
    proc_dir = tmp_path / "processed"
    model_dir = tmp_path / "models"

    sample_df = generate_placement_dataset(n_samples=150, random_state=42)
    sample_df.to_csv(raw_csv, index=False)

    summary = run_preprocessing_pipeline(
        raw_data_path=raw_csv,
        output_dir=proc_dir,
        model_dir=model_dir,
        test_size=0.20,
        random_state=42,
    )

    assert summary["n_train"] == 120
    assert summary["n_test"] == 30
    assert summary["n_features_out"] == 29

    # Verify generated files exist
    assert (proc_dir / "X_train.csv").exists()
    assert (proc_dir / "X_test.csv").exists()
    assert (proc_dir / "y_train.csv").exists()
    assert (proc_dir / "y_test.csv").exists()
    assert (model_dir / "preprocessor.joblib").exists()

    # Verify preprocessor deserialization
    loaded_pipe = joblib.load(model_dir / "preprocessor.joblib")
    assert hasattr(loaded_pipe, "transform")
