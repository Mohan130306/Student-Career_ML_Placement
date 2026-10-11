"""
Unit Tests for Benchmark Dataset Generator (Phase 1)
AI-Powered Student Career & Placement Prediction System
"""

import pytest
import pandas as pd
import numpy as np

from src.data.generate_dataset import (
    ALL_DATASET_COLUMNS,
    COL_STUDENT_ID,
    COL_DEPARTMENT,
    COL_GENDER,
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
    COL_PLACEMENT_STATUS,
    COL_PLACED_ROLE,
    ROLE_NOT_PLACED,
    generate_placement_dataset,
)


@pytest.fixture(scope="module")
def benchmark_df() -> pd.DataFrame:
    """Fixture providing a standard 200-sample generated benchmark dataset."""
    return generate_placement_dataset(n_samples=200, random_state=42)


def test_schema_and_dimensions(benchmark_df: pd.DataFrame):
    """Verifies row count, column count, and exact column naming."""
    assert len(benchmark_df) == 200
    assert len(benchmark_df.columns) == len(ALL_DATASET_COLUMNS)
    assert list(benchmark_df.columns) == ALL_DATASET_COLUMNS


def test_no_missing_values(benchmark_df: pd.DataFrame):
    """Ensures generator produces complete data without unhandled NaNs."""
    assert benchmark_df.isnull().sum().sum() == 0


def test_numeric_ranges(benchmark_df: pd.DataFrame):
    """Verifies that all continuous features stay strictly within domain bounds."""
    # Academics
    assert benchmark_df[COL_10TH_PERCENT].between(50.0, 100.0).all()
    assert benchmark_df[COL_12TH_PERCENT].between(50.0, 100.0).all()
    assert benchmark_df[COL_CGPA].between(5.0, 10.0).all()

    # Skill scores (0 to 100)
    score_cols = [
        COL_DSA_SCORE,
        COL_JAVA_SCORE,
        COL_PYTHON_SCORE,
        COL_SQL_SCORE,
        COL_WEB_DEV_SCORE,
        COL_APTITUDE_SCORE,
        COL_CODING_SCORE,
        COL_COMMUNICATION_SCORE,
    ]
    for col in score_cols:
        assert benchmark_df[col].between(0.0, 100.0).all(), f"{col} has values outside [0, 100]"

    # Counts
    assert (benchmark_df[COL_PROJECTS_COUNT] >= 0).all()
    assert (benchmark_df[COL_INTERNSHIPS_COUNT] >= 0).all()
    assert (benchmark_df[COL_CERTIFICATIONS_COUNT] >= 0).all()
    assert (benchmark_df[COL_LEETCODE_SOLVED] >= 0).all()

    # Binary flags
    for col in [COL_HISTORY_OF_ARREARS, COL_GITHUB_PROFILE, COL_LINKEDIN_PROFILE, COL_PLACEMENT_STATUS]:
        assert set(benchmark_df[col].unique()).issubset({0, 1})


def test_logical_consistency(benchmark_df: pd.DataFrame):
    """Tests domain invariant rules."""
    # 1. Arrears logic: No current arrears if no history of arrears
    no_history = benchmark_df[benchmark_df[COL_HISTORY_OF_ARREARS] == 0]
    assert (no_history[COL_CURRENT_ARREARS] == 0).all()

    # 2. Placement target vs Role consistency:
    # Unplaced students must have role 'Not Placed'
    unplaced = benchmark_df[benchmark_df[COL_PLACEMENT_STATUS] == 0]
    assert (unplaced[COL_PLACED_ROLE] == ROLE_NOT_PLACED).all()

    # Placed students must have a valid role and NOT 'Not Placed'
    placed = benchmark_df[benchmark_df[COL_PLACEMENT_STATUS] == 1]
    assert (placed[COL_PLACED_ROLE] != ROLE_NOT_PLACED).all()


def test_statistical_correlations(benchmark_df: pd.DataFrame):
    """Verifies that placed students have demonstrably higher metrics than unplaced."""
    placed = benchmark_df[benchmark_df[COL_PLACEMENT_STATUS] == 1]
    unplaced = benchmark_df[benchmark_df[COL_PLACEMENT_STATUS] == 0]

    assert placed[COL_CODING_SCORE].mean() > unplaced[COL_CODING_SCORE].mean()
    assert placed[COL_DSA_SCORE].mean() > unplaced[COL_DSA_SCORE].mean()
    assert placed[COL_CGPA].mean() > unplaced[COL_CGPA].mean()
    assert placed[COL_APTITUDE_SCORE].mean() > unplaced[COL_APTITUDE_SCORE].mean()


def test_deterministic_reproducibility():
    """Ensures same random seed produces bit-identical DataFrame."""
    df1 = generate_placement_dataset(n_samples=50, random_state=123)
    df2 = generate_placement_dataset(n_samples=50, random_state=123)
    pd.testing.assert_frame_equal(df1, df2)
