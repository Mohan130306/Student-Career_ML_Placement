"""
Unit Tests for Exploratory Data Analysis & Diagnostic Engine (Phase 3)
AI-Powered Student Career & Placement Prediction System
"""

import pytest
import pandas as pd
import numpy as np

from src.data.generate_dataset import (
    generate_placement_dataset,
    COL_PLACEMENT_STATUS,
    COL_PLACED_ROLE,
    COL_CODING_SCORE,
    COL_DSA_SCORE,
    COL_CGPA,
    ROLE_SDE,
)
from src.features.build_features import add_domain_features
from src.analysis.eda import (
    compute_correlations,
    compare_cohorts,
    compute_role_skill_profiles,
    analyze_departments,
    analyze_demographic_parity,
)


@pytest.fixture(scope="module")
def sample_featured_df() -> pd.DataFrame:
    """Fixture providing 150 featured student records."""
    raw_df = generate_placement_dataset(n_samples=150, random_state=42)
    return add_domain_features(raw_df)


def test_compute_correlations(sample_featured_df: pd.DataFrame):
    """Verifies correlation calculations and boundary invariants."""
    corr_df = compute_correlations(sample_featured_df)

    assert isinstance(corr_df, pd.DataFrame)
    assert "feature" in corr_df.columns
    assert "correlation" in corr_df.columns

    # All correlation coefficients must lie in [-1, 1]
    assert corr_df["correlation"].between(-1.0, 1.0).all()

    # Coding and DSA should be positively correlated with placement
    pos_feats = corr_df[corr_df["correlation"] > 0]["feature"].tolist()
    assert COL_CODING_SCORE in pos_feats
    assert COL_DSA_SCORE in pos_feats


def test_compare_cohorts(sample_featured_df: pd.DataFrame):
    """Verifies placed vs unplaced difference metrics."""
    cohort_df = compare_cohorts(sample_featured_df)

    assert isinstance(cohort_df, pd.DataFrame)
    assert len(cohort_df) > 5

    coding_row = cohort_df[cohort_df["feature"] == COL_CODING_SCORE].iloc[0]
    assert coding_row["placed_mean"] > coding_row["unplaced_mean"]
    assert coding_row["difference_in_means"] > 0


def test_compute_role_skill_profiles(sample_featured_df: pd.DataFrame):
    """Verifies that role benchmarks are calculated for all placed roles."""
    role_benchmarks = compute_role_skill_profiles(sample_featured_df)

    assert isinstance(role_benchmarks, pd.DataFrame)
    assert len(role_benchmarks) > 0
    assert COL_DSA_SCORE in role_benchmarks.columns
    assert COL_CODING_SCORE in role_benchmarks.columns


def test_analyze_departments(sample_featured_df: pd.DataFrame):
    """Verifies department-level aggregation metrics."""
    dept_df = analyze_departments(sample_featured_df)

    assert isinstance(dept_df, pd.DataFrame)
    assert len(dept_df) == 5  # CSE, IT, ECE, AIDS, CSBS
    assert dept_df["placement_rate_pct"].between(0.0, 100.0).all()
    assert dept_df["total_students"].sum() == len(sample_featured_df)


def test_analyze_demographic_parity(sample_featured_df: pd.DataFrame):
    """Verifies demographic parity and adverse impact calculations."""
    parity = analyze_demographic_parity(sample_featured_df)

    assert "rates_pct" in parity
    assert "counts" in parity
    assert "adverse_impact_ratio" in parity
    assert parity["adverse_impact_ratio"] > 0.0
