"""
Unit Tests for Data Audit Pipeline (Phase 2A)
AI-Powered Student Career & Placement Prediction System

NOTE: All test datasets below are synthetic, minimal test fixtures created
exclusively to verify auditing logic and edge cases. They do NOT contain real student data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.audit_dataset import (
    EXPECTED_COLUMNS,
    COL_STUDENT_NAME,
    COL_REGISTER_NUMBER,
    COL_GENDER,
    COL_10TH_PERCENT,
    COL_10TH_YEAR,
    COL_DIPLOMA,
    COL_12TH_OR_DIPLOMA_PERCENT,
    COL_12TH_CUTOFF,
    COL_CGPA,
    COL_RESUME_LINK,
    COL_HISTORY_OF_ARREAR,
    COL_CURRENT_ARREAR,
    COL_NO_OF_CURRENT_ARREARS,
    COL_LINKEDIN_ID,
    COL_HACKERRANK_ID,
    COL_GITHUB_ID,
    COL_LEETCODE_SOLVED,
    check_schema_and_columns,
    check_missing_values,
    check_duplicate_records,
    check_numeric_ranges_and_validity,
    check_categorical_distributions,
    check_profile_link_availability,
    check_logical_inconsistencies,
    audit_student_dataset,
    format_audit_summary,
)
from src.data.load_excel import validate_excel_path


# =====================================================================
# SYNTHETIC TEST FIXTURES (TEST FIXTURES ONLY — NOT PROJECT DATA)
# =====================================================================

@pytest.fixture
def valid_mock_student_df() -> pd.DataFrame:
    """Creates a minimal valid DataFrame strictly conforming to the 17 columns."""
    return pd.DataFrame({
        COL_STUDENT_NAME: ["TEST STUDENT A", "TEST STUDENT B"],
        COL_REGISTER_NUMBER: ["REG99901", "REG99902"],
        COL_GENDER: ["Male", "Female"],
        COL_10TH_PERCENT: [88.5, 92.0],
        COL_10TH_YEAR: [2019, 2019],
        COL_DIPLOMA: ["No", "No"],
        COL_12TH_OR_DIPLOMA_PERCENT: [85.0, 90.5],
        COL_12TH_CUTOFF: [175.5, 185.0],
        COL_CGPA: [8.2, 8.9],
        COL_RESUME_LINK: ["https://example.com/resumeA", "https://example.com/resumeB"],
        COL_HISTORY_OF_ARREAR: ["No", "Yes"],
        COL_CURRENT_ARREAR: ["No", "No"],
        COL_NO_OF_CURRENT_ARREARS: [0, 0],
        COL_LINKEDIN_ID: ["https://linkedin.com/in/testA", "https://linkedin.com/in/testB"],
        COL_HACKERRANK_ID: ["testuserA", "testuserB"],
        COL_GITHUB_ID: ["https://github.com/testA", "https://github.com/testB"],
        COL_LEETCODE_SOLVED: [120, 250],
    })


# =====================================================================
# 1. SCHEMA AND COLUMN TESTS
# =====================================================================

def test_schema_exact_match(valid_mock_student_df):
    """Verifies that a valid schema passes with exact match True."""
    report = check_schema_and_columns(valid_mock_student_df)
    assert report["exact_match"] is True
    assert report["contains_all_expected"] is True
    assert len(report["missing_columns"]) == 0
    assert len(report["unexpected_columns"]) == 0
    assert report["column_count"] == 17


def test_schema_missing_column(valid_mock_student_df):
    """Verifies detection of missing required columns."""
    dropped_df = valid_mock_student_df.drop(columns=[COL_CGPA, COL_REGISTER_NUMBER])
    report = check_schema_and_columns(dropped_df)
    assert report["exact_match"] is False
    assert report["contains_all_expected"] is False
    assert COL_CGPA in report["missing_columns"]
    assert COL_REGISTER_NUMBER in report["missing_columns"]


def test_schema_unexpected_column(valid_mock_student_df):
    """Verifies detection of unexpected extra columns."""
    extra_df = valid_mock_student_df.copy()
    extra_df["EXTRA_COLUMN"] = [1, 2]
    report = check_schema_and_columns(extra_df)
    assert report["exact_match"] is False
    assert "EXTRA_COLUMN" in report["unexpected_columns"]


# =====================================================================
# 2. MISSING VALUES AND PLACEHOLDERS TESTS
# =====================================================================

def test_missing_values_detection(valid_mock_student_df):
    """Verifies that NaNs, empty strings, and placeholders like 'N/A' are detected."""
    df_with_missing = valid_mock_student_df.copy()
    df_with_missing.loc[0, COL_RESUME_LINK] = None
    df_with_missing.loc[1, COL_LINKEDIN_ID] = "N/A"
    df_with_missing.loc[0, COL_10TH_PERCENT] = np.nan
    df_with_missing.loc[1, COL_HACKERRANK_ID] = "  -  "

    report = check_missing_values(df_with_missing)
    missing_by_col = report["missing_by_column"]

    assert missing_by_col[COL_RESUME_LINK]["missing_count"] == 1
    assert missing_by_col[COL_LINKEDIN_ID]["missing_count"] == 1
    assert missing_by_col[COL_10TH_PERCENT]["missing_count"] == 1
    assert missing_by_col[COL_HACKERRANK_ID]["missing_count"] == 1
    assert report["complete_rows_count"] == 0


# =====================================================================
# 3. DUPLICATE DETECTION TESTS
# =====================================================================

def test_exact_duplicate_rows(valid_mock_student_df):
    """Verifies detection of duplicate rows."""
    duplicated_df = pd.concat([valid_mock_student_df, valid_mock_student_df.iloc[[0]]], ignore_index=True)
    report = check_duplicate_records(duplicated_df)
    assert report["exact_duplicate_rows_count"] == 1
    assert report["exact_duplicate_affected_rows"] == 2


def test_duplicate_register_numbers(valid_mock_student_df):
    """Verifies detection of duplicate register numbers even if other fields differ."""
    dup_reg_df = valid_mock_student_df.copy()
    # Assign same register number to second student
    dup_reg_df.loc[1, COL_REGISTER_NUMBER] = dup_reg_df.loc[0, COL_REGISTER_NUMBER]
    report = check_duplicate_records(dup_reg_df)
    assert report["duplicate_register_numbers_count"] == 1
    assert report["duplicate_register_affected_rows"] == 2


# =====================================================================
# 4. NUMERIC RANGE & VALIDATION TESTS
# =====================================================================

def test_percentage_out_of_bounds(valid_mock_student_df):
    """Flags percentage values < 0 or > 100."""
    df_invalid = valid_mock_student_df.copy()
    df_invalid.loc[0, COL_10TH_PERCENT] = 105.0  # > 100
    df_invalid.loc[1, COL_12TH_OR_DIPLOMA_PERCENT] = -5.0  # < 0

    report = check_numeric_ranges_and_validity(df_invalid)
    assert report[COL_10TH_PERCENT]["out_of_bounds_count"] == 1
    assert report[COL_12TH_OR_DIPLOMA_PERCENT]["out_of_bounds_count"] == 1


def test_percentage_non_numeric_entry(valid_mock_student_df):
    """Flags non-numeric text in percentage column."""
    df_invalid = valid_mock_student_df.copy()
    df_invalid[COL_10TH_PERCENT] = df_invalid[COL_10TH_PERCENT].astype(object)
    df_invalid.loc[0, COL_10TH_PERCENT] = "AB"

    report = check_numeric_ranges_and_validity(df_invalid)
    assert report[COL_10TH_PERCENT]["invalid_non_numeric_count"] == 1


def test_cgpa_validation(valid_mock_student_df):
    """Flags CGPA values > 10.0 or negative."""
    df_invalid = valid_mock_student_df.copy()
    df_invalid.loc[0, COL_CGPA] = 12.5  # > 10.0
    df_invalid.loc[1, COL_CGPA] = -1.0  # Negative

    report = check_numeric_ranges_and_validity(df_invalid)
    assert report[COL_CGPA]["out_of_bounds_count"] == 2


def test_arrear_count_validation(valid_mock_student_df):
    """Flags negative arrear counts or non-integer floats."""
    df_invalid = valid_mock_student_df.copy()
    df_invalid[COL_NO_OF_CURRENT_ARREARS] = df_invalid[COL_NO_OF_CURRENT_ARREARS].astype(object)
    df_invalid.loc[0, COL_NO_OF_CURRENT_ARREARS] = -2   # Negative
    df_invalid.loc[1, COL_NO_OF_CURRENT_ARREARS] = 1.5  # Non-integer

    report = check_numeric_ranges_and_validity(df_invalid)
    assert report[COL_NO_OF_CURRENT_ARREARS]["out_of_bounds_count"] == 1  # -2 is < 0
    assert report[COL_NO_OF_CURRENT_ARREARS]["non_integer_count"] == 1    # 1.5 is float


# =====================================================================
# 5. CATEGORICAL DISTRIBUTION TESTS
# =====================================================================

def test_categorical_distributions(valid_mock_student_df):
    """Verifies that unique raw categorical values are recorded accurately."""
    df_cats = valid_mock_student_df.copy()
    df_cats.loc[0, COL_DIPLOMA] = "Yes"
    df_cats.loc[1, COL_DIPLOMA] = "No"

    report = check_categorical_distributions(df_cats)
    assert "Yes" in report[COL_DIPLOMA]["unique_values"]
    assert "No" in report[COL_DIPLOMA]["unique_values"]
    assert report[COL_DIPLOMA]["unique_values_count"] == 2


# =====================================================================
# 6. PROFILE LINK AVAILABILITY TESTS
# =====================================================================

def test_profile_link_availability(valid_mock_student_df):
    """Verifies binary availability flags for social/coding profile links."""
    df_profiles = valid_mock_student_df.copy()
    df_profiles.loc[0, COL_RESUME_LINK] = "https://example.com/res"
    df_profiles.loc[0, COL_LINKEDIN_ID] = "in/user1"
    df_profiles.loc[0, COL_HACKERRANK_ID] = "user1"
    df_profiles.loc[0, COL_GITHUB_ID] = "git1"

    # Student 1 has all missing or placeholders
    df_profiles.loc[1, COL_RESUME_LINK] = None
    df_profiles.loc[1, COL_LINKEDIN_ID] = "N/A"
    df_profiles.loc[1, COL_HACKERRANK_ID] = ""
    df_profiles.loc[1, COL_GITHUB_ID] = "-"

    report = check_profile_link_availability(df_profiles)
    assert report[COL_RESUME_LINK]["available_count"] == 1
    assert report[COL_LINKEDIN_ID]["available_count"] == 1
    assert report[COL_HACKERRANK_ID]["available_count"] == 1
    assert report[COL_GITHUB_ID]["available_count"] == 1
    assert report["all_profiles_present_count"] == 1
    assert report["zero_profiles_present_count"] == 1


# =====================================================================
# 7. LOGICAL INCONSISTENCY TESTS
# =====================================================================

def test_logical_inconsistency_no_arrear_but_positive_count(valid_mock_student_df):
    """Detects Current Arrear = No but Count > 0."""
    df_inconsistent = valid_mock_student_df.copy()
    df_inconsistent.loc[0, COL_CURRENT_ARREAR] = "No"
    df_inconsistent.loc[0, COL_NO_OF_CURRENT_ARREARS] = 2

    report = check_logical_inconsistencies(df_inconsistent)
    rule_ids = [item["rule_id"] for item in report["inconsistencies_list"]]
    assert "LOGIC_01_NO_ARREAR_BUT_COUNT_POSITIVE" in rule_ids


def test_logical_inconsistency_yes_arrear_but_zero_count(valid_mock_student_df):
    """Detects Current Arrear = Yes but Count = 0."""
    df_inconsistent = valid_mock_student_df.copy()
    df_inconsistent.loc[0, COL_CURRENT_ARREAR] = "Yes"
    df_inconsistent.loc[0, COL_NO_OF_CURRENT_ARREARS] = 0

    report = check_logical_inconsistencies(df_inconsistent)
    rule_ids = [item["rule_id"] for item in report["inconsistencies_list"]]
    assert "LOGIC_02_YES_ARREAR_BUT_COUNT_ZERO_OR_MISSING" in rule_ids


def test_logical_inconsistency_no_history_but_active_current(valid_mock_student_df):
    """Detects History = No but Current = Yes."""
    df_inconsistent = valid_mock_student_df.copy()
    df_inconsistent.loc[0, COL_HISTORY_OF_ARREAR] = "No"
    df_inconsistent.loc[0, COL_CURRENT_ARREAR] = "Yes"
    df_inconsistent.loc[0, COL_NO_OF_CURRENT_ARREARS] = 1

    report = check_logical_inconsistencies(df_inconsistent)
    rule_ids = [item["rule_id"] for item in report["inconsistencies_list"]]
    assert "LOGIC_03_NO_HISTORY_BUT_ACTIVE_CURRENT_ARREAR" in rule_ids


# =====================================================================
# 8. PRIVACY COMPLIANCE TESTS
# =====================================================================

def test_privacy_in_summary_formatter(valid_mock_student_df):
    """
    Critical Test: Ensures student names, register numbers, and raw profile URLs
    NEVER appear anywhere in the generated audit summary output.
    """
    full_audit = audit_student_dataset(valid_mock_student_df)
    summary_text = format_audit_summary(full_audit)

    # Assert names are absent
    assert "TEST STUDENT A" not in summary_text
    assert "TEST STUDENT B" not in summary_text

    # Assert register numbers are absent
    assert "REG99901" not in summary_text
    assert "REG99902" not in summary_text

    # Assert URLs are absent
    assert "https://example.com/resumeA" not in summary_text
    assert "https://linkedin.com/in/testA" not in summary_text
    assert "https://github.com/testA" not in summary_text


# =====================================================================
# 9. EXCEL LOADER AND PATH VALIDATION TESTS
# =====================================================================

def test_validate_excel_path_nonexistent():
    """Ensures FileNotFoundError is raised with helpful message when path is invalid."""
    with pytest.raises(FileNotFoundError, match="Excel dataset not found"):
        validate_excel_path("non_existent_student_database.xlsx")


def test_validate_excel_path_bad_extension(tmp_path):
    """Ensures ValueError is raised for non-Excel extensions."""
    dummy_txt = tmp_path / "data.txt"
    dummy_txt.write_text("dummy content")
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_excel_path(dummy_txt)
