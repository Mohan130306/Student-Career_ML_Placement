"""
Dataset Audit Module
AI-Powered Student Career & Placement Prediction System

Performs comprehensive data auditing, schema verification, data-quality validation,
and logical-consistency checks on student profile Excel datasets.
Adheres strictly to student data privacy protocols (no PII logged or exposed).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import sys
import numpy as np
import pandas as pd


# =====================================================================
# EXACT EXPECTED COLUMN CONSTANTS (17 Columns)
# =====================================================================
COL_STUDENT_NAME = "NAME OF THE STUDENT (INITIAL AT THE LAST)"
COL_REGISTER_NUMBER = "REGISTER NUMBER"
COL_GENDER = "GENDER"
COL_10TH_PERCENT = "10TH % (ENTER VALUE ALONE)"
COL_10TH_YEAR = "10TH YEAR OF PASSING"
COL_DIPLOMA = "DIPLOMA STUDENT"
COL_12TH_OR_DIPLOMA_PERCENT = "12TH % (ENTER VALUE ALONE) / DIPLOMA%"
COL_12TH_CUTOFF = "12TH CUT OFF"
COL_CGPA = "CGPA (TILL SEMESTER 05)"
COL_RESUME_LINK = "RESUME LINK"
COL_HISTORY_OF_ARREAR = "HISTORY OF ARREAR (YES/NO)"
COL_CURRENT_ARREAR = "CURRENT ARREAR (YES/NO)"
COL_NO_OF_CURRENT_ARREARS = "No. OF CURRENT ARREARS"
COL_LINKEDIN_ID = "LINKED - IN ID"
COL_HACKERRANK_ID = "HACKER RANK ID"
COL_GITHUB_ID = "GIT HUB ID"
COL_LEETCODE_SOLVED = "NO. OF LEET CODE PROGRAMS SOLVED TILL NOW"

EXPECTED_COLUMNS: List[str] = [
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
]

# =====================================================================
# COLUMN CLASSIFICATIONS
# =====================================================================
# Kept strictly for source-record identification. Never use as ML features.
IDENTIFIER_COLUMNS: List[str] = [
    COL_STUDENT_NAME,
    COL_REGISTER_NUMBER,
]

# Kept for bias/fairness analysis, but NOT an initial predictive feature.
PROTECTED_ANALYSIS_COLUMNS: List[str] = [
    COL_GENDER,
]

# Candidate numeric/categorical features for placement outcome models.
CANDIDATE_ML_FEATURES: List[str] = [
    COL_10TH_PERCENT,
    COL_10TH_YEAR,
    COL_DIPLOMA,
    COL_12TH_OR_DIPLOMA_PERCENT,
    COL_12TH_CUTOFF,
    COL_CGPA,
    COL_HISTORY_OF_ARREAR,
    COL_CURRENT_ARREAR,
    COL_NO_OF_CURRENT_ARREARS,
    COL_LEETCODE_SOLVED,
]

# Raw profile links: Do NOT use raw links/IDs as features.
# Used exclusively to derive binary availability flags (0 or 1).
PROFILE_LINK_COLUMNS: List[str] = [
    COL_RESUME_LINK,
    COL_LINKEDIN_ID,
    COL_HACKERRANK_ID,
    COL_GITHUB_ID,
]

# Placeholder strings commonly entered in spreadsheets representing missing values
MISSING_PLACEHOLDER_STRINGS = {
    "",
    "na",
    "n/a",
    "n.a.",
    "nil",
    "none",
    "null",
    "-",
    "--",
    "?",
    "not available",
    "not applicable",
}


def _is_missing_entry(val: Any) -> bool:
    """Checks if a single value is null, NaN, empty string, or a placeholder."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in MISSING_PLACEHOLDER_STRINGS:
            return True
    return False


def _to_anonymized_row_labels(indices: Union[List[int], pd.Index]) -> List[str]:
    """Converts 0-based DataFrame indices to human-readable 1-based Excel row labels."""
    # Row 1 is header in Excel, so row data starts at Excel row index + 2
    return [f"Row {idx + 2} (Index {idx})" for idx in indices]


# =====================================================================
# INDIVIDUAL AUDIT CHECK FUNCTIONS
# =====================================================================

def check_schema_and_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks row count, column count, exact column name matches, and schema ordering.
    """
    actual_columns = list(df.columns)
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in actual_columns]
    unexpected_columns = [col for col in actual_columns if col not in EXPECTED_COLUMNS]

    exact_match = (actual_columns == EXPECTED_COLUMNS)
    contains_all_expected = (len(missing_columns) == 0)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(actual_columns)),
        "expected_column_count": len(EXPECTED_COLUMNS),
        "exact_match": exact_match,
        "contains_all_expected": contains_all_expected,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "actual_columns": actual_columns,
    }


def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates missing value metrics, detecting both standard NaNs and text placeholders.
    """
    total_rows = len(df)
    missing_by_column: Dict[str, Dict[str, Any]] = {}
    total_missing_cells = 0

    for col in df.columns:
        is_missing_series = df[col].apply(_is_missing_entry)
        missing_count = int(is_missing_series.sum())
        total_missing_cells += missing_count
        missing_pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        missing_by_column[str(col)] = {
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "has_missing": missing_count > 0,
        }

    # Count rows with zero missing values across all columns
    if total_rows > 0:
        row_has_missing = df.apply(lambda row: any(_is_missing_entry(v) for v in row), axis=1)
        complete_rows_count = int((~row_has_missing).sum())
    else:
        complete_rows_count = 0

    return {
        "total_cells": int(total_rows * len(df.columns)),
        "total_missing_cells": total_missing_cells,
        "overall_missing_percentage": round(
            (total_missing_cells / (total_rows * len(df.columns)) * 100), 2
        ) if (total_rows * len(df.columns)) > 0 else 0.0,
        "complete_rows_count": complete_rows_count,
        "complete_rows_percentage": round((complete_rows_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "missing_by_column": missing_by_column,
    }


def check_duplicate_records(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits duplicate rows across all columns and duplicate student register numbers.
    Maintains strict student privacy (only anonymized row references reported).
    """
    total_rows = len(df)
    
    # 1. Exact Duplicate Rows across all columns
    dup_rows_mask = df.duplicated(keep=False)
    dup_row_indices = list(df.index[dup_rows_mask])
    dup_rows_count = int(df.duplicated(keep="first").sum())

    # 2. Duplicate Register Numbers
    dup_reg_count = 0
    dup_reg_row_indices: List[int] = []

    if COL_REGISTER_NUMBER in df.columns and total_rows > 0:
        # Ignore empty/missing register numbers when evaluating duplicates
        valid_reg_mask = ~df[COL_REGISTER_NUMBER].apply(_is_missing_entry)
        reg_series = df.loc[valid_reg_mask, COL_REGISTER_NUMBER].astype(str).str.strip()
        
        dup_reg_mask = reg_series.duplicated(keep=False)
        dup_reg_row_indices = list(reg_series.index[dup_reg_mask])
        dup_reg_count = int(reg_series.duplicated(keep="first").sum())

    return {
        "exact_duplicate_rows_count": dup_rows_count,
        "exact_duplicate_affected_rows": len(dup_row_indices),
        "exact_duplicate_anonymized_rows": _to_anonymized_row_labels(dup_row_indices),
        "duplicate_register_numbers_count": dup_reg_count,
        "duplicate_register_affected_rows": len(dup_reg_row_indices),
        "duplicate_register_anonymized_rows": _to_anonymized_row_labels(dup_reg_row_indices),
    }


def check_data_types(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Reports detected pandas dtypes per column."""
    return {
        str(col): {
            "detected_dtype": str(df[col].dtype),
        }
        for col in df.columns
    }


def check_categorical_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts actual unique categorical values without making assumptions.
    Reports raw unique values for human review.
    """
    categorical_targets = [
        COL_GENDER,
        COL_DIPLOMA,
        COL_HISTORY_OF_ARREAR,
        COL_CURRENT_ARREAR,
    ]

    distributions: Dict[str, Any] = {}

    for col in categorical_targets:
        if col not in df.columns:
            distributions[col] = {"status": "COLUMN_MISSING"}
            continue

        # Extract normalized strings for inspection
        val_counts = df[col].dropna().astype(str).str.strip().value_counts().to_dict()
        unique_vals = list(val_counts.keys())
        missing_count = int(df[col].apply(_is_missing_entry).sum())

        distributions[col] = {
            "unique_values_count": len(unique_vals),
            "unique_values": unique_vals,
            "value_counts": val_counts,
            "missing_count": missing_count,
        }

    return distributions


def check_numeric_ranges_and_validity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits numeric columns for valid ranges, invalid non-numeric entries, and out-of-bounds metrics.
    Rules:
      - 10th %: 0.0 to 100.0
      - 12th % / Diploma%: 0.0 to 100.0
      - 12th Cut Off: numeric, identify anomalies
      - CGPA: 0.0 to 10.0 scale, flag negative or > 10.0
      - No. of Current Arrears: integer >= 0
      - LeetCode Solved: integer >= 0
    """
    results: Dict[str, Any] = {}

    # 1. 10th Percentage Validation
    if COL_10TH_PERCENT in df.columns:
        results[COL_10TH_PERCENT] = _validate_numeric_series(
            series=df[COL_10TH_PERCENT],
            min_bound=0.0,
            max_bound=100.0,
            integer_only=False,
            column_label="10th Percentage",
        )

    # 2. 12th / Diploma Percentage Validation
    if COL_12TH_OR_DIPLOMA_PERCENT in df.columns:
        results[COL_12TH_OR_DIPLOMA_PERCENT] = _validate_numeric_series(
            series=df[COL_12TH_OR_DIPLOMA_PERCENT],
            min_bound=0.0,
            max_bound=100.0,
            integer_only=False,
            column_label="12th / Diploma Percentage",
        )

    # 3. 12th Cut Off Validation
    if COL_12TH_CUTOFF in df.columns:
        # Standard cutoffs are typically out of 200 (engineering) or 100
        results[COL_12TH_CUTOFF] = _validate_numeric_series(
            series=df[COL_12TH_CUTOFF],
            min_bound=0.0,
            max_bound=200.0,
            integer_only=False,
            column_label="12th Cut Off",
        )

    # 4. CGPA (Till Semester 05) Validation
    if COL_CGPA in df.columns:
        results[COL_CGPA] = _validate_numeric_series(
            series=df[COL_CGPA],
            min_bound=0.0,
            max_bound=10.0,
            integer_only=False,
            column_label="CGPA (Semester 05)",
        )

    # 5. Number of Current Arrears Validation
    if COL_NO_OF_CURRENT_ARREARS in df.columns:
        results[COL_NO_OF_CURRENT_ARREARS] = _validate_numeric_series(
            series=df[COL_NO_OF_CURRENT_ARREARS],
            min_bound=0.0,
            max_bound=None,
            integer_only=True,
            column_label="No. of Current Arrears",
        )

    # 6. LeetCode Solved Count Validation
    if COL_LEETCODE_SOLVED in df.columns:
        results[COL_LEETCODE_SOLVED] = _validate_numeric_series(
            series=df[COL_LEETCODE_SOLVED],
            min_bound=0.0,
            max_bound=None,
            integer_only=True,
            column_label="LeetCode Solved",
        )

    return results


def _validate_numeric_series(
    series: pd.Series,
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
    integer_only: bool = False,
    column_label: str = "",
) -> Dict[str, Any]:
    """Helper to audit numeric boundaries and coercion failures."""
    total_count = len(series)
    missing_count = int(series.apply(_is_missing_entry).sum())

    # Attempt coercion to numeric
    numeric_series = pd.to_numeric(series, errors="coerce")
    
    # Values that were non-missing in original series but coerced to NaN are invalid
    non_missing_mask = ~series.apply(_is_missing_entry)
    coercion_failed_mask = non_missing_mask & numeric_series.isna()
    invalid_non_numeric_indices = list(series.index[coercion_failed_mask])

    valid_numerics = numeric_series.dropna()

    out_of_bounds_indices: List[int] = []
    non_integer_indices: List[int] = []

    if not valid_numerics.empty:
        if min_bound is not None:
            below_min = valid_numerics[valid_numerics < min_bound]
            out_of_bounds_indices.extend(list(below_min.index))
        if max_bound is not None:
            above_max = valid_numerics[valid_numerics > max_bound]
            out_of_bounds_indices.extend(list(above_max.index))

        if integer_only:
            # Check for non-integers or floating values
            non_ints = valid_numerics[valid_numerics.apply(lambda v: v % 1 != 0)]
            non_integer_indices.extend(list(non_ints.index))

    # Calculate standard distribution stats for valid numbers
    stats: Dict[str, Any] = {}
    if not valid_numerics.empty:
        stats = {
            "min": float(valid_numerics.min()),
            "max": float(valid_numerics.max()),
            "mean": round(float(valid_numerics.mean()), 2),
            "median": round(float(valid_numerics.median()), 2),
            "std": round(float(valid_numerics.std()), 2) if len(valid_numerics) > 1 else 0.0,
            "q25": round(float(valid_numerics.quantile(0.25)), 2),
            "q75": round(float(valid_numerics.quantile(0.75)), 2),
        }

    return {
        "column_label": column_label,
        "total_records": total_count,
        "missing_count": missing_count,
        "valid_numeric_count": len(valid_numerics),
        "invalid_non_numeric_count": len(invalid_non_numeric_indices),
        "invalid_non_numeric_anonymized_rows": _to_anonymized_row_labels(invalid_non_numeric_indices),
        "out_of_bounds_count": len(out_of_bounds_indices),
        "out_of_bounds_anonymized_rows": _to_anonymized_row_labels(out_of_bounds_indices),
        "non_integer_count": len(non_integer_indices),
        "non_integer_anonymized_rows": _to_anonymized_row_labels(non_integer_indices),
        "min_bound": min_bound,
        "max_bound": max_bound,
        "integer_only": integer_only,
        "statistics": stats,
    }


def check_profile_link_availability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits profile link presence and creates summary availability flags.
    Maintains strict privacy: NEVER prints or logs raw URLs/IDs.
    """
    total_students = len(df)
    results: Dict[str, Any] = {}
    flags_matrix: Dict[str, pd.Series] = {}

    for col in PROFILE_LINK_COLUMNS:
        if col not in df.columns:
            results[col] = {"status": "COLUMN_MISSING"}
            continue

        # Available if not null, not whitespace, and not placeholder
        is_avail = ~df[col].apply(_is_missing_entry)
        available_count = int(is_avail.sum())
        missing_count = total_students - available_count
        avail_pct = round((available_count / total_students * 100), 2) if total_students > 0 else 0.0

        flags_matrix[col] = is_avail
        results[col] = {
            "available_count": available_count,
            "missing_count": missing_count,
            "available_percentage": avail_pct,
        }

    # Combined readiness indicators
    if flags_matrix and total_students > 0:
        combined_df = pd.DataFrame(flags_matrix)
        total_profiles_per_student = combined_df.sum(axis=1)

        results["all_profiles_present_count"] = int((total_profiles_per_student == len(PROFILE_LINK_COLUMNS)).sum())
        results["zero_profiles_present_count"] = int((total_profiles_per_student == 0).sum())
        results["average_profiles_per_student"] = round(float(total_profiles_per_student.mean()), 2)

    return results


def check_logical_inconsistencies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates domain business logic and consistency across related columns.
    Identifies potential errors for human review without mutating data.
    """
    inconsistencies: List[Dict[str, Any]] = []

    has_curr_arr = COL_CURRENT_ARREAR in df.columns
    has_num_curr_arr = COL_NO_OF_CURRENT_ARREARS in df.columns
    has_hist_arr = COL_HISTORY_OF_ARREAR in df.columns

    if has_curr_arr and has_num_curr_arr:
        for idx, row in df.iterrows():
            curr_str = str(row[COL_CURRENT_ARREAR]).strip().lower() if not pd.isna(row[COL_CURRENT_ARREAR]) else ""
            num_raw = row[COL_NO_OF_CURRENT_ARREARS]

            # Try parsing count
            try:
                count_val = float(num_raw) if not _is_missing_entry(num_raw) else 0.0
            except (ValueError, TypeError):
                count_val = None

            # Case A: Current Arrear is 'No' but count is > 0
            if curr_str in {"no", "n"} and count_val is not None and count_val > 0:
                inconsistencies.append({
                    "rule_id": "LOGIC_01_NO_ARREAR_BUT_COUNT_POSITIVE",
                    "description": "CURRENT ARREAR is marked 'No', but No. OF CURRENT ARREARS is greater than 0.",
                    "anonymized_row": f"Row {idx + 2} (Index {idx})",
                    "row_index": int(idx),
                })

            # Case B: Current Arrear is 'Yes' but count is 0 or missing
            if curr_str in {"yes", "y"} and (count_val is None or count_val == 0):
                inconsistencies.append({
                    "rule_id": "LOGIC_02_YES_ARREAR_BUT_COUNT_ZERO_OR_MISSING",
                    "description": "CURRENT ARREAR is marked 'Yes', but No. OF CURRENT ARREARS is 0 or missing.",
                    "anonymized_row": f"Row {idx + 2} (Index {idx})",
                    "row_index": int(idx),
                })

    # Case C: History of Arrear is 'No' but Current Arrear is 'Yes'
    if has_hist_arr and has_curr_arr:
        for idx, row in df.iterrows():
            hist_str = str(row[COL_HISTORY_OF_ARREAR]).strip().lower() if not pd.isna(row[COL_HISTORY_OF_ARREAR]) else ""
            curr_str = str(row[COL_CURRENT_ARREAR]).strip().lower() if not pd.isna(row[COL_CURRENT_ARREAR]) else ""

            if hist_str in {"no", "n"} and curr_str in {"yes", "y"}:
                inconsistencies.append({
                    "rule_id": "LOGIC_03_NO_HISTORY_BUT_ACTIVE_CURRENT_ARREAR",
                    "description": "HISTORY OF ARREAR is marked 'No', but CURRENT ARREAR is 'Yes'.",
                    "anonymized_row": f"Row {idx + 2} (Index {idx})",
                    "row_index": int(idx),
                })

    return {
        "total_inconsistencies_detected": len(inconsistencies),
        "inconsistencies_list": inconsistencies,
    }


# =====================================================================
# MASTER AUDIT ORCHESTRATOR
# =====================================================================

def audit_student_dataset(
    df: pd.DataFrame,
    source_path: Optional[Union[str, Path]] = None,
    sheet_name: Optional[Union[str, int]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 2A Data Audit pipeline across all 19 inspection checkpoints.

    Args:
        df: The pandas DataFrame of student records.
        source_path: Optional file path to the source Excel workbook.
        sheet_name: Optional sheet name loaded.

    Returns:
        Dict[str, Any]: Structured audit findings dictionary.
    """
    audit_report: Dict[str, Any] = {
        "audit_metadata": {
            "source_path": str(source_path) if source_path else "In-Memory DataFrame",
            "sheet_name": str(sheet_name) if sheet_name is not None else "Default / First Sheet",
            "audit_phase": "Phase 2A — Data Audit Pipeline",
            "supervised_ml_status": "No placement outcome target present; supervised prediction deferred.",
        },
        "schema_audit": check_schema_and_columns(df),
        "missing_value_audit": check_missing_values(df),
        "duplicate_audit": check_duplicate_records(df),
        "datatype_audit": check_data_types(df),
        "categorical_distribution_audit": check_categorical_distributions(df),
        "numeric_range_audit": check_numeric_ranges_and_validity(df),
        "profile_link_audit": check_profile_link_availability(df),
        "logical_inconsistency_audit": check_logical_inconsistencies(df),
    }

    return audit_report


# =====================================================================
# AUDIT SUMMARY FORMATTER (HUMAN READABLE & PRIVACY PRESERVING)
# =====================================================================

def format_audit_summary(audit_results: Dict[str, Any], anonymize: bool = True) -> str:
    """
    Generates a clean, privacy-compliant Markdown summary of the audit results.
    Strictly excludes all student names, register numbers, and web links.
    """
    meta = audit_results.get("audit_metadata", {})
    schema = audit_results.get("schema_audit", {})
    missing = audit_results.get("missing_value_audit", {})
    duplicates = audit_results.get("duplicate_audit", {})
    categoricals = audit_results.get("categorical_distribution_audit", {})
    numerics = audit_results.get("numeric_range_audit", {})
    profiles = audit_results.get("profile_link_audit", {})
    logic = audit_results.get("logical_inconsistency_audit", {})

    lines: List[str] = [
        "=" * 78,
        "STUDENT DATASET AUDIT REPORT -- PHASE 2A",
        "=" * 78,
        f"Audit Phase        : {meta.get('audit_phase')}",
        f"Source Dataset     : {meta.get('source_path')}",
        f"Selected Sheet     : {meta.get('sheet_name')}",
        f"ML Target Status   : {meta.get('supervised_ml_status')}",
        "-" * 78,
        "1. SCHEMA & DIMENSIONS",
        f"  Total Rows       : {schema.get('row_count')}",
        f"  Total Columns    : {schema.get('column_count')} (Expected: {schema.get('expected_column_count')})",
        f"  Exact Schema Match: {'PASS' if schema.get('exact_match') else 'FAIL'}",
    ]

    if not schema.get("contains_all_expected"):
        lines.append(f"  Missing Columns  : {schema.get('missing_columns')}")
    if schema.get("unexpected_columns"):
        lines.append(f"  Extra Columns    : {schema.get('unexpected_columns')}")

    lines.extend([
        "-" * 78,
        "2. MISSING VALUE SUMMARY",
        f"  Total Missing Cells : {missing.get('total_missing_cells')} ({missing.get('overall_missing_percentage')}%)",
        f"  Complete Records    : {missing.get('complete_rows_count')} ({missing.get('complete_rows_percentage')}%)",
    ])

    missing_cols = missing.get("missing_by_column", {})
    for col_name, m_data in missing_cols.items():
        if m_data.get("has_missing"):
            lines.append(f"  - {col_name}: {m_data.get('missing_count')} missing ({m_data.get('missing_percentage')}%)")

    lines.extend([
        "-" * 78,
        "3. DUPLICATE RECORDS",
        f"  Exact Duplicate Rows        : {duplicates.get('exact_duplicate_rows_count')}",
        f"  Duplicate Register Numbers   : {duplicates.get('duplicate_register_numbers_count')}",
    ])
    if duplicates.get("duplicate_register_anonymized_rows"):
        lines.append(f"  Affected Rows (Anonymized)   : {', '.join(duplicates.get('duplicate_register_anonymized_rows')[:10])}")

    lines.extend([
        "-" * 78,
        "4. CATEGORICAL DISTRIBUTIONS (RAW SOURCE VALUES)",
    ])
    for cat_col, cat_data in categoricals.items():
        if isinstance(cat_data, dict) and "unique_values" in cat_data:
            lines.append(f"  * {cat_col}:")
            lines.append(f"    Observed Values : {cat_data.get('unique_values')}")
            lines.append(f"    Frequencies     : {cat_data.get('value_counts')}")

    lines.extend([
        "-" * 78,
        "5. NUMERIC RANGE & DATA QUALITY CHECKS",
    ])
    for num_col, num_data in numerics.items():
        label = num_data.get("column_label", num_col)
        stats = num_data.get("statistics", {})
        oob_count = num_data.get("out_of_bounds_count", 0)
        invalid_count = num_data.get("invalid_non_numeric_count", 0)
        non_int_count = num_data.get("non_integer_count", 0)

        lines.append(f"  * {label}:")
        if stats:
            lines.append(f"    Range: [{stats.get('min')} to {stats.get('max')}], Mean: {stats.get('mean')}, Median: {stats.get('median')}")
        if invalid_count > 0:
            lines.append(f"    WARNING: {invalid_count} non-numeric invalid entries found!")
        if oob_count > 0:
            lines.append(f"    WARNING: {oob_count} out-of-bounds values detected! (Bounds: {num_data.get('min_bound')} to {num_data.get('max_bound')})")
        if non_int_count > 0:
            lines.append(f"    WARNING: {non_int_count} non-integer values detected!")

    lines.extend([
        "-" * 78,
        "6. PROFILE LINK AVAILABILITY (DERIVED BINARY FLAGS)",
    ])
    for p_col in PROFILE_LINK_COLUMNS:
        p_data = profiles.get(p_col, {})
        if isinstance(p_data, dict) and "available_count" in p_data:
            lines.append(f"  - {p_col}: {p_data.get('available_count')} present ({p_data.get('available_percentage')}%), {p_data.get('missing_count')} missing")
    if "all_profiles_present_count" in profiles:
        lines.append(f"  Students with All 4 Profiles : {profiles.get('all_profiles_present_count')}")
        lines.append(f"  Students with 0 Profiles    : {profiles.get('zero_profiles_present_count')}")

    lines.extend([
        "-" * 78,
        "7. LOGICAL INCONSISTENCIES",
        f"  Total Inconsistencies Found : {logic.get('total_inconsistencies_detected')}",
    ])
    for item in logic.get("inconsistencies_list", [])[:10]:
        lines.append(f"  - [{item.get('rule_id')}] {item.get('anonymized_row')}: {item.get('description')}")
    if logic.get("total_inconsistencies_detected", 0) > 10:
        lines.append(f"  ... and {logic.get('total_inconsistencies_detected') - 10} more.")

    lines.extend([
        "=" * 78,
        "PRIVACY COMPLIANCE: No student names, register numbers, or URLs are logged.",
        "=" * 78,
    ])

    return "\n".join(lines)


# =====================================================================
# COMMAND LINE INTERFACE
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit student Excel dataset for Phase 2A of Career & Placement Prediction System."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to the student Excel database file (.xlsx or .xls).",
    )
    parser.add_argument(
        "--sheet",
        "-s",
        type=str,
        default="0",
        help="Workbook sheet name or zero-based index to load (default: 0).",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        default=None,
        help="Optional path to export markdown audit report.",
    )

    args = parser.parse_args()

    if not args.file:
        print("\n" + "=" * 78)
        print("DATA AUDIT PIPELINE -- USAGE INSTRUCTIONS")
        print("=" * 78)
        print("The student Excel dataset is not currently connected to this project.")
        print("\nWhen you are ready to audit your dataset:")
        print("  1. Copy your Excel file to: data/raw/student_data.xlsx")
        print("  2. Run the audit command:")
        print("     python -m src.data.audit_dataset --file data/raw/student_data.xlsx")
        print("\nTo see all CLI options, run:")
        print("     python -m src.data.audit_dataset --help")
        print("=" * 78 + "\n")
        sys.exit(0)

    from src.data.load_excel import load_student_excel

    target_path = Path(args.file)
    print(f"\n[INFO] Loading Excel file from: {target_path}...")

    sheet_arg: Union[str, int] = int(args.sheet) if args.sheet.isdigit() else args.sheet

    try:
        df = load_student_excel(target_path, sheet_name=sheet_arg)
    except Exception as e:
        print(f"\n[ERROR] Failed to load Excel file:\n{e}\n")
        sys.exit(1)

    print(f"[INFO] Auditing dataset with {len(df)} rows and {len(df.columns)} columns...")
    audit_report = audit_student_dataset(df, source_path=target_path, sheet_name=sheet_arg)
    summary_text = format_audit_summary(audit_report)

    print("\n" + summary_text + "\n")

    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(summary_text, encoding="utf-8")
        print(f"[INFO] Audit summary saved to: {export_path}")


if __name__ == "__main__":
    main()
