"""
Data Ingestion and Auditing Module
AI-Powered Student Career & Placement Prediction System
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.load_excel import get_excel_sheet_names, load_student_excel
    from src.data.audit_dataset import (
        EXPECTED_COLUMNS,
        IDENTIFIER_COLUMNS,
        PROTECTED_ANALYSIS_COLUMNS,
        CANDIDATE_ML_FEATURES,
        PROFILE_LINK_COLUMNS,
        audit_student_dataset,
        format_audit_summary,
    )


def __getattr__(name: str):
    if name in {"get_excel_sheet_names", "load_student_excel"}:
        from src.data import load_excel
        return getattr(load_excel, name)
    elif name in {
        "EXPECTED_COLUMNS",
        "IDENTIFIER_COLUMNS",
        "PROTECTED_ANALYSIS_COLUMNS",
        "CANDIDATE_ML_FEATURES",
        "PROFILE_LINK_COLUMNS",
        "audit_student_dataset",
        "format_audit_summary",
    }:
        from src.data import audit_dataset
        return getattr(audit_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_excel_sheet_names",
    "load_student_excel",
    "EXPECTED_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "PROTECTED_ANALYSIS_COLUMNS",
    "CANDIDATE_ML_FEATURES",
    "PROFILE_LINK_COLUMNS",
    "audit_student_dataset",
    "format_audit_summary",
]
