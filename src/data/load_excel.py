"""
Excel Data Ingestion Utility
AI-Powered Student Career & Placement Prediction System

Provides safe loading, path validation, and sheet discovery for student Excel databases.
"""

from pathlib import Path
from typing import List, Union, Optional
import pandas as pd


VALID_EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}


def validate_excel_path(file_path: Union[str, Path]) -> Path:
    """
    Validates that the target path exists and is an Excel workbook.

    Args:
        file_path: Path to the target Excel file.

    Returns:
        Path: Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a recognized Excel extension.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Excel dataset not found at: '{path}'.\n"
            f"Please ensure the real student database is placed in 'data/raw/' or "
            f"provide the correct absolute path to the Excel file."
        )

    if not path.is_file():
        raise ValueError(f"Path '{path}' is a directory, not an Excel file.")

    if path.suffix.lower() not in VALID_EXCEL_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. "
            f"Supported extensions are: {', '.join(sorted(VALID_EXCEL_EXTENSIONS))}"
        )

    return path


def get_excel_sheet_names(file_path: Union[str, Path]) -> List[str]:
    """
    Retrieves all sheet names from an Excel workbook without reading full contents.

    Args:
        file_path: Path to the Excel workbook.

    Returns:
        List[str]: List of sheet names present in the workbook.
    """
    path = validate_excel_path(file_path)
    excel_file = pd.ExcelFile(path)
    return excel_file.sheet_names


def load_student_excel(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    preserve_raw_strings: bool = False,
) -> pd.DataFrame:
    """
    Loads student data from a specified sheet of an Excel workbook.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Sheet name or index (defaults to the first sheet, 0).
        preserve_raw_strings: If True, reads all columns as raw strings (dtype=str)
                              useful for low-level parsing diagnostics.

    Returns:
        pd.DataFrame: Loaded student dataframe.
    """
    path = validate_excel_path(file_path)

    dtype = str if preserve_raw_strings else None
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=dtype)

    return df
