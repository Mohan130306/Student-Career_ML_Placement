"""
Data Preprocessing & Splitting Execution Script
AI-Powered Student Career & Placement Prediction System

Executes leak-free preprocessing:
1. Loads raw benchmark dataset
2. Adds domain-engineered features
3. Splits data into Stratified Train (80%) and Test (20%) sets
4. Fits ColumnTransformer strictly on X_train (No Data Leakage)
5. Transforms X_train and X_test
6. Serializes preprocessor and saves processed datasets
"""

from pathlib import Path
from typing import Dict, Tuple
import argparse
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.generate_dataset import (
    COL_STUDENT_ID,
    COL_GENDER,
    COL_PLACEMENT_STATUS,
    COL_PLACED_ROLE,
)
from src.features.build_features import add_domain_features
from src.features.pipeline import (
    create_preprocessing_pipeline,
    get_feature_names_from_preprocessor,
)


def run_preprocessing_pipeline(
    raw_data_path: Path,
    output_dir: Path,
    model_dir: Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Executes the full leak-free data preprocessing and serialization flow.

    Args:
        raw_data_path: Path to raw input dataset (CSV).
        output_dir: Directory where processed CSVs will be saved.
        model_dir: Directory where the fitted preprocessor pipeline will be saved.
        test_size: Proportion of test split (default: 0.20).
        random_state: Random state for deterministic splitting.

    Returns:
        Dict: Metadata and shapes of processed splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Raw Dataset
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Source raw data file not found at: {raw_data_path}")

    raw_df = pd.read_csv(raw_data_path)

    # 2. Add Domain Engineered Features
    featured_df = add_domain_features(raw_df)

    # 3. Segregate Features, Targets, and Audit Metadata
    # Held out from primary model features:
    # - student_id: Identifier (prevent identity memorize)
    # - gender: Demographic attribute held out for fairness evaluation
    # - placement_status: Target variable
    # - placed_role: Downstream role target (avoid target leakage)
    drop_from_X = [
        COL_STUDENT_ID,
        COL_GENDER,
        COL_PLACEMENT_STATUS,
        COL_PLACED_ROLE,
    ]

    X = featured_df.drop(columns=drop_from_X)
    y = featured_df[COL_PLACEMENT_STATUS]
    metadata = featured_df[[COL_STUDENT_ID, COL_GENDER, COL_PLACED_ROLE]]

    # 4. Stratified Train/Test Split BEFORE Any Preprocessor Fitting
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 5. Fit Preprocessing Pipeline ONLY on X_train
    preprocessor = create_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 6. Retrieve Processed Feature Names
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # Reconstruct Processed DataFrames with Clean Column Names
    X_train_proc = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_proc = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    # 7. Save Processed Artifacts
    X_train_proc.to_csv(output_dir / "X_train.csv", index=False)
    X_test_proc.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(name=COL_PLACEMENT_STATUS).to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame(name=COL_PLACEMENT_STATUS).to_csv(output_dir / "y_test.csv", index=False)
    meta_train.to_csv(output_dir / "meta_train.csv", index=False)
    meta_test.to_csv(output_dir / "meta_test.csv", index=False)

    # 8. Save Preprocessor Pipeline
    preprocessor_path = model_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_in": X.shape[1],
        "n_features_out": len(feature_names),
        "preprocessor_path": str(preprocessor_path),
        "feature_names": feature_names,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run leak-free student feature preprocessing pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/student_placement_benchmark.csv",
        help="Path to input raw CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/processed",
        help="Output directory for processed CSVs.",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        default="models",
        help="Output directory for serialized preprocessor.",
    )
    args = parser.parse_args()

    results = run_preprocessing_pipeline(
        raw_data_path=Path(args.input),
        output_dir=Path(args.outdir),
        model_dir=Path(args.modeldir),
    )

    print("=" * 60)
    print("DATA PREPROCESSING & FEATURE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Train Records (Stratified 80%) : {results['n_train']}")
    print(f"Test Records  (Stratified 20%) : {results['n_test']}")
    print(f"Input Raw/Engineered Features  : {results['n_features_in']}")
    print(f"Transformed Model Features     : {results['n_features_out']}")
    print(f"Preprocessor Pipeline Saved    : {results['preprocessor_path']}")
    print(f"\nFinal Feature Names ({results['n_features_out']}):")
    for idx, name in enumerate(results['feature_names'], 1):
        print(f"  {idx:02d}. {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
