# Data Directory

This directory manages all datasets used across the lifecycle of the **AI-Powered Student Career & Placement Prediction System**.

## Directory Structure

```
data/
├── raw/          # Immutable original datasets
├── processed/    # Cleaned, transformed, and encoded datasets
└── README.md     # Documentation and data management guidelines
```

## Subdirectory Details

### 1. `raw/`
- **Purpose**: Stores initial raw data extracts (e.g., CSV, JSON, Excel) collected from academic institutions, student records, or assessment portals.
- **Rule of Immutability**: Files in this folder are treated as read-only. Never modify, clean, or overwrite raw files directly.
- **Tracking**: Large data files are ignored by git via `.gitignore` to keep repository size lean and prevent accidental commits of proprietary information.

### 2. `processed/`
- **Purpose**: Stores intermediate and finalized datasets generated through reproducible preprocessing and feature engineering pipelines.
- **Contents**:
  - Imputed and validated records
  - Encoded categorical variables
  - Scaled numerical features
  - Train/test splits generated with strict seed control
- **Reproducibility**: Any file in this directory must be reproducible on-demand by executing scripts in `src/` or documented pipeline notebooks.

## Data Governance & Integrity Rules

1. **No Data Leakage**: Preprocessing transformations (scaling, imputation, encoding) must be fitted strictly on training data splits and applied subsequently to validation/test sets.
2. **Privacy & Security**: Student personally identifiable information (PII) such as full names, contact details, or institutional IDs must be anonymized or excluded before saving to processed data stores.
3. **Traceability**: All processing transformations must be documented with version stamps or pipeline logs.
4. **Current Status**: In Phase 1, no datasets have been created or modified. Data ingestion and exploratory analysis will take place in Phase 2.
