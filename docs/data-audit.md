# Phase 2A — Data Audit Pipeline Specification

## Document Information
- **Project**: AI-Powered Student Career & Placement Prediction System
- **Current Phase**: Phase 2A — Data Audit Pipeline
- **Dataset Status**: **External / Not Connected**. No raw student database is currently committed or generated within this repository.

> **CRITICAL NOTE ON MACHINE LEARNING SCOPE**:  
> **The current project does not have a placement outcome target. Therefore supervised placement prediction is not being implemented at this stage.**  
> No models (clustering, classification, PCA, or regression) will be trained, and no synthetic target labels will be fabricated.

---

## 1. Purpose of the Data Audit

The purpose of Phase 2A is to build an objective, automated, and reusable data auditing framework that will evaluate the real institutional Excel database once it is connected to the project. 

The audit pipeline inspects data quality, flags potential schema mismatches, detects missing and duplicate entries, benchmarks numeric ranges, audits categorical distributions, assesses profile link availability, and surfaces logical anomalies—**all while strictly enforcing student data privacy standards**.

---

## 2. Expected Source Columns (17 Columns)

The source Excel spreadsheet is expected to contain exactly the following 17 columns:

1. `NAME OF THE STUDENT (INITIAL AT THE LAST)`
2. `REGISTER NUMBER`
3. `GENDER`
4. `10TH % (ENTER VALUE ALONE)`
5. `10TH YEAR OF PASSING`
6. `DIPLOMA STUDENT`
7. `12TH % (ENTER VALUE ALONE) / DIPLOMA%`
8. `12TH CUT OFF`
9. `CGPA (TILL SEMESTER 05)`
10. `RESUME LINK`
11. `HISTORY OF ARREAR (YES/NO)`
12. `CURRENT ARREAR (YES/NO)`
13. `No. OF CURRENT ARREARS`
14. `LINKED - IN ID`
15. `HACKER RANK ID`
16. `GIT HUB ID`
17. `NO. OF LEET CODE PROGRAMS SOLVED TILL NOW`

---

## 3. Column Classification Framework

To ensure ethical, leak-free, and privacy-preserving machine learning engineering, every column is categorized into a strict operational class:

### A. Identifier Only (Never Used in ML)
- `NAME OF THE STUDENT (INITIAL AT THE LAST)`
- `REGISTER NUMBER`
- **Policy**: Used strictly for source-record tracing and institutional auditing. These fields must **never** be used as machine learning features, inputs to dimensionality reduction, or clustering.

### B. Protected Analysis Only (Fairness & Bias Diagnostics)
- `GENDER`
- **Policy**: Excluded from predictive feature sets during baseline modeling. Reserved exclusively for downstream fairness audits, demographic parity tests, and institutional diversity reporting.

### C. Candidate ML Features
- `10TH % (ENTER VALUE ALONE)` (Numeric)
- `10TH YEAR OF PASSING` (Numeric / Temporal)
- `DIPLOMA STUDENT` (Categorical)
- `12TH % (ENTER VALUE ALONE) / DIPLOMA%` (Numeric)
- `12TH CUT OFF` (Numeric)
- `CGPA (TILL SEMESTER 05)` (Numeric)
- `HISTORY OF ARREAR (YES/NO)` (Categorical)
- `CURRENT ARREAR (YES/NO)` (Categorical)
- `No. OF CURRENT ARREARS` (Numeric / Count)
- `NO. OF LEET CODE PROGRAMS SOLVED TILL NOW` (Numeric / Count)

### D. Derived Profile Availability Flags (Binary: 0 or 1)
- `RESUME LINK` → `resume_available`
- `LINKED - IN ID` → `linkedin_available`
- `HACKER RANK ID` → `hackerrank_available`
- `GIT HUB ID` → `github_available`
- **Policy**: The actual URLs, usernames, or hyperlinks must **never** be fed directly into machine learning models. Instead, the pipeline derives binary readiness indicators:
  - If a valid link/ID is present: `1`
  - If missing, blank, or placeholder (`N/A`, `-`): `0`

---

## 4. Validation Rules

| Column | Expected Type | Valid Range / Rules | Action on Violation |
| :--- | :--- | :--- | :--- |
| `10TH % (ENTER VALUE ALONE)` | Numeric (Float) | `0.0` to `100.0` | Flag out-of-bounds or non-numeric entries; report anonymized row references. |
| `12TH % / DIPLOMA%` | Numeric (Float) | `0.0` to `100.0` | Flag out-of-bounds or non-numeric entries; report anonymized row references. |
| `12TH CUT OFF` | Numeric (Float) | `0.0` to `200.0` (or `100.0`) | Identify unusual or non-numeric values. Do **not** automatically delete unusual values. |
| `CGPA (TILL SEMESTER 05)` | Numeric (Float) | `0.0` to `10.0` | Flag negative or `> 10.0` values. Do **not** silently modify or clip values. |
| `No. OF CURRENT ARREARS` | Integer | Integer `≥ 0` | Flag negative values, non-integers, or text strings. |
| `NO. OF LEET CODE PROGRAMS SOLVED` | Integer | Integer `≥ 0` | Flag negative values, non-integers, or text strings. |
| `DIPLOMA STUDENT` | Categorical | Unique source values | Extract and report actual values; do not assume strict `Yes`/`No` format. |
| `HISTORY OF ARREAR (YES/NO)` | Categorical | Unique source values | Extract and report actual values for human review. |
| `CURRENT ARREAR (YES/NO)` | Categorical | Unique source values | Extract and report actual values for human review. |

---

## 5. Privacy Rules

Student records contain sensitive educational and personally identifiable information (PII). The audit system adheres to the following privacy protocols:

1. **No PII Logging**: Full student names, register numbers, and profile URLs must never appear in console logs, audit reports, or test fixtures.
2. **Anonymized References**: When flagging anomalies (e.g. invalid CGPA or duplicate rows), the audit references anonymized record indices (e.g., `Row 14 (Index 12)`).
3. **Repository Exclusion**: Excel workbooks (`*.xlsx`, `*.xls`, `*.xlsm`, `*.xlsb`) and raw CSV spreadsheets are blocked from Git tracking via `.gitignore`.
4. **No Real Data in Tests**: Unit tests use small, synthetic fixtures explicitly isolated from project data.

---

## 6. Missing-Data Policy

- **Placeholder Recognition**: The audit detects both native `NaN` / `None` values and common manual text placeholders (such as `"N/A"`, `"NA"`, `"NIL"`, `"-"`, `"--"`, `"None"`).
- **No Premature Imputation**: In Phase 2A, missing values are purely quantified and reported. No imputation (mean, median, mode, or KNN) is performed until data cleaning pipelines are formally designed in Phase 3.

---

## 7. Duplicate-Data Policy

The audit inspects two independent dimensions of duplication:
1. **Exact Row Duplication**: Identical values across all 17 columns.
2. **Register Number Duplication**: Multiple rows sharing the same register number with conflicting academic/profile entries.

All occurrences are aggregated, counted, and cited by anonymized row indices for institutional review.

---

## 8. Logical Consistency Rules

The audit identifies discrepancies across interdependent columns without mutating the underlying data:

* **Rule LOGIC_01**: `CURRENT ARREAR (YES/NO)` is `"No"` AND `No. OF CURRENT ARREARS` `> 0`.
* **Rule LOGIC_02**: `CURRENT ARREAR (YES/NO)` is `"Yes"` AND `No. OF CURRENT ARREARS` is `0` or missing.
* **Rule LOGIC_03**: `HISTORY OF ARREAR (YES/NO)` is `"No"` AND `CURRENT ARREAR (YES/NO)` is `"Yes"` (a student with active arrears must historically have had arrears).

---

## 9. What the Audit Does NOT Do

To maintain scientific integrity and prevent premature assumptions:
1. **No Synthetic Data Generation**: Does not generate fake students or dummy placement datasets.
2. **No Data Mutation**: Does not silently delete, overwrite, or auto-correct anomalies.
3. **No Machine Learning**: Does not train clustering (K-Means, DBSCAN), dimensionality reduction (PCA), or classification models.
4. **No Target Creation**: Does not generate arbitrary placement outcomes or labels.

---

## 10. Next Phase

- **Phase 2B — Dataset Connection & Live Audit**: Connect the genuine institutional Excel file to `data/raw/student_data.xlsx` and execute the audit report.
- **Phase 2C — Exploratory Data Analysis (EDA)**: Conduct statistical profiling, correlation discovery, and visual diagnostics on verified student records.
- **Phase 3 — Preprocessing & Feature Engineering Pipeline**: Implement leak-free transformers for imputation, encoding, scaling, and deriving profile availability flags.
