# Data Preprocessing & Feature Engineering Pipeline

## AI-Powered Student Career & Placement Prediction System
- **Phase**: Phase 2 — Leak-Free Preprocessing Pipeline
- **Modules**: `src/features/build_features.py`, `src/features/pipeline.py`, `src/features/preprocess.py`
- **Output Artifacts**:
  - `data/processed/X_train.csv` (800 rows, 29 features)
  - `data/processed/X_test.csv` (200 rows, 29 features)
  - `data/processed/y_train.csv` / `y_test.csv` (Stratified target binary)
  - `data/processed/meta_train.csv` / `meta_test.csv` (Student ID, Gender, Role)
  - `models/preprocessor.joblib` (Serialized ColumnTransformer pipeline)

---

## 1. Pipeline Architecture

```
                  Raw Student Data (1,000 records)
                               │
                               ▼
                 Domain Feature Engineering
               (5 Composite Strategic Features)
                               │
                               ▼
                 Feature & Metadata Separation
              - Drop student_id (Identity memorize)
              - Hold out gender (Demographic parity audit)
              - Separate placement_status (Target)
              - Separate placed_role (Downstream role evaluation)
                               │
                               ▼
                 Stratified Train / Test Split
                    (80% Train : 20% Test)
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
           X_train (800)                 X_test (200)
                │                             │
                ▼                             │
       fit_transform()                        │
                │                             │
                ▼                             ▼
      Fitted ColumnTransformer ──► transform() (NO fit)
                │                             │
                ▼                             ▼
       X_train_proc (29 cols)        X_test_proc (29 cols)
```

---

## 2. Engineered Domain Features

| Feature Name | Formula / Derivation | Domain Rationale |
| :--- | :--- | :--- |
| `fe_technical_aggregate` | $0.30 \times \text{DSA} + 0.25 \times \text{Coding} + 0.15 \times \text{SQL} + 0.15 \times \frac{\text{Java} + \text{Python}}{2} + 0.15 \times \text{Web}$ | Balanced technical composite measuring core engineering breadth and algorithmic depth. |
| `fe_coding_dexterity` | $0.60 \times \text{Coding Score} + 0.40 \times \min\left(100, \frac{\text{LeetCode Solved}}{3}\right)$ | Bridges theoretical platform coding score with sustained competitive problem-solving volume. |
| `fe_profile_richness` | Projects ($\times 12$) + Internships ($\times 20$) + Certifications ($\times 6$) + GitHub ($\times 8$) + LinkedIn ($\times 8$) | Quantifies candidate proactivity and tangible portfolio evidence beyond test marks. |
| `fe_academic_consistency` | $\frac{\text{12th Percentage}}{\max(\text{10th Percentage}, 1)}$ | Captures academic stability across higher secondary education. |
| `fe_arrear_burden` | $\text{Current Arrears} \times 3.0 + \text{History of Arrears} \times 0.75$ | Models real-world recruiter screening severity (standing arrears incur strict disqualification). |

---

## 3. Transformations & ColumnTransformer Specification

The pipeline uses Scikit-learn's `ColumnTransformer`:

* **Numeric Sub-pipeline (24 features)**:
  - `SimpleImputer(strategy='median')`: Defensive protection against missing values in user submissions.
  - `StandardScaler()`: Zero-mean, unit-variance normalization essential for Logistic Regression, distance metrics, and neural architectures.
* **Categorical Sub-pipeline (1 feature -> 5 columns)**:
  - `department`: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` producing:
    - `department_AIDS`
    - `department_CSBS`
    - `department_CSE`
    - `department_ECE`
    - `department_IT`

---

## 4. Leakage Prevention Protocol

To guarantee mathematical integrity:
1. **Strict Splitting Sequence**: The train/test split is performed **before** fitting scalers, encoders, or imputers.
2. **Train-Only Parameter Estimation**: Means and standard deviations for scaling are calculated strictly on `X_train`. `X_test` is transformed using the fitted training statistics without re-estimation.
3. **Target Quarantine**: `placement_status` and `placed_role` are quarantined from `X` prior to transformation.

---

## 5. Execution Command

To re-run preprocessing:
```bash
python -m src.features.preprocess --input data/raw/student_placement_benchmark.csv --outdir data/processed --modeldir models
```
