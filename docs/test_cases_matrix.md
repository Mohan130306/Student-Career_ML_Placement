# Automated Test Cases & Quality Assurance Matrix

## AI-Powered Student Career & Placement Prediction System
- **Test Suite Status**: 51 Passed / 0 Failed (100% Pass Rate)
- **Framework**: `pytest`
- **Execution Speed**: ~17 seconds for full end-to-end regression test suite

---

## 1. Test Suite Summary Inventory

| Test Module | Subsystem Tested | Test Count | Status | Execution Scope |
| :--- | :--- | :---: | :---: | :--- |
| `tests/test_audit.py` | Data Audit & Excel Ingestion | 18 | ✅ PASS | Schema compliance, missing values, duplicates, range checking, logical rules. |
| `tests/test_generator.py` | Benchmark Generator | 6 | ✅ PASS | Distribution bounds, deterministic seeds, placement-role invariant consistency. |
| `tests/test_preprocessing.py`| Feature Engineering & Scaler | 4 | ✅ PASS | Composite features, Scikit-learn API, imputer robustness, leakage prevention. |
| `tests/test_eda.py` | Statistical & Diagnostic Profiling | 5 | ✅ PASS | Correlation bounds, cohort difference metrics, demographic parity ratios. |
| `tests/test_models.py` | ML Training & Inference Engine | 5 | ✅ PASS | Candidate model initialization, 5-fold CV evaluation, prediction probability tiers. |
| `tests/test_explainability.py`| Explainable AI (SHAP) | 3 | ✅ PASS | Global SHAP feature importance, individual attribution decomposition, waterfall export. |
| `tests/test_recommendations.py`| Recommendation & Skill-Gap | 5 | ✅ PASS | Competency gap analysis, role compatibility matching, company tier screening, roadmaps. |
| `tests/test_db.py` | SQLAlchemy ORM & Repository | 1 | ✅ PASS | In-memory SQLite session, record persistence, batch summary calculations. |
| `tests/test_api.py` | FastAPI REST Endpoints | 4 | ✅ PASS | `/health`, `/predict`, `/explain`, and unified `/assess` with DB persistence. |
| **TOTAL** | **Entire Project Ecosystem** | **51** | **✅ 100%** | **Comprehensive Full-Pipeline Quality Verification** |

---

## 2. Detailed Test Cases Matrix

### Module 1: `tests/test_audit.py` (18 Test Cases)
* **TC-AUD-01 to TC-AUD-04**: Validates exact college Excel schema match, casing insensitivity, and extra column detection.
* **TC-AUD-05 to TC-AUD-08**: Tests missing value detection, spreadsheet placeholder strings (`"na"`, `"-"`, `""`), and complete record ratios.
* **TC-AUD-09 to TC-AUD-11**: Tests primary key / register number duplicate detection and exact row duplicate detection.
* **TC-AUD-12 to TC-AUD-14**: Validates numeric boundaries (CGPA in $[0, 10]$, percentage in $[0, 100]$, arrear count $\ge 0$).
* **TC-AUD-15 to TC-AUD-18**: Validates logical invariants (e.g. Current Arrear = "YES" requires count $> 0$) and privacy-compliant URL flag derivation.

### Module 2: `tests/test_generator.py` (6 Test Cases)
* **TC-GEN-01 (Dimensions & Columns)**: Verifies exact 24 columns and target sample count ($N=1,000$).
* **TC-GEN-02 (Null Integrity)**: Asserts zero unhandled nulls or NaNs in generated records.
* **TC-GEN-03 (Domain Bounds)**: Asserts all continuous features stay strictly in realistic ranges.
* **TC-GEN-04 (Logical Invariants)**: Asserts unplaced students are labeled `Not Placed`; students without history of arrears have 0 current arrears.
* **TC-GEN-05 (Statistical Grounding)**: Asserts placed cohorts have higher mean coding score, DSA, and CGPA than unplaced cohorts.
* **TC-GEN-06 (Deterministic Seed)**: Asserts identical seeds generate bit-identical DataFrames.

### Module 3: `tests/test_preprocessing.py` (4 Test Cases)
* **TC-PRE-01 (Feature Engineering)**: Asserts all 5 composite features (`fe_technical_aggregate`, `fe_coding_dexterity`, `fe_profile_richness`, `fe_academic_consistency`, `fe_arrear_burden`) are calculated accurately.
* **TC-PRE-02 (Transformer API)**: Asserts `DomainFeatureTransformer` strictly follows Scikit-learn's `TransformerMixin` contract.
* **TC-PRE-03 (Imputation & Scaling)**: Injects deliberate NaNs into test DataFrame; asserts `ColumnTransformer` imputes without leakage and scales to 29 columns.
* **TC-PRE-04 (End-to-End Pipeline & Artifacts)**: Executes complete stratified split, verifies CSV exports, and tests `.joblib` serialization/deserialization.

### Module 4: `tests/test_eda.py` (5 Test Cases)
* **TC-EDA-01 (Correlation Boundaries)**: Asserts all Pearson/point-biserial correlations lie in $[-1.0, 1.0]$ and identifies top positive drivers.
* **TC-EDA-02 (Cohort Distributions)**: Asserts placed mean exceeds unplaced mean on key competencies.
* **TC-EDA-03 (Role Benchmarks)**: Asserts benchmark profiles are computed for all 5 roles.
* **TC-EDA-04 (Department Aggregation)**: Asserts department placement rates lie in $[0\%, 100\%]$.
* **TC-EDA-05 (Demographic Parity)**: Asserts Adverse Impact Ratio calculation satisfies non-zero bounds.

### Module 5: `tests/test_models.py` (5 Test Cases)
* **TC-MOD-01 (Architecture Verification)**: Asserts all 4 models (Logistic Regression, Random Forest, GBDT, XGBoost) initialize.
* **TC-MOD-02 (CV & Metric Leaderboard)**: Asserts 5-fold cross-validation and test set evaluations produce non-null metrics in $[0, 1]$.
* **TC-MOD-03 (High-Readiness Inference)**: Evaluates student with strong profile; asserts prediction = 1, probability $> 0.65$, tier = `High Placement Readiness`.
* **TC-MOD-04 (At-Risk Inference)**: Evaluates student with low profile and standing arrears; asserts prediction = 0, probability $< 0.40$, tier = `At-Risk`.
* **TC-MOD-05 (Batch Scoring)**: Asserts batch DataFrame scoring appends `predicted_placement`, `placement_probability`, and `readiness_tier`.

### Module 6: `tests/test_explainability.py` (3 Test Cases)
* **TC-XAI-01 (Global SHAP Importance)**: Asserts global feature importance vector has length 29 and non-negative mean absolute SHAP values.
* **TC-XAI-02 (Local Attribution Decomposition)**: Asserts individual student explanation partitions positive amplifiers and negative risk penalties.
* **TC-XAI-03 (Waterfall Plot Generation)**: Asserts waterfall diagnostic figure renders and exports to disk.

### Module 7: `tests/test_recommendations.py` (5 Test Cases)
* **TC-REC-01 (Skill-Gap Differentiation)**: Asserts strong profile exhibits strengths while struggling profile highlights critical deficits.
* **TC-REC-02 (Role Alignment Ranking)**: Asserts all 5 roles receive ranked compatibility percentages.
* **TC-REC-03 (Corporate Screening Rules)**: Asserts product tier disqualification under standing arrears or low CGPA.
* **TC-REC-04 (Roadmap Generation)**: Asserts 4-week structured plan with arrear remediation priority.
* **TC-REC-05 (Unified Career Pipeline)**: Asserts end-to-end integration of all recommendation services.

### Module 8 & 9: `tests/test_db.py` & `tests/test_api.py` (5 Test Cases)
* **TC-DB-01 (ORM Persistence)**: Asserts student assessment saves to SQLite/PostgreSQL table and calculates batch summary.
* **TC-API-01 (Health Check)**: Asserts `GET /health` returns status 200 OK.
* **TC-API-02 (Predict Endpoint)**: Asserts `POST /api/v1/predict` returns valid typed JSON.
* **TC-API-03 (Explain Endpoint)**: Asserts `POST /api/v1/explain` returns SHAP positive and negative factors.
* **TC-API-04 (Unified Assessment Endpoint)**: Asserts `POST /api/v1/assess` executes unified prediction, SHAP, career roadmap, and optional DB write.

---

## 3. How to Execute the QA Suite

```bash
# Run all 51 tests with verbose progress
pytest -v

# Run with execution duration profiling
pytest --durations=5
```
