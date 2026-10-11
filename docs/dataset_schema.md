# Dataset Schema & Data Dictionary

## AI-Powered Student Career & Placement Prediction System
- **Phase**: Phase 1 — Benchmark Dataset Specification
- **Dataset File**: `data/raw/student_placement_benchmark.csv` and `.xlsx`
- **Total Records**: 1,000
- **Total Features**: 24

---

## 1. Column Classifications & Data Governance

To prevent **target leakage**, **data snooping**, and **unethical algorithmic bias**, all 24 columns are strictly segregated into functional roles:

| Column Name | Data Type | Domain Range | Governance Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `student_id` | `str` | `STU_2026_0001`+ | **Identifier (Drop)** | Unique record key; excluded from ML models. |
| `department` | `category` | `CSE`, `IT`, `ECE`, `AIDS`, `CSBS` | **ML Feature** | Degree specialization (one-hot encoded). |
| `gender` | `category` | `Male`, `Female` | **Demographic Audit** | Held out from primary model to prevent gender bias; retained for demographic parity audits. |
| `tenth_percentage` | `float` | $50.0 - 100.0$ | **ML Feature** | Secondary school academic performance. |
| `twelfth_percentage`| `float` | $50.0 - 100.0$ | **ML Feature** | Higher secondary academic percentage. |
| `cgpa` | `float` | $5.0 - 10.0$ | **ML Feature** | Cumulative Grade Point Average (through Sem 5/6). |
| `history_of_arrears`| `int` (0/1) | $0 \text{ or } 1$ | **ML Feature** | Binary flag indicating if student ever had an arrear. |
| `current_arrears` | `int` | $\ge 0$ | **ML Feature** | Active standing arrears during placement season. |
| `dsa_score` | `float` | $0.0 - 100.0$ | **ML Feature** | Data Structures & Algorithms proficiency rating. |
| `java_score` | `float` | $0.0 - 100.0$ | **ML Feature** | Enterprise/Core Java programming score. |
| `python_score` | `float` | $0.0 - 100.0$ | **ML Feature** | Python scripting & data analysis score. |
| `sql_score` | `float` | $0.0 - 100.0$ | **ML Feature** | Relational database & query proficiency. |
| `web_dev_score` | `float` | $0.0 - 100.0$ | **ML Feature** | Frontend & Backend web development rating. |
| `aptitude_score` | `float` | $0.0 - 100.0$ | **ML Feature** | Quantitative, analytical & logical reasoning test. |
| `coding_score` | `float` | $0.0 - 100.0$ | **ML Feature** | Standardized platform/mock coding round score. |
| `communication_score`| `float` | $0.0 - 100.0$ | **ML Feature** | Group Discussion & HR communication rating. |
| `projects_count` | `int` | $0 - 6$ | **ML Feature** | Verified academic/capstone project portfolio count. |
| `internships_count` | `int` | $0 - 3$ | **ML Feature** | Completed industry internship count. |
| `certifications_count`| `int` | $0 - 5$ | **ML Feature** | Verified professional certifications count. |
| `leetcode_solved` | `int` | $0 - 550$ | **ML Feature** | Total coding problems solved on competitive platforms. |
| `github_profile` | `int` (0/1) | $0 \text{ or } 1$ | **ML Feature** | Active public GitHub repository presence. |
| `linkedin_profile` | `int` (0/1) | $0 \text{ or } 1$ | **ML Feature** | Completed professional LinkedIn profile presence. |
| `placement_status` | `int` (0/1) | $0 \text{ or } 1$ | **Primary ML Target** | Binary placement label (`1` = Placed, `0` = Not Placed). |
| `placed_role` | `str` | 5 Roles / Not Placed | **Downstream Target** | Ground-truth role alignment for career recommendation evaluation. |

---

## 2. Career Job-Role Taxonomy

For students with `placement_status = 1`, the role is assigned based on skill domain alignment:

1. **Software Development Engineer (SDE)**:
   - Primary drivers: High DSA score ($\ge 75$), high coding test score ($\ge 75$), and high LeetCode problem solve count ($\ge 200$).
2. **Data Analyst / Data Engineer**:
   - Primary drivers: High Python proficiency ($\ge 70$), strong SQL database proficiency ($\ge 70$), and quantitative aptitude.
3. **Backend / Full Stack Developer**:
   - Primary drivers: High Java/Web development score ($\ge 70$), relational databases (SQL), and project portfolio.
4. **Quality Assurance / Test Engineer**:
   - Primary drivers: Strong analytical aptitude, solid communication ratings, and foundational scripting.
5. **Systems / Support Engineer**:
   - Primary drivers: Solid communication skills, consistent academic foundation, and general technical aptitude.

Students with `placement_status = 0` are assigned `placed_role = "Not Placed"`.

---

## 3. Data Generation Mathematical Grounding

Unlike naive uniform random generation, the benchmark generator uses a **multi-factor latent ability framework**:

1. **Academic Correlation**: Higher 12th percentage correlates with higher university CGPA.
2. **Arrear Risk Function**: Probability of arrears increases exponentially when CGPA $< 7.0$:
   $$P(\text{Arrear History}) = \text{clip}\left(0.65 - (\text{CGPA} - 5.5) \times 0.12, 0.04, 0.60\right)$$
3. **Latent Problem-Solving Ability ($Z \sim \mathcal{N}(0, 1)$)**:
   - Connects Aptitude, Coding Score, DSA proficiency, and LeetCode solve counts.
4. **Placement Probability (Sigmoid Formulation)**:
   $$\text{Logit} = 0.18 Z_{\text{Acad}} + 0.32 Z_{\text{Tech}} + 0.18 Z_{\text{Apt}} + 0.14 Z_{\text{Comm}} + 0.18 Z_{\text{Exp}} - \text{Penalty}_{\text{Arrears}} + \epsilon$$
   $$\text{Placement Status} \sim \text{Bernoulli}(\sigma(\text{Logit}))$$
5. **Eligibility Filter**: Students with $\ge 2$ active standing arrears face strict campus screening failure ($P(\text{Placed}) \approx 0$).

---

## 4. How to Regenerate or Expand the Dataset

Run from project root:
```bash
# Generate 1,000 records (default)
python -m src.data.generate_dataset --samples 1000 --seed 42

# Generate custom batch (e.g. 2,500 records)
python -m src.data.generate_dataset --samples 2500 --seed 101 --outdir data/raw
```
