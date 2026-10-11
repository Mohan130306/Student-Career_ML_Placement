"""
Synthetic Placement Benchmark Dataset Generator
AI-Powered Student Career & Placement Prediction System

Generates a realistic, statistically coherent benchmark dataset for student career
and placement prediction modeling, skill-gap analysis, and institutional analytics.

Design Principles:
1. Multi-factor Grounding: Attributes reflect real engineering student distributions.
2. Realistic Correlation: Academics, coding dexterity, internships, and communication
   probabilistically determine placement rather than random independent noise.
3. Strict Governance: Distinct separation between student IDs, model features,
   demographic audit features, and downstream target roles.
4. Reproducibility: Seeded random generation ensuring deterministic outputs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import sys
import numpy as np
import pandas as pd


# =====================================================================
# COLUMN NAMES SPECIFICATION
# =====================================================================
COL_STUDENT_ID = "student_id"
COL_DEPARTMENT = "department"
COL_GENDER = "gender"
COL_10TH_PERCENT = "tenth_percentage"
COL_12TH_PERCENT = "twelfth_percentage"
COL_CGPA = "cgpa"
COL_HISTORY_OF_ARREARS = "history_of_arrears"
COL_CURRENT_ARREARS = "current_arrears"

COL_DSA_SCORE = "dsa_score"
COL_JAVA_SCORE = "java_score"
COL_PYTHON_SCORE = "python_score"
COL_SQL_SCORE = "sql_score"
COL_WEB_DEV_SCORE = "web_dev_score"

COL_APTITUDE_SCORE = "aptitude_score"
COL_CODING_SCORE = "coding_score"
COL_COMMUNICATION_SCORE = "communication_score"

COL_PROJECTS_COUNT = "projects_count"
COL_INTERNSHIPS_COUNT = "internships_count"
COL_CERTIFICATIONS_COUNT = "certifications_count"
COL_LEETCODE_SOLVED = "leetcode_solved"
COL_GITHUB_PROFILE = "github_profile"
COL_LINKEDIN_PROFILE = "linkedin_profile"

COL_PLACEMENT_STATUS = "placement_status"
COL_PLACED_ROLE = "placed_role"

ALL_DATASET_COLUMNS = [
    COL_STUDENT_ID,
    COL_DEPARTMENT,
    COL_GENDER,
    COL_10TH_PERCENT,
    COL_12TH_PERCENT,
    COL_CGPA,
    COL_HISTORY_OF_ARREARS,
    COL_CURRENT_ARREARS,
    COL_DSA_SCORE,
    COL_JAVA_SCORE,
    COL_PYTHON_SCORE,
    COL_SQL_SCORE,
    COL_WEB_DEV_SCORE,
    COL_APTITUDE_SCORE,
    COL_CODING_SCORE,
    COL_COMMUNICATION_SCORE,
    COL_PROJECTS_COUNT,
    COL_INTERNSHIPS_COUNT,
    COL_CERTIFICATIONS_COUNT,
    COL_LEETCODE_SOLVED,
    COL_GITHUB_PROFILE,
    COL_LINKEDIN_PROFILE,
    COL_PLACEMENT_STATUS,
    COL_PLACED_ROLE,
]

# Roles taxonomy
ROLE_NOT_PLACED = "Not Placed"
ROLE_SDE = "Software Development Engineer (SDE)"
ROLE_DATA_ANALYST = "Data Analyst / Data Engineer"
ROLE_BACKEND_FULLSTACK = "Backend / Full Stack Developer"
ROLE_QA_TEST = "Quality Assurance / Test Engineer"
ROLE_SYSTEMS_SUPPORT = "Systems / Support Engineer"

ROLE_LIST = [
    ROLE_SDE,
    ROLE_DATA_ANALYST,
    ROLE_BACKEND_FULLSTACK,
    ROLE_QA_TEST,
    ROLE_SYSTEMS_SUPPORT,
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def generate_placement_dataset(
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic student placement benchmark DataFrame.

    Args:
        n_samples: Number of student records to generate.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Complete synthetic dataset.
    """
    rng = np.random.default_rng(random_state)

    # 1. Identifiers
    student_ids = [f"STU_{2026}_{i+1:04d}" for i in range(n_samples)]

    # 2. Demographics & Academics
    departments = rng.choice(
        ["CSE", "IT", "ECE", "AIDS", "CSBS"],
        size=n_samples,
        p=[0.35, 0.25, 0.20, 0.12, 0.08],
    )
    genders = rng.choice(["Male", "Female"], size=n_samples, p=[0.54, 0.46])

    # 10th and 12th percentages (normal distributions)
    tenth_pct = np.clip(rng.normal(loc=82.0, scale=8.5, size=n_samples), 55.0, 99.5)
    # 12th is moderately correlated with 10th
    twelfth_pct = np.clip(
        0.55 * tenth_pct + rng.normal(loc=33.0, scale=7.5, size=n_samples),
        50.0,
        98.5,
    )

    # CGPA (on 10.0 scale) correlated with 12th percentage
    cgpa_base = 3.2 + (twelfth_pct / 100.0) * 5.0
    cgpa = np.clip(cgpa_base + rng.normal(loc=0.0, scale=0.55, size=n_samples), 5.5, 9.85)
    cgpa = np.round(cgpa, 2)
    tenth_pct = np.round(tenth_pct, 1)
    twelfth_pct = np.round(twelfth_pct, 1)

    # Arrears logic: lower CGPA -> higher chance of arrears
    # Probability of history of arrears is higher when CGPA < 7.2
    arrear_prob = np.clip(0.65 - (cgpa - 5.5) * 0.12, 0.04, 0.60)
    history_arrears = (rng.uniform(size=n_samples) < arrear_prob).astype(int)

    # Current arrears: only students with history can have current arrears
    current_arrears = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if history_arrears[i] == 1:
            # 70% have cleared their arrears, 20% have 1, 10% have 2+
            current_arrears[i] = rng.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.07, 0.03])

    # 3. Core Assessments (Aptitude, Coding, Communication)
    # Latent general problem solving ability factor (Z ~ N(0, 1))
    latent_ability = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    aptitude = np.clip(62.0 + latent_ability * 12.0 + (cgpa - 7.5) * 4.0 + rng.normal(0, 7.0, n_samples), 20.0, 99.0)
    coding_score = np.clip(58.0 + latent_ability * 16.0 + rng.normal(0, 9.0, n_samples), 15.0, 99.0)
    communication = np.clip(65.0 + rng.normal(loc=0.0, scale=12.0, size=n_samples), 25.0, 99.0)

    # 4. Technical Skill Ratings (0 - 100)
    dsa_score = np.clip(0.65 * coding_score + latent_ability * 8.0 + rng.normal(15, 8, n_samples), 10.0, 99.0)
    python_score = np.clip(55.0 + latent_ability * 10.0 + rng.normal(0, 14, n_samples), 15.0, 99.0)
    java_score = np.clip(52.0 + latent_ability * 10.0 + rng.normal(0, 15, n_samples), 15.0, 99.0)
    sql_score = np.clip(56.0 + (aptitude / 100.0) * 20.0 + rng.normal(10, 12, n_samples), 20.0, 99.0)
    web_dev_score = np.clip(50.0 + rng.normal(0, 18, n_samples) + (python_score + java_score) * 0.15, 10.0, 99.0)

    # Department affinity adjustments
    # ECE students slightly more hardware/systems focused, CSE/IT higher web/dsa
    for i in range(n_samples):
        if departments[i] in ["CSE", "IT"]:
            dsa_score[i] = min(99.0, dsa_score[i] + 4.0)
            web_dev_score[i] = min(99.0, web_dev_score[i] + 3.0)
        elif departments[i] == "AIDS":
            python_score[i] = min(99.0, python_score[i] + 7.0)
            sql_score[i] = min(99.0, sql_score[i] + 5.0)

    # 5. Practical Experience & Profiles
    # LeetCode solved correlated with DSA score and coding score
    leetcode_mean = np.maximum(5, (dsa_score - 20) * 4.2 + (coding_score - 20) * 1.5)
    leetcode_solved = np.clip(
        np.round(rng.normal(loc=leetcode_mean, scale=35.0)),
        0,
        550,
    ).astype(int)

    # Projects count (Poisson distributed, mean ~2.4)
    projects_count = np.clip(rng.poisson(lam=2.3, size=n_samples), 0, 6)

    # Internships count: students with higher CGPA and coding score more likely to get internships
    intern_logit = (cgpa - 7.5) * 0.7 + (coding_score - 60) * 0.03 + (projects_count - 2) * 0.3
    intern_prob = sigmoid(intern_logit)
    internships_count = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p = intern_prob[i]
        internships_count[i] = rng.choice([0, 1, 2, 3], p=[max(0.0, 1.0 - p), p * 0.7, p * 0.25, p * 0.05] / np.sum([max(0.0, 1.0 - p), p * 0.7, p * 0.25, p * 0.05]))

    # Certifications count
    certifications_count = np.clip(rng.poisson(lam=1.6, size=n_samples), 0, 5)

    # Online Profiles (GitHub, LinkedIn)
    github_prob = sigmoid((coding_score - 50) * 0.05 + (projects_count - 1) * 0.4)
    github_profile = (rng.uniform(size=n_samples) < github_prob).astype(int)

    linkedin_prob = sigmoid((communication - 50) * 0.04 + (internships_count) * 0.5)
    linkedin_profile = (rng.uniform(size=n_samples) < linkedin_prob).astype(int)

    # 6. Placement Outcome Calculation (Multi-Factor Probabilistic Model)
    # Real placement evaluation components:
    # Academics (15%), Technical & DSA (30%), Aptitude (20%), Communication (15%), Experience (20%)
    z_academics = (cgpa - 7.0) / 1.0
    z_tech = ((dsa_score * 0.5 + coding_score * 0.5) - 60.0) / 15.0
    z_aptitude = (aptitude - 60.0) / 15.0
    z_comm = (communication - 60.0) / 15.0
    z_experience = (internships_count * 1.5 + projects_count * 0.6 + (leetcode_solved / 150.0)) - 2.0

    # Arrears penalty: current arrears heavily penalizes campus eligibility
    arrears_penalty = current_arrears * 1.6 + history_arrears * 0.4

    # Combined latent placement index
    placement_latent = (
        0.18 * z_academics
        + 0.32 * z_tech
        + 0.18 * z_aptitude
        + 0.14 * z_comm
        + 0.18 * z_experience
        - arrears_penalty
        + rng.normal(0, 0.45, n_samples)  # Unobserved interview day variance
    )

    placement_prob = sigmoid(placement_latent)
    placement_status = (rng.uniform(size=n_samples) < placement_prob).astype(int)

    # Hard eligibility safeguard: Current arrears > 2 almost never placed in early campus rounds
    for i in range(n_samples):
        if current_arrears[i] >= 2 and rng.uniform() < 0.85:
            placement_status[i] = 0

    # 7. Job Role Assignment (Multi-class Career Alignment)
    placed_roles = []
    for i in range(n_samples):
        if placement_status[i] == 0:
            placed_roles.append(ROLE_NOT_PLACED)
            continue

        # For placed students, determine the best aligned job role
        sde_score = 0.45 * dsa_score[i] + 0.35 * coding_score[i] + 0.20 * (leetcode_solved[i] / 4.0)
        data_score = 0.40 * python_score[i] + 0.35 * sql_score[i] + 0.25 * aptitude[i]
        backend_score = 0.35 * java_score[i] + 0.35 * web_dev_score[i] + 0.30 * sql_score[i]
        qa_score = 0.35 * aptitude[i] + 0.35 * communication[i] + 0.30 * (100.0 - abs(coding_score[i] - 55.0))
        systems_score = 0.35 * communication[i] + 0.35 * (cgpa[i] * 10.0) + 0.30 * aptitude[i]

        scores = {
            ROLE_SDE: sde_score,
            ROLE_DATA_ANALYST: data_score,
            ROLE_BACKEND_FULLSTACK: backend_score,
            ROLE_QA_TEST: qa_score,
            ROLE_SYSTEMS_SUPPORT: systems_score,
        }

        # Role assignment with high affinity to top score
        best_role = max(scores, key=scores.get)
        placed_roles.append(best_role)

    # 8. Assemble Clean DataFrame
    df = pd.DataFrame({
        COL_STUDENT_ID: student_ids,
        COL_DEPARTMENT: departments,
        COL_GENDER: genders,
        COL_10TH_PERCENT: np.round(tenth_pct, 1),
        COL_12TH_PERCENT: np.round(twelfth_pct, 1),
        COL_CGPA: np.round(cgpa, 2),
        COL_HISTORY_OF_ARREARS: history_arrears,
        COL_CURRENT_ARREARS: current_arrears,
        COL_DSA_SCORE: np.round(dsa_score, 1),
        COL_JAVA_SCORE: np.round(java_score, 1),
        COL_PYTHON_SCORE: np.round(python_score, 1),
        COL_SQL_SCORE: np.round(sql_score, 1),
        COL_WEB_DEV_SCORE: np.round(web_dev_score, 1),
        COL_APTITUDE_SCORE: np.round(aptitude, 1),
        COL_CODING_SCORE: np.round(coding_score, 1),
        COL_COMMUNICATION_SCORE: np.round(communication, 1),
        COL_PROJECTS_COUNT: projects_count,
        COL_INTERNSHIPS_COUNT: internships_count,
        COL_CERTIFICATIONS_COUNT: certifications_count,
        COL_LEETCODE_SOLVED: leetcode_solved,
        COL_GITHUB_PROFILE: github_profile,
        COL_LINKEDIN_PROFILE: linkedin_profile,
        COL_PLACEMENT_STATUS: placement_status,
        COL_PLACED_ROLE: placed_roles,
    })

    return df


def save_benchmark_datasets(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/raw",
    base_name: str = "student_placement_benchmark",
) -> Tuple[Path, Path]:
    """
    Saves generated dataset to both CSV and Excel formats.

    Args:
        df: The DataFrame to save.
        output_dir: Target directory path.
        base_name: Base filename prefix.

    Returns:
        Tuple[Path, Path]: Paths to saved CSV and Excel files.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / f"{base_name}.csv"
    excel_path = out_path / f"{base_name}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, sheet_name="Student Profiles", index=False)

    return csv_path, excel_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic student placement benchmark dataset."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of student records to generate (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw).",
    )
    args = parser.parse_args()

    print(f"Generating {args.samples} benchmark student records (seed={args.seed})...")
    df = generate_placement_dataset(n_samples=args.samples, random_state=args.seed)

    csv_path, excel_path = save_benchmark_datasets(df, output_dir=args.outdir)

    placed_count = int(df[COL_PLACEMENT_STATUS].sum())
    placed_pct = (placed_count / len(df)) * 100.0

    print("=" * 60)
    print("BENCHMARK DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Records Generated : {len(df)}")
    print(f"Total Columns           : {len(df.columns)}")
    print(f"Overall Placement Rate  : {placed_count}/{len(df)} ({placed_pct:.1f}%)")
    print("\nRole Breakdown:")
    print(df[COL_PLACED_ROLE].value_counts().to_string())
    print("\nFiles Saved Successfully:")
    print(f"  CSV   : {csv_path}")
    print(f"  Excel : {excel_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
