# Explainable AI (SHAP) & Career Recommendation Engine

## AI-Powered Student Career & Placement Prediction System
- **Phases**: Phase 5 (Explainable AI) & Phase 6 (Career & Skill Recommendations)
- **Key Modules**:
  - `src/explainability/explainer.py` (`PlacementExplainer`)
  - `src/recommendations/engine.py` (`CareerRecommendationEngine`)

---

## 1. Explainable AI Architecture (SHAP)

Rather than treating student placement predictions as a black box, the system uses **Shapley Additive exPlanations (SHAP)** grounded in cooperative game theory to provide transparent feature attribution:

```
                      Student Profile
                             │
                             ▼
                    Placement Predictor
                     (Probability: 81.4%)
                             │
                             ▼
                  SHAP LinearExplainer
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Positive Amplifiers               Risk Penalties
   - Internships (+0.35 log-odds)    - Current Arrears (-0.28 log-odds)
   - Projects (+0.31 log-odds)       - Low Aptitude (-0.12 log-odds)
   - High DSA (+0.26 log-odds)
```

### Global Feature Attribution Findings
Global SHAP evaluation across the benchmark student cohort identified the top features driving placement likelihood:
1. **Industry Internships Count** (Mean $|SHAP| = 0.407$)
2. **Verified Projects Count** (Mean $|SHAP| = 0.399$)
3. **Portfolio Richness Score** (Mean $|SHAP| = 0.302$)
4. **Active Current Arrears Penalty** (Mean $|SHAP| = 0.274$)
5. **Arrear Burden Composite** (Mean $|SHAP| = 0.253$)
6. **Academic Consistency** (Mean $|SHAP| = 0.207$)
7. **Coding Dexterity Index** (Mean $|SHAP| = 0.202$)
8. **LeetCode Problems Solved** (Mean $|SHAP| = 0.196$)

---

## 2. Skill-Gap Diagnostic Engine

The skill-gap engine evaluates an individual's competency scores against the **empirically observed benchmarks** of placed students across 5 roles:

$$\text{Gap} = \text{Benchmark Score} - \text{Student Score}$$
$$\text{Fulfillment \%} = \min\left(100, \frac{\text{Student Score}}{\text{Benchmark Score}} \times 100\right)$$

* **Critical Deficit** ($\text{Gap} > 15.0$): Urgent blocker requiring intensive remediation.
* **Moderate Deficit** ($3.0 < \text{Gap} \le 15.0$): Secondary improvement area.
* **Proficient / Strength** ($\text{Gap} \le 3.0$): Meets or exceeds industry placement standard.

---

## 3. Career Role Compatibility Alignment

The system computes a multi-dimensional affinity index across all 5 career tracks:
- **Software Development Engineer (SDE)**: DSA, platform coding, LeetCode problem volume.
- **Data Analyst / Data Engineer**: Python, SQL, quantitative aptitude.
- **Backend / Full Stack Developer**: Java, Web Development, SQL.
- **Quality Assurance / Test Engineer**: Logical reasoning, test automation foundations, communication.
- **Systems / Support Engineer**: Consistent academics, infrastructure understanding, professional communication.

---

## 4. Corporate Eligibility Screening

Automated screening across 3 industry tiers:
* **Tier 1 (Product Companies — 8+ LPA)**: CGPA $\ge 7.50$, 0 standing arrears, $\le 1$ historical arrear, 10th/12th $\ge 70\%$, DSA $\ge 65$.
* **Tier 2 (Tech Solutions — 5 to 8 LPA)**: CGPA $\ge 6.50$, $\le 1$ standing arrear, 10th/12th $\ge 60\%$.
* **Tier 3 (IT Services / Mass Recruiters — 3.5 to 5 LPA)**: CGPA $\ge 6.00$, clear active arrears before joining, 10th/12th $\ge 60\%$.

---

## 5. Personalized 4-Week Tactical Preparation Roadmap

Generates a targeted, dynamic roadmap:
* **Week 1**: Foundational Recovery & Eligibility Stabilization (arrear remediation, quantitative speed math, array problem solving).
* **Week 2**: Intermediate Data Structures & Database Implementation (JOINs, aggregations, linked lists/stacks/queues).
* **Week 3**: Advanced Problem Solving & System Architecture (Trees, BFS/DFS, REST APIs, or Data analytics sprints).
* **Week 4**: Placement Drive Simulation & Profile Polish (Timed mock tests, STAR interview method, GitHub & LinkedIn portfolio polish).
