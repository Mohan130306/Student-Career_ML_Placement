==============================================================================
STUDENT DATASET AUDIT REPORT -- PHASE 2A
==============================================================================
Audit Phase        : Phase 2A — Data Audit Pipeline
Source Dataset     : data\raw\for project.xlsx
Selected Sheet     : Individual Personal Details
ML Target Status   : No placement outcome target present; supervised prediction deferred.
------------------------------------------------------------------------------
1. SCHEMA & DIMENSIONS
  Total Rows       : 39
  Total Columns    : 18 (Expected: 17)
  Exact Schema Match: FAIL
  Missing Columns  : ['No. OF CURRENT ARREARS']
  Extra Columns    : ['S.No', 'No. of CURRENT ARREARS']
  Casing Variations Detected:
    * Expected 'No. OF CURRENT ARREARS' but found 'No. of CURRENT ARREARS' (case/spacing difference)
------------------------------------------------------------------------------
2. MISSING VALUE SUMMARY
  Total Missing Cells : 38 (5.41%)
  Complete Records    : 1 (2.56%)
  - GIT HUB ID: 38 missing (97.44%)
------------------------------------------------------------------------------
3. DUPLICATE RECORDS
  Exact Duplicate Rows        : 0
  Duplicate Register Numbers   : 0
------------------------------------------------------------------------------
4. CATEGORICAL DISTRIBUTIONS (RAW SOURCE VALUES)
  * GENDER:
    Observed Values : ['MALE', 'FEMALE']
    Frequencies     : {'MALE': 24, 'FEMALE': 15}
  * DIPLOMA STUDENT:
    Observed Values : ['NO']
    Frequencies     : {'NO': 39}
  * HISTORY OF ARREAR (YES/NO):
    Observed Values : ['YES', 'NO']
    Frequencies     : {'YES': 21, 'NO': 18}
  * CURRENT ARREAR (YES/NO):
    Observed Values : ['NO', 'YES']
    Frequencies     : {'NO': 33, 'YES': 6}
------------------------------------------------------------------------------
5. NUMERIC RANGE & DATA QUALITY CHECKS
  * 10th Percentage:
    WARNING: 39 non-numeric invalid entries found!
  * 12th / Diploma Percentage:
    Range: [49.6666666666667 to 85.5], Mean: 75.27, Median: 78.17
  * 12th Cut Off:
    Range: [95.5 to 161.0], Mean: 138.32, Median: 140.5
  * CGPA (Semester 05):
    Range: [7.32 to 8.83], Mean: 8.18, Median: 8.16
  * No. of Current Arrears:
    Range: [0.0 to 2.0], Mean: 0.21, Median: 0.0
  * LeetCode Solved:
    Range: [10.0 to 439.0], Mean: 184.03, Median: 191.0
------------------------------------------------------------------------------
6. PROFILE LINK AVAILABILITY (DERIVED BINARY FLAGS)
  - RESUME LINK: 39 present (100.0%), 0 missing
  - LINKED - IN ID: 39 present (100.0%), 0 missing
  - HACKER RANK ID: 39 present (100.0%), 0 missing
  - GIT HUB ID: 1 present (2.56%), 38 missing
  Students with All 4 Profiles : 1
  Students with 0 Profiles    : 0
------------------------------------------------------------------------------
7. LOGICAL INCONSISTENCIES
  Total Inconsistencies Found : 0
==============================================================================
PRIVACY COMPLIANCE: No student names, register numbers, or URLs are logged.
==============================================================================