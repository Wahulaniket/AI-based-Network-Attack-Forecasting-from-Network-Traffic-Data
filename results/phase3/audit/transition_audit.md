# Phase 3.1 Transition Audit Report

## Verdict
**PHASE 3.1 TRANSITION AUDIT: PASS**

## 1. Load Data
- **S_t Rows/Cols**: 28706 / 77
- **Y_t Rows/Cols**: 28706 / 1

## 2. Timestamp Check
- **Min Timestamp**: 2018-02-14 01:00:00
- **Max Timestamp**: 2018-03-02 12:59:50
- **Is Chronological**: True
- **Duplicate Timestamps**: 0

## 3. Alignment
- **Row Alignment**: True

## 4. State Dimension
- **Exactly 77 Features**: True
- **Numeric Integrity**: True
- **No NaN/Inf**: True
- **Target Isolation**: True

## 5. Transition Construction
- **Total Valid Transitions**: 28705
*Note: Because empty windows were intentionally excluded, these transitions represent consecutive OBSERVED traffic states, not necessarily adjacent wall-clock 10-second intervals.*

## 6. Temporal Gap Audit
- **Exactly 10s Gaps**: 28318
- **> 10s Gaps**: 387
- **< 10s Gaps**: 0
- **Median Gap**: 10.0s

## 7. Causality Check
- **Passed**: True

## 8. Split Audit (Transitions)
- **TRAIN**: 22964 transitions (Start: 2018-02-14 01:00:00, End: 2018-03-01 03:14:40)
- **VALIDATION**: 2870 transitions (Start: 2018-03-01 03:14:40, End: 2018-03-02 01:45:10)
- **TEST**: 2871 transitions (Start: 2018-03-02 01:45:10, End: 2018-03-02 12:59:50)

## 9. Label Separation
- **TRAIN Attack %**: 18.05%
- **VAL Attack %**: 26.24%
- **TEST Attack %**: 61.76%
*(Labels are not used as input features, only for downstream validation)*

## 10. Manual Transition Examples (Inspection Only)
- **Ex 1**: 2018-02-14 01:00:00 -> 2018-02-14 01:00:10 (Gap: 10.0s). Flow count 131.0 -> 548.0. Labels: 0 -> 0
- **Ex 2**: 2018-02-15 10:22:50 -> 2018-02-15 10:23:00 (Gap: 10.0s). Flow count 201.0 -> 361.0. Labels: 0 -> 0
- **Ex 3**: 2018-02-20 12:59:00 -> 2018-02-20 12:59:10 (Gap: 10.0s). Flow count 2402.0 -> 2106.0. Labels: 0 -> 0
- **Ex 4**: 2018-02-23 08:45:30 -> 2018-02-23 08:45:40 (Gap: 10.0s). Flow count 286.0 -> 395.0. Labels: 0 -> 0
- **Ex 5**: 2018-03-01 03:14:40 -> 2018-03-01 03:14:50 (Gap: 10.0s). Flow count 57.0 -> 66.0. Labels: 1 -> 1
