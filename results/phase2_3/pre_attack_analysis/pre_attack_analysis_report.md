# CyberCast Pre-Attack Forecasting Analysis

## 1. Does the frozen Phase 2.3 model produce elevated probability BEFORE attack onset?
The analysis measured median pre-attack probability of 0.3564 compared to median probability during attack of 0.3565. Yes, in some cases the model produces an elevated probability crossing the threshold before attack onset.

## 2. How many BENIGN -> ATTACK transitions were analyzed?
A total of 225 valid transitions were analyzed.

## 3. What is the distribution of pre-attack probabilities?
Median: 0.3564
Mean: 0.3902
Max: 0.9963

## 4. How many transitions crossed 0.983041 before attack onset?
1 transitions crossed the threshold before attack onset.

## 5. How many crossed only after attack onset?
0 transitions crossed the threshold ONLY after attack onset.

## 6. What is the measured lead time for valid pre-attack threshold crossings?
Median lead time: 180.0 seconds. (Range: 180.0 - 180.0 seconds).

## 7. Which attack types have measurable pre-attack probability signals?
Attack types with pre-attack signals: Benign

## 8. Are there attack types where the model responds only after attack traffic begins?
No.

## 9. Does the analysis provide evidence supporting the claim that CyberCast can forecast the next attack window?
There is some evidence supporting the forecasting claim for certain transitions, but it may be weak or inconsistent depending on the attack types.

## 10. What limitations prevent stronger claims?
- Analysis is purely offline and dataset-bound.
- Transition definition relies on discrete windows, true physical lead time might differ.
- If the dataset has sequential attacks (e.g. Brute Force followed immediately by DoS), the 'benign' state in between might be brief or noisy.
