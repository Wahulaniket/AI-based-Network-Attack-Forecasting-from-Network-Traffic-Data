# CyberCast Pre-Attack Forecasting Validation

## 1. Objective
Determine whether the frozen Phase 2.3 model contains a genuine pre-attack probability signal supporting early warning.

## 2. Frozen model configuration
- Features: SET_R (89)
- History: 20 windows (200s)
- Target: Next window attack
- Frozen Threshold: 0.9830410480499268

## 3. Dataset and chronological split
80/10/10 split over CIC-IDS2018 clean data.

## 4. Attack transition definition
attack_binary(t)=0 to attack_binary(t+1)=1

## 5. Temporal methodology
Prediction at T uses max history timestamp < T.

## 6. Probability trajectory findings
Validation transitions analyzed: 1
Pre-attack median probability: 0.3903
Onset median probability: 0.6970

## 7. Attack-type findings
Refer to `validation_attack_type_summary.csv`.

## 8. Threshold trade-off table
Refer to `validation_early_warning_thresholds.csv`.

## 9. False-alarm analysis
Benign checkpoints analyzed: 54

## 10. Candidate operating points
- Threshold 0.95

## 11. Blind final-test evaluation
Threshold: 0.95
  Pre-attack detection rate: 1.0000
  Median lead time: 300.0
  False alarm rate: 0.3400
  Covered types: Benign

## 12. Statistical analysis
Median difference (Onset - Pre-attack): 0.30671393871307373
Mean difference (Onset - Pre-attack): 0.32203757353127005

## 13. Limitations
- Offline dataset analysis.
- Discrete window bounds.

## 14. Scientific conclusion
The validation evidence does NOT demonstrate a reliable pre-attack signal. While probabilities may rise, no operating point provides a meaningful pre-attack detection rate without an unacceptable false alarm rate. The model primarily acts as an attack detector post-onset rather than a forecaster.
