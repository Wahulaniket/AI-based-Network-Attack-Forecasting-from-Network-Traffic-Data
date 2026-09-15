# CyberCast Forecasting Methodology

## Scientific Limitations

1. **Phase 2.3 LSTM predicts attack probability:** The LSTM strictly outputs a risk probability `[0, 1]`. It does not perform multi-class ATT&CK stage classification.
2. **ATT&CK mapping is heuristic/evidence-based:** The `stage_mapper.py` interprets the network traffic features and assigns a stage if sufficient evidence is present. It operates entirely independently of the LSTM's learned weights.
3. **ATT&CK stage thresholds are not necessarily empirically validated:** The thresholds used in the mapper (e.g., `RST_SYN_Ratio > 0.5`) are human-crafted heuristics derived from domain knowledge, not empirically validated against labeled ground truth. They must be treated as `HEURISTIC / NOT EMPIRICALLY VALIDATED`.
4. **Next-stage forecasting is heuristic:** The progression engine does not implement a learned transition model (`P(S[t+1] | S[t])`). It is a `HEURISTIC NEXT-STAGE FORECAST` that probabilistically suggests the next logical stage based on current observed stage and risk trajectory.
5. **Perturbation attribution measures local model sensitivity:** The `explainability` module uses a sequence masking technique to observe how the LSTM's risk probability drops when a feature or temporal window is zeroed out. This measures *local model sensitivity / attribution* and does not constitute proof of internal neural causality.
6. **Not Ground-Truth Attribution:** None of these layers should be interpreted as guaranteed ground-truth attack attribution. They represent an evidence-based intelligence layer interpreting the traffic state.

## 1. What the LSTM Predicts
The Phase 2.3 `SET_R` LSTM model directly predicts **attack-risk probability** from raw network traffic features. It learns a continuous mapping from sequential history (twenty 10-second temporal windows) to a binary target variable representing whether the current window contains malicious activity.

## 2. What the ATT&CK Layer Interprets
The ATT&CK layer is a separate, deterministic, and evidence-based intelligence layer that infers the current attack stage (e.g., Reconnaissance, Command & Control, Exfiltration) based on strictly interpreted heuristics from the current network state. The system output uses careful language (e.g., "Traffic evidence is consistent with...") to avoid overclaiming model capabilities.

## 3. Stage Evidence Calculation
The ATT&CK mapper calculates evidence scores by evaluating specific features from the 89-feature `SET_R` representation. For instance:
- High `RST_SYN_Ratio` heavily scores towards **Reconnaissance** (port scanning).
- Extremely low `Fwd_Bwd_Byte_Ratio` scores towards **Exfiltration**.

## 4. Confidence Determination
Confidence is determined by the accumulated evidence score:
- **High**: Multiple strong heuristics met (`score >= 0.7`).
- **Medium**: Partial evidence met (`0.4 <= score < 0.7`).
- **Low**: Weak evidence (`0.3 <= score < 0.4`).
- **UNKNOWN / INSUFFICIENT EVIDENCE**: Evidence score below `0.3`. The system explicitly avoids guessing without evidence.

## 5. Next-Stage Forecasting
Next-stage forecasting is strictly a heuristic progression based on the cyber kill-chain. If risk is escalating (positive probability delta), confidence is increased. If risk is stable or decreasing, confidence is low, or the forecast returns `None`.

## 6. Information Exclusions
- Target labels (`Label`, `binary_attack`, etc.) are explicitly scrubbed and completely hidden from the intelligence layer.
- Future time windows are strictly excluded. 

## 7. Temporal Causality Enforcement
Temporal causality is enforced at the windowing and sequence generation layer. For any prediction at time $t$, the system only receives data up to time $t$. The historical risk trend used for forecasting only evaluates $t-1, t-2, \ldots$, preventing any future data from influencing current forecasts.
