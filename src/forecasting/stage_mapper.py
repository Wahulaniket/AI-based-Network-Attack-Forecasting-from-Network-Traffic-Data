import pandas as pd
from src.forecasting.schemas import StageScore, StageEvidence

def infer_attack_stage(window_features: pd.Series) -> StageScore:
    """Deterministic, rule-based inference of the MITRE ATT&CK stage.
    
    Operates strictly on the 89 SET_R features of the current temporal window.
    Does NOT use probability, binary labels, or future data.
    """
    scores = {
        "Reconnaissance": 0.0,
        "Command and Control": 0.0,
        "Exfiltration": 0.0,
        "Impact": 0.0
    }
    evidence_log = {k: [] for k in scores.keys()}
    
    # Heuristics based on SET_R
    # 1. Reconnaissance (Port Scanning)
    if window_features.get('RST_SYN_Ratio', 0) > 0.5:
        scores["Reconnaissance"] += 0.4
        evidence_log["Reconnaissance"].append(StageEvidence("RST_SYN_Ratio", "High RST/SYN ratio indicates port scanning."))
        
    if window_features.get('Avg_Pkt_Size', 100) < 60:
        scores["Reconnaissance"] += 0.3
        evidence_log["Reconnaissance"].append(StageEvidence("Avg_Pkt_Size", "Small packet sizes consistent with scanning probes."))
        
    # 2. Command and Control (Periodic / Beaconing)
    if window_features.get('Flow IAT Mean', 0) > 500000: # 0.5s avg inter-arrival
        scores["Command and Control"] += 0.4
        evidence_log["Command and Control"].append(StageEvidence("Flow IAT Mean", "High mean IAT may indicate periodic beaconing."))
        
    if 0 < window_features.get('Total_Bytes', 0) < 5000:
        scores["Command and Control"] += 0.3
        evidence_log["Command and Control"].append(StageEvidence("Total_Bytes", "Low data volume consistent with C2 check-ins."))
        
    # 3. Exfiltration (Data Transfer out)
    if window_features.get('Fwd_Bwd_Byte_Ratio', 1) < 0.2:
        scores["Exfiltration"] += 0.5
        evidence_log["Exfiltration"].append(StageEvidence("Fwd_Bwd_Byte_Ratio", "High relative backward byte volume indicates data exfiltration."))
        
    # 4. Impact (DoS/DDoS)
    if window_features.get('Pkts_Per_Sec', 0) > 5000:
        scores["Impact"] += 0.5
        evidence_log["Impact"].append(StageEvidence("Pkts_Per_Sec", "Extremely high packet rate indicative of DoS attack."))
        
    if window_features.get('SYN_FIN_Ratio', 0) > 0.8:
        scores["Impact"] += 0.3
        evidence_log["Impact"].append(StageEvidence("SYN_FIN_Ratio", "High SYN/FIN ratio indicates SYN flood (DoS)."))
        
    # Determine the most likely stage
    best_stage = max(scores, key=scores.get)
    best_score = scores[best_stage]
    
    if best_score < 0.3:
        return StageScore(
            stage="UNKNOWN / INSUFFICIENT EVIDENCE",
            score=best_score,
            confidence="low",
            evidence=[]
        )
        
    confidence = "high" if best_score >= 0.7 else ("medium" if best_score >= 0.4 else "low")
    
    return StageScore(
        stage=best_stage,
        score=best_score,
        confidence=confidence,
        evidence=evidence_log[best_stage]
    )
