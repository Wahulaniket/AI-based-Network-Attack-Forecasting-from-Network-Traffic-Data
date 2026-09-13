"""
Update CyberCast_Complete.ipynb with threshold analysis cells.
Inserts new cells and modifies existing ones without touching the model training.
"""

import json
import copy

NB_PATH = 'nootebooks/CyberCast_Complete.ipynb'

# Load notebook
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else source.split('\n')
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source if isinstance(source, list) else source.split('\n'),
        "outputs": [],
        "execution_count": None
    }

# Fix cell source to be list of lines (with newlines)
def fix_source(text):
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result

cells = nb['cells']

# ============================================================
# 1. INSERT: After cell 29 (test_evaluation plots), add threshold analysis
# ============================================================

# Section 18A: Threshold Sweep Analysis
md_18a = fix_source("""---
## Section 18A -- Threshold / Operating-Point Analysis

Comprehensive threshold sweep to find optimal operating points for the
CyberCast World Model. The F1-optimal threshold serves as a **research**
threshold, while a separate **operational** threshold is selected using
SOC-oriented criteria.

**All thresholds selected on VALIDATION data only. Test labels are NOT used
for threshold selection.**""")

code_18a = fix_source("""# ============================================================
# SECTION 18A -- Threshold Sweep on VALIDATION Data
# ============================================================
# NOTE: Thresholds selected exclusively on VALIDATION data.
#       Test labels are NOT used for threshold selection.

thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
assert len(thresholds) == 91, f"Expected 91 thresholds, got {len(thresholds)}"
print(f"Sweeping {len(thresholds)} thresholds on VALIDATION set: {thresholds[0]} to {thresholds[-1]}")

threshold_results = []
for t in thresholds:
    bins = (val_preds >= t).astype(int)
    cm_t = confusion_matrix(val_tgts, bins, labels=[0, 1])
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0
    fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
    fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0
    j_t = rec_t - fpr_t

    threshold_results.append({
        'threshold': t, 'precision': prec_t, 'recall': rec_t,
        'f1': f1_t, 'fpr': fpr_t, 'fnr': fnr_t, 'youdens_j': j_t,
        'TP': tp_t, 'TN': tn_t, 'FP': fp_t, 'FN': fn_t
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(RESULTS_DIR / 'threshold_analysis_val.csv', index=False)
print(f"Saved: threshold_analysis_val.csv ({len(threshold_df)} rows)")

# Print table
print(f"\\n{'Threshold':>10s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'FPR':>8s} {'FNR':>8s} {'J':>8s}")
print("-" * 62)
for _, row in threshold_df.iterrows():
    print(f"{row['threshold']:>10.2f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
          f"{row['f1']:>8.4f} {row['fpr']:>8.4f} {row['fnr']:>8.4f} {row['youdens_j']:>8.4f}")""")

# Section 18B: Threshold Selection
md_18b = fix_source("""---
## Section 18B -- Threshold Selection (Validation Only)

Three candidate thresholds:
1. **Research threshold**: F1-optimal on validation
2. **Youden's J threshold**: Maximizes `recall - FPR` (reported candidate)
3. **Operational threshold**: SOC-oriented selection

SOC selection rule:
- Among thresholds with recall ≥ 0.90, prefer lowest FPR
- If recall ≥ 0.90 requires FPR > 50%, fall back to Youden's J""")

code_18b = fix_source("""# ============================================================
# SECTION 18B -- Threshold Selection (VALIDATION DATA ONLY)
# ============================================================

# --- Research Threshold (F1-optimal) ---
best_f1_idx = threshold_df['f1'].idxmax()
research_threshold = threshold_df.loc[best_f1_idx, 'threshold']
research_row = threshold_df.loc[best_f1_idx]

print("RESEARCH THRESHOLD (F1-optimal on validation):")
print(f"  Threshold: {research_threshold}")
print(f"  Precision: {research_row['precision']:.4f}  Recall: {research_row['recall']:.4f}")
print(f"  F1: {research_row['f1']:.4f}  FPR: {research_row['fpr']:.4f}  FNR: {research_row['fnr']:.4f}")

# --- Youden's J Threshold ---
best_j_idx = threshold_df['youdens_j'].idxmax()
youdens_threshold = threshold_df.loc[best_j_idx, 'threshold']
youdens_row = threshold_df.loc[best_j_idx]

print(f"\\nYOUDEN'S J THRESHOLD (candidate):")
print(f"  Threshold: {youdens_threshold}")
print(f"  Precision: {youdens_row['precision']:.4f}  Recall: {youdens_row['recall']:.4f}")
print(f"  F1: {youdens_row['f1']:.4f}  FPR: {youdens_row['fpr']:.4f}  J: {youdens_row['youdens_j']:.4f}")

# --- Operational Threshold (SOC-oriented) ---
print(f"\\nOPERATIONAL THRESHOLD SELECTION (SOC-oriented):")
high_recall_df = threshold_df[threshold_df['recall'] >= 0.90].copy()
recall_90_achievable = len(high_recall_df) > 0
recall_90_reasonable = False

if recall_90_achievable:
    min_fpr_at_90 = high_recall_df['fpr'].min()
    print(f"  Recall >= 0.90: {len(high_recall_df)} thresholds, FPR range [{min_fpr_at_90:.4f}, {high_recall_df['fpr'].max():.4f}]")
    if min_fpr_at_90 <= 0.50:
        recall_90_reasonable = True
        best_op_idx = high_recall_df['fpr'].idxmin()
        operational_threshold = high_recall_df.loc[best_op_idx, 'threshold']
        operational_row = high_recall_df.loc[best_op_idx]
        print(f"  Selected: t={operational_threshold} (lowest FPR at recall >= 0.90)")
    else:
        print(f"  WARNING: Recall >= 0.90 requires FPR >= {min_fpr_at_90:.4f} (too high)")
        print(f"  The model's bimodal prediction distribution prevents high recall at low FPR.")
else:
    print("  Recall >= 0.90 NOT achievable at any threshold.")

if not recall_90_reasonable:
    print(f"\\n  Best available operating points at relaxed recall levels:")
    for min_rec in [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15]:
        cands = threshold_df[threshold_df['recall'] >= min_rec]
        if len(cands) > 0:
            bi = cands['fpr'].idxmin()
            r = cands.loc[bi]
            note = ' <-- FPR too high' if min_rec == 0.90 and r['fpr'] > 0.50 else ''
            print(f"    Recall>={min_rec:.2f}: t={r['threshold']:.2f} Recall={r['recall']:.4f} "
                  f"FPR={r['fpr']:.4f} F1={r['f1']:.4f}{note}")
    operational_threshold = youdens_threshold
    operational_row = youdens_row
    print(f"\\n  DECISION: Using Youden's J threshold ({operational_threshold})")
    print(f"  Rationale: Best tradeoff between recall and FPR on validation data.")

print(f"\\n{'='*60}")
print(f"LOCKED THRESHOLDS (validation-only, FINAL):")
print(f"  Research:    {research_threshold}")
print(f"  Operational: {operational_threshold}")
print(f"  Youden's J:  {youdens_threshold} (candidate)")
print(f"{'='*60}")
print(f"\\nSTATEMENT: Thresholds were selected exclusively on validation")
print(f"data. Test labels were not used for threshold selection.")""")

# Section 18C: Threshold Analysis Plots
md_18c = fix_source("""---
## Section 18C -- Threshold Analysis Plots""")

code_18c = fix_source("""# ============================================================
# SECTION 18C -- Threshold Analysis Plots
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('CyberCast Threshold Analysis (Validation Set)', fontsize=16, fontweight='bold', y=0.98)

# F1 vs Threshold
ax = axes[0, 0]
ax.plot(threshold_df['threshold'], threshold_df['f1'], 'b-', lw=2, label='F1')
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research (t={research_threshold})')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational (t={operational_threshold})')
if youdens_threshold != research_threshold and youdens_threshold != operational_threshold:
    ax.axvline(x=youdens_threshold, color='orange', ls=':', alpha=0.7, label=f'Youden J (t={youdens_threshold})')
ax.set_xlabel('Threshold'); ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs Threshold', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0.05, 0.95)

# Precision vs Threshold
ax = axes[0, 1]
ax.plot(threshold_df['threshold'], threshold_df['precision'], 'r-', lw=2)
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational')
ax.set_xlabel('Threshold'); ax.set_ylabel('Precision')
ax.set_title('Precision vs Threshold', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0.05, 0.95)

# Recall vs Threshold
ax = axes[0, 2]
ax.plot(threshold_df['threshold'], threshold_df['recall'], 'g-', lw=2)
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational')
ax.axhline(y=0.90, color='gray', ls=':', alpha=0.5, label='Recall = 0.90')
ax.set_xlabel('Threshold'); ax.set_ylabel('Recall')
ax.set_title('Recall vs Threshold', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0.05, 0.95)

# FPR vs Threshold
ax = axes[1, 0]
ax.plot(threshold_df['threshold'], threshold_df['fpr'], 'm-', lw=2)
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational')
ax.set_xlabel('Threshold'); ax.set_ylabel('False Positive Rate')
ax.set_title('FPR vs Threshold', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0.05, 0.95)

# Precision-Recall Tradeoff
ax = axes[1, 1]
ax.plot(threshold_df['recall'], threshold_df['precision'], 'k-', lw=2, alpha=0.7)
ax.scatter([research_row['recall']], [research_row['precision']],
           c='red', s=150, zorder=5, marker='*', label=f'Research (t={research_threshold})')
ax.scatter([operational_row['recall']], [operational_row['precision']],
           c='green', s=150, zorder=5, marker='D', label=f'Operational (t={operational_threshold})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Tradeoff', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Youden's J vs Threshold
ax = axes[1, 2]
ax.plot(threshold_df['threshold'], threshold_df['youdens_j'], 'c-', lw=2)
ax.axvline(x=youdens_threshold, color='orange', ls='--', alpha=0.8, label=f'Best J (t={youdens_threshold})')
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.6)
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.6)
ax.set_xlabel('Threshold'); ax.set_ylabel("Youden's J")
ax.set_title("Youden's J vs Threshold", fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0.05, 0.95)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(PLOTS_DIR / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: threshold_analysis.png")""")

# Section 18D: Test Evaluation at Both Thresholds
md_18d = fix_source("""---
## Section 18D -- Test Evaluation at Locked Thresholds

Single application of locked thresholds to the untouched test set.
3-way comparison: LR vs CyberCast@Research vs CyberCast@Operational.""")

code_18d = fix_source("""# ============================================================
# SECTION 18D -- Test Evaluation at Locked Thresholds
# ============================================================
# NOTE: Thresholds LOCKED from validation. This is the ONLY
#       application to the test set.

def evaluate_at_threshold(preds, tgts, threshold, name=""):
    bins = (preds >= threshold).astype(int)
    cm_e = confusion_matrix(tgts, bins, labels=[0, 1])
    tn_e, fp_e, fn_e, tp_e = cm_e.ravel()
    metrics = {
        'precision': float(precision_score(tgts, bins, zero_division=0)),
        'recall': float(recall_score(tgts, bins, zero_division=0)),
        'f1': float(f1_score(tgts, bins, zero_division=0)),
        'roc_auc': float(roc_auc_score(tgts, preds)) if len(np.unique(tgts)) > 1 else 0.0,
        'pr_auc': float(average_precision_score(tgts, preds)) if len(np.unique(tgts)) > 1 else 0.0,
        'fpr': float(fp_e / (fp_e + tn_e)) if (fp_e + tn_e) > 0 else 0.0,
        'fnr': float(fn_e / (fn_e + tp_e)) if (fn_e + tp_e) > 0 else 0.0,
        'TP': int(tp_e), 'TN': int(tn_e), 'FP': int(fp_e), 'FN': int(fn_e)
    }
    if name:
        print(f"\\n  {name} (threshold={threshold}):")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v}")
    return metrics

test_research = evaluate_at_threshold(test_preds, test_tgts, research_threshold, "CyberCast @ Research")
test_operational = evaluate_at_threshold(test_preds, test_tgts, operational_threshold, "CyberCast @ Operational")

# 3-way comparison
print("\\n" + "=" * 70)
print("   3-WAY MODEL COMPARISON (TEST SET)")
print("=" * 70)
comp_metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'fpr', 'fnr']
print(f"{'Metric':20s} {'LR':>12s} {'CC@Research':>12s} {'CC@Operational':>15s} {'Winner':>10s}")
print("-" * 72)
for m in comp_metrics:
    lv = lr_metrics[m]
    rv = test_research[m]
    ov = test_operational[m]
    if m in ['fpr', 'fnr']:
        best = min(lv, rv, ov)
        winner = 'LR' if lv == best else ('CC@Res' if rv == best else 'CC@Oper')
    else:
        best = max(lv, rv, ov)
        winner = 'LR' if lv == best else ('CC@Res' if rv == best else 'CC@Oper')
    print(f"{m:20s} {lv:>12.4f} {rv:>12.4f} {ov:>15.4f} {winner:>10s}")

print(f"\\nConfusion Matrices:")
print(f"  LR:             TN={cm.ravel()[0]:>5,} FP={cm.ravel()[1]:>5,} FN={cm.ravel()[2]:>5,} TP={cm.ravel()[3]:>5,}")
print(f"  CC@Research:    TN={test_research['TN']:>5,} FP={test_research['FP']:>5,} FN={test_research['FN']:>5,} TP={test_research['TP']:>5,}")
print(f"  CC@Operational: TN={test_operational['TN']:>5,} FP={test_operational['FP']:>5,} FN={test_operational['FN']:>5,} TP={test_operational['TP']:>5,}")

comparison_df = pd.DataFrame({
    'LogisticRegression': {m: lr_metrics[m] for m in comp_metrics},
    'CyberCast_Research': {m: test_research[m] for m in comp_metrics},
    'CyberCast_Operational': {m: test_operational[m] for m in comp_metrics},
})
comparison_df.to_csv(RESULTS_DIR / 'metrics_comparison.csv')
print("Saved: metrics_comparison.csv")""")

# Section 18E: False Positive Analysis
md_18e = fix_source("""---
## Section 18E -- False Positive Analysis (Test Set)""")

code_18e = fix_source("""# ============================================================
# SECTION 18E -- False Positive Analysis (TEST SET)
# ============================================================

fp_analysis = {}
for thresh_name, thresh_val in [('Operational', operational_threshold), ('Research', research_threshold)]:
    t_bins = (test_preds >= thresh_val).astype(int)
    cm_fp = confusion_matrix(test_tgts, t_bins, labels=[0, 1])
    tn_fp, fp_fp, fn_fp, tp_fp = cm_fp.ravel()
    total_benign = int(tn_fp + fp_fp)
    fp_pct = fp_fp / total_benign * 100 if total_benign > 0 else 0

    print(f"\\n--- {thresh_name} threshold ({thresh_val}) ---")
    print(f"  Total benign windows: {total_benign:,}")
    print(f"  False positives:      {int(fp_fp):,} ({fp_pct:.2f}%)")
    print(f"  True negatives:       {int(tn_fp):,}")

    # FP bursts
    fp_mask = ((test_tgts == 0) & (t_bins == 1))
    bursts = []
    in_b, b_start, b_len = False, 0, 0
    for i in range(len(fp_mask)):
        if fp_mask[i]:
            if not in_b: in_b, b_start, b_len = True, i, 1
            else: b_len += 1
        else:
            if in_b:
                bursts.append({'start': b_start, 'length': b_len})
                in_b = False
    if in_b: bursts.append({'start': b_start, 'length': b_len})

    print(f"  FP burst count: {len(bursts)}")
    if bursts:
        bl = [b['length'] for b in bursts]
        print(f"  Mean burst: {np.mean(bl):.1f} windows ({np.mean(bl)*WINDOW_SECONDS:.1f}s)")
        print(f"  Max burst:  {max(bl)} windows ({max(bl)*WINDOW_SECONDS}s)")

    # FP proximity to attacks
    attack_idx = set(np.where(test_tgts == 1)[0])
    fp_idx = np.where(fp_mask)[0]
    near = sum(1 for fi in fp_idx if any((fi-d) in attack_idx or (fi+d) in attack_idx for d in range(1,11)))
    near_pct = near / len(fp_idx) * 100 if len(fp_idx) > 0 else 0
    print(f"  FPs near attacks (<100s): {near:,} ({near_pct:.1f}%)")
    print(f"  FPs far from attacks:     {len(fp_idx)-near:,}")

    fp_analysis[thresh_name.lower()] = {
        'threshold': float(thresh_val), 'total_benign': total_benign,
        'false_positives': int(fp_fp), 'fp_rate': float(fp_fp/total_benign) if total_benign > 0 else 0,
        'fp_near_attack': near, 'fp_far_from_attack': len(fp_idx)-near
    }

pd.DataFrame(fp_analysis).T.to_csv(RESULTS_DIR / 'false_positive_analysis.csv')
print("\\nSaved: false_positive_analysis.csv")""")

# ============================================================
# 2. MODIFY Cell 30 (Section 19 markdown): Label as VALIDATION demo
# ============================================================

old_cell30_src = ''.join(cells[30]['source'])
if 'Genuine autoregressive rollout' in old_cell30_src:
    cells[30]['source'] = fix_source("""---
## Section 19 -- True Recursive K-Step Forecasting

Genuine autoregressive rollout: predict S(t+1), append PREDICTED state
(NOT ground truth), predict S(t+2), etc.

**Note:** The demonstration in this section uses VALIDATION-period data (2018-02-28).
For a TEST set demonstration, see Section 19A below.""")

# ============================================================
# 3. INSERT: After recursive forecasting (cell 33 = risk scoring),
#    add test set recursive demo after Section 20
# ============================================================

# New Section 19A: Test Set Recursive Demonstration
md_19a = fix_source("""---
## Section 19A -- Test Set Recursive Demonstration

Deterministic demonstration using TEST data. Episode selected as the **first
eligible attack episode** in chronological test set order.

An eligible episode: current window is benign, attack occurs within forecast horizon.""")

code_19a = fix_source("""# ============================================================
# SECTION 19A -- Test Set Recursive Demonstration (TEST DATA)
# ============================================================

# Find ALL eligible attack episodes in the test set
eligible_episodes = []
for i in range(len(y_test_raw) - HISTORY - FORECAST_HORIZON):
    if y_test_raw[i + HISTORY - 1] == 0:
        future_attacks = []
        for j in range(FORECAST_HORIZON):
            future_idx = i + HISTORY + j
            if future_idx < len(y_test_raw) and y_test_raw[future_idx] == 1:
                future_attacks.append(j + 1)
        if future_attacks:
            ts_off = HISTORY + i
            base_ts_ep = test_states['Timestamp'].iloc[ts_off] if ts_off < len(test_states) else None
            eligible_episodes.append({
                'seq_idx': i, 'ts_offset': ts_off, 'base_timestamp': base_ts_ep,
                'first_attack_step': future_attacks[0], 'attack_steps': future_attacks
            })

print(f"Total eligible test attack episodes: {len(eligible_episodes)}")

if eligible_episodes:
    # Deterministic selection: FIRST eligible episode
    demo_ep = eligible_episodes[0]
    demo_idx_test = demo_ep['seq_idx']
    base_ts_test = demo_ep['base_timestamp']

    print(f"\\nDemonstration (TEST data, first eligible episode):")
    print(f"  Sequence index: {demo_idx_test}")
    print(f"  Base timestamp: {base_ts_test}")
    print(f"  First attack at: t+{demo_ep['first_attack_step']}")

    fcs_test = recursive_kstep_forecast(model, X_test_seq[demo_idx_test], FORECAST_HORIZON, device)
    fc_probs_test = [fc['attack_prob'] for fc in fcs_test]

    print(f"\\n  {'Step':>5s} {'Timestamp':>22s} {'P(attack)':>10s} {'Risk':>6s} {'Level':>10s} {'Stage':>30s} {'GT':>8s} {'Warning?':>10s}")
    print(f"  {'-'*105}")

    first_warning = None
    first_attack = demo_ep['first_attack_step']

    demo_rows = []
    for fc in fcs_test:
        k = fc['step']
        p = fc['attack_prob']
        rs, rl = compute_risk_score(p)
        stg, ev, cert = infer_attack_stage(fc['state_pred'], state_feature_cols, p, operational_threshold)
        ts_str = str(base_ts_test + pd.Timedelta(seconds=k*WINDOW_SECONDS))[:19] if base_ts_test else ''
        ai = demo_idx_test + HISTORY + k - 1
        gt = 'ATTACK' if (ai < len(y_test_raw) and y_test_raw[ai] == 1) else 'BENIGN'

        if first_warning is None and p >= operational_threshold:
            first_warning = k

        warning = ''
        if k == first_attack:
            if first_warning is not None and first_warning < k:
                warning = f'YES (t+{first_warning})'
            elif first_warning == k:
                warning = 'CONCURRENT'
            else:
                warning = 'NO'

        print(f"  t+{k:2d}  {ts_str:>22s} {p:>10.4f} {rs:>6.0f} {rl:>10s} {stg:>30s} {gt:>8s} {warning:>10s}")
        demo_rows.append({
            'step': f't+{k}', 'timestamp': ts_str, 'predicted_prob': round(p, 4),
            'risk_score': rs, 'risk_level': rl, 'predicted_stage': stg,
            'ground_truth': gt, 'predicted_label': 'ATTACK' if p >= operational_threshold else 'BENIGN'
        })

    warning_preceded = first_warning is not None and first_warning < first_attack
    print(f"\\n  WARNING PRECEDED ATTACK: {'YES' if warning_preceded else 'NO'}")
    if first_warning:
        ewt_demo = (first_attack - first_warning) * WINDOW_SECONDS
        print(f"  Early Warning Time: {ewt_demo}s ({first_attack - first_warning} steps)")
    print(f"  DATA SOURCE: TEST set")
    print(f"  SELECTION: Deterministic (first eligible episode)")

    pd.DataFrame(demo_rows).to_csv(RESULTS_DIR / 'test_demonstration.csv', index=False)
    print("Saved: test_demonstration.csv")""")

# ============================================================
# 4. New Section 23A: Episode-Level Early Warning on Test Set
# ============================================================

md_23a = fix_source("""---
## Section 23A -- Episode-Level Early Warning Metrics (Test Set)

Evaluate early warning performance at both thresholds on the test set.""")

code_23a = fix_source("""# ============================================================
# SECTION 23A -- Episode-Level Early Warning Metrics (TEST SET)
# ============================================================

print("Early Warning Analysis at OPERATIONAL threshold ({:.2f}):".format(operational_threshold))
ew_test_operational = evaluate_early_warning(test_preds, test_tgts, operational_threshold, WINDOW_SECONDS, HISTORY)
print(f"  Total episodes: {ew_test_operational['total_episodes']}")
print(f"  Detected early: {ew_test_operational.get('detected_early', 0)}")
print(f"  Detection rate: {ew_test_operational.get('detection_rate', 0):.4f}")
print(f"  Mean EWT: {ew_test_operational.get('mean_ewt_seconds', 0):.1f}s")
print(f"  Median EWT: {ew_test_operational.get('median_ewt_seconds', 0):.1f}s")
print(f"  Max EWT: {ew_test_operational.get('max_ewt_seconds', 0):.1f}s")

print(f"\\nEarly Warning Analysis at RESEARCH threshold ({research_threshold}):")
ew_test_research = evaluate_early_warning(test_preds, test_tgts, research_threshold, WINDOW_SECONDS, HISTORY)
print(f"  Total episodes: {ew_test_research['total_episodes']}")
print(f"  Detected early: {ew_test_research.get('detected_early', 0)}")
print(f"  Detection rate: {ew_test_research.get('detection_rate', 0):.4f}")
print(f"  Mean EWT: {ew_test_research.get('mean_ewt_seconds', 0):.1f}s")
print(f"  Median EWT: {ew_test_research.get('median_ewt_seconds', 0):.1f}s")
print(f"  Max EWT: {ew_test_research.get('max_ewt_seconds', 0):.1f}s")

ew_combined = {
    'operational': {'threshold': operational_threshold, **ew_test_operational},
    'research': {'threshold': research_threshold, **ew_test_research}
}
with open(RESULTS_DIR / 'early_warning_results.json', 'w') as f:
    json.dump(ew_combined, f, indent=2, default=str)
print("Saved: early_warning_results.json")""")

# ============================================================
# 5. Modify Cell 41 (Model Comparison) to use 3-way comparison
# ============================================================

new_cell41_src = fix_source("""# Already printed in Section 18D. Restate summary:
print("=" * 70)
print("   MODEL COMPARISON SUMMARY (TEST SET)")
print("=" * 70)
print(f"\\n  Logistic Regression (t={lr_metrics.get('threshold', 0.5)}):")
for k in ['precision', 'recall', 'f1', 'roc_auc', 'fpr', 'fnr']:
    print(f"    {k:15s}: {lr_metrics[k]:.4f}")

print(f"\\n  CyberCast @ Research (t={research_threshold}):")
for k in ['precision', 'recall', 'f1', 'roc_auc', 'fpr', 'fnr']:
    print(f"    {k:15s}: {test_research[k]:.4f}")

print(f"\\n  CyberCast @ Operational (t={operational_threshold}):")
for k in ['precision', 'recall', 'f1', 'roc_auc', 'fpr', 'fnr']:
    print(f"    {k:15s}: {test_operational[k]:.4f}")

comp_df = pd.DataFrame({
    'LogisticRegression': lr_metrics,
    'CyberCast_Research': {k:v for k,v in test_research.items()},
    'CyberCast_Operational': {k:v for k,v in test_operational.items()}
})
comp_df.to_csv(RESULTS_DIR / 'metrics.csv')
print("Saved: metrics.csv")""")

cells[41]['source'] = new_cell41_src

# ============================================================
# NOW INSERT ALL NEW CELLS
# ============================================================

# Insert order matters - insert from end to start to preserve indices

# After cell 39 (Section 23 - Early Warning), insert 23A
cells.insert(40, make_md_cell(md_23a))
cells.insert(41, make_code_cell(code_23a))

# After cell 33 (risk scoring), insert 19A (test demo)
# But first we need to find where Section 20 is...
# Cell 33 = risk scoring, cell 34 = attack stage md, cell 35 = attack stage code
# Insert 19A after cell 35 (attack stage)
cells.insert(36, make_md_cell(md_19a))
cells.insert(37, make_code_cell(code_19a))

# After cell 29 (test evaluation plots), insert 18A-18E
# Insert in reverse order so indices remain correct
cells.insert(30, make_md_cell(md_18a))
cells.insert(31, make_code_cell(code_18a))
cells.insert(32, make_md_cell(md_18b))
cells.insert(33, make_code_cell(code_18b))
cells.insert(34, make_md_cell(md_18c))
cells.insert(35, make_code_cell(code_18c))
cells.insert(36, make_md_cell(md_18d))
cells.insert(37, make_code_cell(code_18d))
cells.insert(38, make_md_cell(md_18e))
cells.insert(39, make_code_cell(code_18e))

# Save updated notebook
nb['cells'] = cells
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated notebook: {NB_PATH}")
print(f"Total cells: {len(cells)}")
print("New cells added: 18A (md+code), 18B (md+code), 18C (md+code), 18D (md+code), 18E (md+code), 19A (md+code), 23A (md+code)")
print("Modified cells: Section 19 header (validation label), Section 24 (3-way comparison)")
