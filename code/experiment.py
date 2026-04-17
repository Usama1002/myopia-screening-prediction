"""
Paper 1 Final Experiment Pipeline (v2)
======================================
Multi-outcome paediatric myopia risk stratification framework.

PRIMARY HYPOTHESIS: A simple, calibrated logistic regression model trained
on routinely captured screening data can stratify children by risk of
clinically distinct myopia outcomes (onset, mild-to-moderate progression,
sight-threatening high myopia development) with deployment-grade accuracy.

SECONDARY HYPOTHESIS: Automated screening device biometric features (pupil
diameter, interpupil distance, gaze deviation) carry incremental predictive
value beyond standard refraction.

Tasks:
1. Sight-threatening (high) myopia development at 1, 2, 3-year horizons
2. Mild-to-moderate progression (already mildly myopic, becomes >= 3.0 D)
3. Myopia onset (non-myopic, becomes myopic at 1 year)
4. Rapid progression (myopic, rate <= -0.5 D/year)

For each task we evaluate:
- Multi-seed CV AUC with patient-clustered bootstrap CI
- Calibration (slope, intercept, mean abs cal error)
- Decision curve analysis (net benefit at multiple thresholds)
- NRI (T2 to T4 incremental value)
- Subgroup analyses by age and baseline SE

Models: penalised logistic regression (primary), gradient boosting (sensitivity).
Statistical inference: patient-clustered bootstrap (2000 iterations) for both
absolute AUC CIs and paired delta AUC CIs.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from prepare import (
    prepare_all, MYOPIA_THRESHOLD,
    calibration_metrics, decision_curve_net_benefit,
    net_reclassification_improvement,
)

warnings.filterwarnings('ignore')

OUT_DIR = Path('../results')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 2024]
N_BOOTSTRAP = 2000

# Feature tier definitions
T1 = ['age', 'gender_M']
T2 = ['age', 'gender_M', 'SE_OD', 'SE_OS']
T3 = ['age', 'gender_M', 'SE_OD', 'SE_OS',
      'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
      'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS', 'aniso_SE']
T4 = T3 + ['pupil_OD', 'pupil_OS', 'IPD',
            'gaze_mag_OD', 'gaze_mag_OS', 'pupil_asym']

TIERS = {'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4}


# ============================================================================
# COHORT BUILDERS
# ============================================================================
def build_pairs(clean):
    pairs = []
    for pid, group in clean.sort_values(['pid', 'year']).groupby('pid'):
        group = group.sort_values('year').reset_index(drop=True)
        for i in range(len(group) - 1):
            b = group.iloc[i].copy()
            f = group.iloc[i + 1]
            b['SE_OD_next'] = f['SE_OD']
            b['delta_t'] = f['year'] - b['year']
            pairs.append(b)
    return pd.DataFrame(pairs).reset_index(drop=True)


def build_high_myopia_cohort(clean, horizon):
    """Sight-threatening myopia development within `horizon` years."""
    samples = []
    for pid, g in clean.sort_values(['pid', 'year']).groupby('pid'):
        g = g.sort_values('year').reset_index(drop=True)
        for i in range(len(g)):
            b = g.iloc[i]
            if b['SE_OD'] <= -3.0:
                continue
            future = g.iloc[i + 1:]
            future = future[(future['year'] - b['year']) <= horizon]
            if len(future) == 0:
                continue
            label = (future['SE_OD'] <= -3.0).any()
            s = b.copy()
            s['label'] = int(label)
            s['horizon'] = horizon
            samples.append(s)
    return pd.DataFrame(samples).reset_index(drop=True)


def build_onset_cohort(pairs):
    c = pairs[pairs['SE_OD'] > MYOPIA_THRESHOLD].copy()
    c['label'] = (c['SE_OD_next'] <= MYOPIA_THRESHOLD).astype(int)
    return c.reset_index(drop=True)


def build_rapid_cohort(pairs):
    c = pairs[pairs['SE_OD'] <= MYOPIA_THRESHOLD].copy()
    c['rate'] = (c['SE_OD_next'] - c['SE_OD']) / c['delta_t']
    c['label'] = (c['rate'] <= -0.5).astype(int)
    return c.reset_index(drop=True)


def build_mild_to_moderate_cohort(pairs):
    """Already mildly myopic, becomes moderate (-3.0 D) within follow-up."""
    c = pairs[(pairs['SE_OD'] <= MYOPIA_THRESHOLD) & (pairs['SE_OD'] > -3.0)].copy()
    c['label'] = (c['SE_OD_next'] <= -3.0).astype(int)
    return c.reset_index(drop=True)


# ============================================================================
# CV AND BOOTSTRAP
# ============================================================================
def make_lr(seed):
    return Pipeline([
        ('s', StandardScaler()),
        ('c', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                   class_weight='balanced', random_state=seed))
    ])


def cv_pooled(cohort, feats, seed=42):
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    out_idx, out_y, out_p = [], [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xtr = cohort.iloc[tr][feats].fillna(med).values
        Xte = cohort.iloc[te][feats].fillna(med).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m = make_lr(seed).fit(Xtr, ytr)
        out_idx.extend(te)
        out_y.extend(yte)
        out_p.extend(m.predict_proba(Xte)[:, 1])
    return np.array(out_idx), np.array(out_y), np.array(out_p)


def fast_patient_bootstrap_auc(pids, y, p, n_iter=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    pid_to_indices = {}
    for i, pid in enumerate(pids):
        pid_to_indices.setdefault(pid, []).append(i)
    unique_pids = list(pid_to_indices.keys())
    pid_arrays = [np.array(pid_to_indices[pid]) for pid in unique_pids]
    n_pids = len(unique_pids)
    aucs = []
    for _ in range(n_iter):
        sampled = rng.randint(0, n_pids, size=n_pids)
        idx = np.concatenate([pid_arrays[i] for i in sampled])
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(ys, p[idx]))
        except Exception:
            continue
    return {
        'mean': float(np.mean(aucs)),
        'lo': float(np.percentile(aucs, 2.5)),
        'hi': float(np.percentile(aucs, 97.5)),
    }


def fast_patient_bootstrap_delta(pids, y, p_a, p_b, n_iter=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    pid_to_indices = {}
    for i, pid in enumerate(pids):
        pid_to_indices.setdefault(pid, []).append(i)
    unique_pids = list(pid_to_indices.keys())
    pid_arrays = [np.array(pid_to_indices[pid]) for pid in unique_pids]
    n_pids = len(unique_pids)
    deltas = []
    for _ in range(n_iter):
        sampled = rng.randint(0, n_pids, size=n_pids)
        idx = np.concatenate([pid_arrays[i] for i in sampled])
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            d = roc_auc_score(ys, p_b[idx]) - roc_auc_score(ys, p_a[idx])
            deltas.append(d)
        except Exception:
            continue
    return {
        'mean': float(np.mean(deltas)),
        'lo': float(np.percentile(deltas, 2.5)),
        'hi': float(np.percentile(deltas, 97.5)),
        'p_positive': float((np.array(deltas) > 0).mean()),
    }


def multi_seed_auc(cohort, feats):
    aucs = []
    for s in SEEDS:
        _, y, p = cv_pooled(cohort, feats, seed=s)
        aucs.append(roc_auc_score(y, p))
    return {
        'mean': float(np.mean(aucs)),
        'std': float(np.std(aucs)),
        'per_seed': [float(a) for a in aucs],
    }


# ============================================================================
# COMPLETE TASK EVALUATION
# ============================================================================
def evaluate_task(cohort, task_name):
    print(f'\n=== {task_name} ===')
    print(f'  n={len(cohort)}, events={cohort["label"].sum()}, rate={cohort["label"].mean():.3f}')

    task_results = {
        'name': task_name,
        'n': int(len(cohort)),
        'patients': int(cohort['pid'].nunique()),
        'events': int(cohort['label'].sum()),
        'rate': float(cohort['label'].mean()),
        'tiers': {},
    }

    # Per-tier evaluation
    cached_preds = {}
    for tier_name, fs in TIERS.items():
        # Multi-seed AUC
        ms = multi_seed_auc(cohort, fs)
        # Patient-clustered bootstrap
        idx, y, p = cv_pooled(cohort, fs, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        # Calibration
        cal = calibration_metrics(y, p, n_bins=10)
        # Store predictions for delta computations
        cached_preds[tier_name] = (idx, y, p, pids)
        task_results['tiers'][tier_name] = {
            'multi_seed': ms,
            'bootstrap_auc': boot,
            'calibration': cal,
        }
        print(f'  {tier_name}: AUC={ms["mean"]:.4f} ± {ms["std"]:.4f}  '
              f'[boot CI {boot["lo"]:.3f}, {boot["hi"]:.3f}]  '
              f'cal_slope={cal["cal_slope"]:.2f}')

    # Pairwise deltas
    deltas = {}
    pids = cached_preds['T2'][3]
    p_t2 = cached_preds['T2'][2]
    y_t2 = cached_preds['T2'][1]
    for ref_name, ref_data in cached_preds.items():
        if ref_name in ('T1', 'T2'):
            continue
        d = fast_patient_bootstrap_delta(pids, y_t2, p_t2, ref_data[2])
        deltas[f'{ref_name}_vs_T2'] = d
        print(f'  Delta {ref_name} vs T2: {d["mean"]:+.4f} '
              f'[{d["lo"]:+.4f}, {d["hi"]:+.4f}], P(>0)={d["p_positive"]:.3f}')
    task_results['deltas'] = deltas

    # Decision curve and NRI for T2 (the main clinical model)
    idx_t2, y_t2, p_t2 = cached_preds['T2'][:3]
    idx_t4, y_t4, p_t4 = cached_preds['T4'][:3]
    dca_t2 = decision_curve_net_benefit(y_t2, p_t2,
                                          thresholds=np.arange(0.05, 0.85, 0.05))
    dca_t4 = decision_curve_net_benefit(y_t4, p_t4,
                                          thresholds=np.arange(0.05, 0.85, 0.05))
    task_results['dca'] = {'T2': dca_t2, 'T4': dca_t4}

    # NRI (only when comparable predictions are useful)
    rate = cohort['label'].mean()
    lo_thresh = max(0.05, rate * 0.5)
    hi_thresh = min(0.95, rate * 2)
    try:
        nri = net_reclassification_improvement(
            y_t2, p_t2, p_t4, low_threshold=lo_thresh, high_threshold=hi_thresh)
        task_results['nri'] = float(nri)
    except Exception:
        task_results['nri'] = None

    return task_results, cached_preds


# ============================================================================
# SUBGROUP ANALYSIS
# ============================================================================
def evaluate_subgroups(cohort, group_var, bins, labels):
    cohort = cohort.copy()
    cohort['_grp'] = pd.cut(cohort[group_var], bins=bins, labels=labels, include_lowest=True)
    out = []
    for label in labels:
        sub = cohort[cohort['_grp'] == label].reset_index(drop=True)
        if len(sub) < 50 or sub['label'].sum() < 5 or (sub['label'] == 0).sum() < 5:
            out.append({'group': str(label), 'n': int(len(sub)), 'skipped': True})
            continue
        try:
            t2 = multi_seed_auc(sub, T2)
            t4 = multi_seed_auc(sub, T4)
            idx, y, p = cv_pooled(sub, T2, seed=42)
            pids = sub.iloc[idx]['pid'].values
            boot_t2 = fast_patient_bootstrap_auc(pids, y, p)
            idx4, y4, p4 = cv_pooled(sub, T4, seed=42)
            pids4 = sub.iloc[idx4]['pid'].values
            boot_t4 = fast_patient_bootstrap_auc(pids4, y4, p4)
            out.append({
                'group': str(label),
                'n': int(len(sub)),
                'events': int(sub['label'].sum()),
                't2_auc_mean': t2['mean'],
                't2_boot_lo': boot_t2['lo'],
                't2_boot_hi': boot_t2['hi'],
                't4_auc_mean': t4['mean'],
                't4_boot_lo': boot_t4['lo'],
                't4_boot_hi': boot_t4['hi'],
            })
        except Exception as e:
            out.append({'group': str(label), 'n': int(len(sub)), 'error': str(e)})
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    print('=' * 70)
    print('Paper 1 Final Pipeline (v2): Multi-Outcome Myopia Risk Stratification')
    print('=' * 70)

    clean, _, _ = prepare_all()
    print(f'\nclean records: {len(clean)}')
    print(f'unique patients: {clean["pid"].nunique()}')
    print(f'multi-year patients: {(clean.groupby("pid").size() >= 2).sum()}')

    pairs = build_pairs(clean)
    print(f'longitudinal pairs: {len(pairs)}')

    all_results = {
        'cohort_summary': {
            'records': int(len(clean)),
            'patients': int(clean['pid'].nunique()),
            'multi_year_patients': int((clean.groupby('pid').size() >= 2).sum()),
            'longitudinal_pairs': int(len(pairs)),
        },
        'tasks': {}
    }
    cached_all = {}

    # === PRIMARY TASKS: HIGH MYOPIA at multiple horizons ===
    for h in [1, 2, 3]:
        cohort = build_high_myopia_cohort(clean, h)
        res, preds = evaluate_task(cohort, f'high_myopia_{h}yr')
        all_results['tasks'][f'high_myopia_{h}yr'] = res
        cached_all[f'high_myopia_{h}yr'] = (cohort, preds)

    # === SECONDARY TASKS ===
    onset = build_onset_cohort(pairs)
    res, preds = evaluate_task(onset, 'onset')
    all_results['tasks']['onset'] = res
    cached_all['onset'] = (onset, preds)

    m2m = build_mild_to_moderate_cohort(pairs)
    res, preds = evaluate_task(m2m, 'mild_to_moderate')
    all_results['tasks']['mild_to_moderate'] = res
    cached_all['mild_to_moderate'] = (m2m, preds)

    rapid = build_rapid_cohort(pairs)
    res, preds = evaluate_task(rapid, 'rapid_progression')
    all_results['tasks']['rapid_progression'] = res
    cached_all['rapid_progression'] = (rapid, preds)

    # === SUBGROUP ANALYSES on the headline (high myopia 2-year) ===
    print('\n=== SUBGROUP: high myopia 2-yr by age ===')
    cohort_hm2 = cached_all['high_myopia_2yr'][0]
    by_age = evaluate_subgroups(cohort_hm2, 'age',
                                  [0, 8, 11, 20], ['under_8', '8_to_10', '11_plus'])
    for sg in by_age:
        print(f'  {sg}')
    all_results['high_myopia_2yr_by_age'] = by_age

    print('\n=== SUBGROUP: high myopia 2-yr by baseline SE ===')
    by_se = evaluate_subgroups(cohort_hm2, 'SE_OD',
                                 [-3.0, -1.0, 0.0, 1.0, 5.0],
                                 ['mild_myopic', 'borderline', 'low_hyperopic', 'high_hyperopic'])
    for sg in by_se:
        print(f'  {sg}')
    all_results['high_myopia_2yr_by_se'] = by_se

    # === Save predictions for figure generation (all 4 tiers) ===
    pred_out = {}
    for task_name, (cohort, preds) in cached_all.items():
        for tier_name in ['T1', 'T2', 'T3', 'T4']:
            idx, y, p, _ = preds[tier_name]
            pred_out[f'{task_name}_{tier_name}_idx'] = idx
            pred_out[f'{task_name}_{tier_name}_y'] = y
            pred_out[f'{task_name}_{tier_name}_p'] = p
    np.savez(OUT_DIR / 'predictions_v2.npz', **pred_out)

    # === Save full results ===
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return obj

    with open(OUT_DIR / 'results_v2.json', 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    print(f'\nResults saved to {OUT_DIR}/')

    # Primary metric for the loop tracker
    primary = all_results['tasks']['high_myopia_2yr']['tiers']['T2']['multi_seed']['mean']
    print(f'\nprimary_metric: {primary:.4f}')
    print(f'primary_auc: {primary:.4f}')


if __name__ == '__main__':
    main()
