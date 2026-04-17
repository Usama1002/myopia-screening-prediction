"""
Published-baseline comparison for Paper 1.

Implements and evaluates:
1. Lin et al. 2018 (PLoS Medicine): Random Forest with (age, SE, annual SE
   progression rate). Predicts high myopia (SE <= -6.0 D). We adapt it to
   our -3.0 D threshold for comparability with our primary outcome.
2. Clinical threshold rule: SE <= -1.5 D at age <=10 flags high risk
3. Non-linear baselines: Random Forest, XGBoost, LightGBM (tree-based)
4. Our proposed T2 model (penalised logistic regression) and T4 (with biometrics)

MLPs were excluded from the non-linear comparison set because sample-size
guidance for feed-forward neural networks on tabular data recommends at
least 10-50 samples per free parameter, which exceeds our 1-year high
myopia cohort (n=1,039, 57 events). Tree-based methods remain the
appropriate non-linear comparison class for this cohort size.

All models evaluated with the same patient-grouped 5-fold CV and
patient-clustered bootstrap CI.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from prepare import prepare_all, MYOPIA_THRESHOLD

warnings.filterwarnings('ignore')

OUT_DIR = Path('../results')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 2024]
N_BOOTSTRAP = 2000


# ============================================================================
# COHORT BUILDING
# ============================================================================
def build_high_myopia_cohort(clean, horizon=1):
    """Primary cohort: predict SE <= -3.0 D within `horizon` years."""
    clean = clean.sort_values(['pid', 'year']).reset_index(drop=True)
    samples = []
    for pid, g in clean.groupby('pid'):
        g = g.sort_values('year').reset_index(drop=True)
        for i in range(len(g)):
            b = g.iloc[i].copy()
            if b['SE_OD'] <= -3.0:
                continue
            future = g.iloc[i + 1:]
            future = future[(future['year'] - b['year']) <= horizon]
            if len(future) == 0:
                continue
            # Lin et al. feature: annual progression rate from prior visit
            if i == 0:
                b['prog_rate'] = 0.0
                b['has_prior'] = 0
            else:
                prior = g.iloc[i - 1]
                dt = max(b['age'] - prior['age'], 0.5)
                b['prog_rate'] = (b['SE_OD'] - prior['SE_OD']) / dt
                b['has_prior'] = 1
            label = (future['SE_OD'] <= -3.0).any()
            b['label'] = int(label)
            samples.append(b)
    return pd.DataFrame(samples).reset_index(drop=True)


# ============================================================================
# MODELS: published baselines + modern ML
# ============================================================================
def lin_et_al_2018(seed):
    """Lin et al. 2018: Random Forest with age, SE, progression rate.

    Reference: Lin H et al. (2018) PLoS Medicine 15(11):e1002674.
    Original model: Random forest predicting high myopia (SE <= -6.0 D)
    over 3/5/8 year horizons using age, SE, and annual progression rate.
    We use the same features but on our local dataset and -3.0 D threshold.
    """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=4,
        random_state=seed,
        class_weight='balanced',
    )


def lin_et_al_features():
    return ['age', 'SE_OD', 'prog_rate']


def clinical_threshold_rule(X, feats):
    """Simple rule: SE <= -1.5 D at age <= 10 flags high risk.

    Score is a smoothed linear combination of SE (negative means risk) and
    a young-age bonus. Higher score = higher risk.
    """
    age = X[:, feats.index('age')]
    se = X[:, feats.index('SE_OD')]
    # Score: negative SE gives higher risk, and young age multiplies risk
    score = -se * (1 + 0.1 * np.maximum(0, 12 - age))
    return score


def proposed_t2(seed):
    return Pipeline([
        ('s', StandardScaler()),
        ('c', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                   class_weight='balanced', random_state=seed))
    ])


def proposed_t4(seed):
    return proposed_t2(seed)  # same model, different feature set


def rf_full(seed):
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        n_jobs=4, random_state=seed, class_weight='balanced')


def xgb_full(seed):
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=4, verbosity=0, eval_metric='auc',
        scale_pos_weight=5.0)  # account for ~10% positive rate


def lgb_full(seed):
    return LGBMClassifier(
        n_estimators=200, num_leaves=15, max_depth=4,
        learning_rate=0.05, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=4, verbosity=-1,
        class_weight='balanced')


T1_feats = ['age', 'gender_M']
T2_feats = ['age', 'gender_M', 'SE_OD', 'SE_OS']
T3_feats = ['age', 'gender_M', 'SE_OD', 'SE_OS',
             'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
             'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS', 'aniso_SE']
T4_feats = T3_feats + ['pupil_OD', 'pupil_OS', 'IPD',
                         'gaze_mag_OD', 'gaze_mag_OS', 'pupil_asym']


# ============================================================================
# CV EVALUATION
# ============================================================================
def cv_pooled(cohort, feats, model_fn, seed=42, use_proba=True):
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    out_idx, out_y, out_p = [], [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xtr = cohort.iloc[tr][feats].fillna(med).values
        Xte = cohort.iloc[te][feats].fillna(med).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m = model_fn(seed)
        m.fit(Xtr, ytr)
        out_idx.extend(te)
        out_y.extend(yte)
        if use_proba:
            out_p.extend(m.predict_proba(Xte)[:, 1])
        else:
            out_p.extend(m.predict(Xte))
    return np.array(out_idx), np.array(out_y), np.array(out_p)


def cv_pooled_rule(cohort, feats, rule_fn, seed=42):
    """For non-ML clinical rules that just compute a score."""
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    out_idx, out_y, out_p = [], [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xte = cohort.iloc[te][feats].fillna(med).values
        yte = cohort.iloc[te]['label'].values
        score = rule_fn(Xte, feats)
        out_idx.extend(te)
        out_y.extend(yte)
        out_p.extend(score)
    return np.array(out_idx), np.array(out_y), np.array(out_p)


def multi_seed_auc(cohort, feats, model_fn, is_rule=False):
    aucs = []
    for s in SEEDS:
        if is_rule:
            _, y, p = cv_pooled_rule(cohort, feats, model_fn, seed=s)
        else:
            _, y, p = cv_pooled(cohort, feats, model_fn, seed=s)
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.std(aucs)), [float(a) for a in aucs]


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


# ============================================================================
# MAIN
# ============================================================================
def main():
    print('=' * 70)
    print('PUBLISHED BASELINE COMPARISON: Paper 1')
    print('=' * 70)

    clean, _, _ = prepare_all()

    results = {'tasks': {}}

    for horizon in [1, 2, 3]:
        task_name = f'high_myopia_{horizon}yr'
        print(f'\n{"=" * 70}\n{task_name}\n{"=" * 70}')

        cohort = build_high_myopia_cohort(clean, horizon=horizon)
        print(f'n={len(cohort)}, events={cohort["label"].sum()}, rate={cohort["label"].mean():.3f}')

        task_res = {
            'n': int(len(cohort)),
            'events': int(cohort['label'].sum()),
            'rate': float(cohort['label'].mean()),
            'models': {},
        }

        # Store predictions for delta comparisons
        cached = {}

        # 1. Clinical threshold rule (simplest baseline)
        print('\n-- Clinical threshold rule (SE and age) --')
        m, s, per = multi_seed_auc(cohort, T2_feats, clinical_threshold_rule, is_rule=True)
        idx, y, p = cv_pooled_rule(cohort, T2_feats, clinical_threshold_rule, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        cached['Clinical_rule'] = (idx, y, p)
        task_res['models']['Clinical_rule'] = {
            'description': 'Clinical threshold: -SE * (1 + 0.1*max(0, 12-age))',
            'features': T2_feats,
            'multi_seed_mean': m, 'multi_seed_std': s,
            'bootstrap_auc': boot,
        }
        print(f'  AUC = {m:.4f} +/- {s:.4f}, bootstrap CI [{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        # 2. Lin et al. 2018 baseline
        print('\n-- Lin et al. 2018 (Random Forest, age+SE+progression rate) --')
        m, s, per = multi_seed_auc(cohort, lin_et_al_features(), lin_et_al_2018)
        idx, y, p = cv_pooled(cohort, lin_et_al_features(), lin_et_al_2018, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        cached['Lin2018'] = (idx, y, p)
        task_res['models']['Lin2018'] = {
            'description': 'Lin et al. 2018 PLoS Medicine: Random Forest with age, SE, annual progression rate',
            'features': lin_et_al_features(),
            'multi_seed_mean': m, 'multi_seed_std': s,
            'bootstrap_auc': boot,
        }
        print(f'  AUC = {m:.4f} +/- {s:.4f}, bootstrap CI [{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        # 3. Proposed T2 (age + SE both eyes, logistic regression)
        print('\n-- Proposed T2: age + SE both eyes, L2 logistic regression --')
        m, s, per = multi_seed_auc(cohort, T2_feats, proposed_t2)
        idx, y, p = cv_pooled(cohort, T2_feats, proposed_t2, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        cached['Proposed_T2'] = (idx, y, p)
        task_res['models']['Proposed_T2'] = {
            'description': 'Penalised logistic regression with age, gender, SE (both eyes)',
            'features': T2_feats,
            'multi_seed_mean': m, 'multi_seed_std': s,
            'bootstrap_auc': boot,
        }
        print(f'  AUC = {m:.4f} +/- {s:.4f}, bootstrap CI [{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        # 4. Proposed T4 (+ biometric features)
        print('\n-- Proposed T4: + power vectors + biometric features --')
        m, s, per = multi_seed_auc(cohort, T4_feats, proposed_t4)
        idx, y, p = cv_pooled(cohort, T4_feats, proposed_t4, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        cached['Proposed_T4'] = (idx, y, p)
        task_res['models']['Proposed_T4'] = {
            'description': 'Penalised logistic regression with T4 feature set',
            'features': T4_feats,
            'multi_seed_mean': m, 'multi_seed_std': s,
            'bootstrap_auc': boot,
        }
        print(f'  AUC = {m:.4f} +/- {s:.4f}, bootstrap CI [{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        # 5. Random Forest (full T4 features, capacity check)
        print('\n-- Random Forest (full T4 features) --')
        m, s, per = multi_seed_auc(cohort, T4_feats, rf_full)
        idx, y, p = cv_pooled(cohort, T4_feats, rf_full, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        cached['RF_T4'] = (idx, y, p)
        task_res['models']['RF_T4'] = {
            'description': 'Random Forest (300 trees, depth 6) with T4 feature set',
            'features': T4_feats,
            'multi_seed_mean': m, 'multi_seed_std': s,
            'bootstrap_auc': boot,
        }
        print(f'  AUC = {m:.4f} +/- {s:.4f}, bootstrap CI [{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        # 6. XGBoost
        print('\n-- XGBoost (full T4 features) --')
        m, s, per = multi_seed_auc(cohort, T4_feats, xgb_full)
        idx, y, p = cv_pooled(cohort, T4_feats, xgb_full, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        cached['XGB_T4'] = (idx, y, p)
        task_res['models']['XGB_T4'] = {
            'description': 'XGBoost (200 trees, depth 4) with T4 feature set',
            'features': T4_feats,
            'multi_seed_mean': m, 'multi_seed_std': s,
            'bootstrap_auc': boot,
        }
        print(f'  AUC = {m:.4f} +/- {s:.4f}, bootstrap CI [{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        # 7. LightGBM
        print('\n-- LightGBM (full T4 features) --')
        m, s, per = multi_seed_auc(cohort, T4_feats, lgb_full)
        idx, y, p = cv_pooled(cohort, T4_feats, lgb_full, seed=42)
        pids = cohort.iloc[idx]['pid'].values
        boot = fast_patient_bootstrap_auc(pids, y, p)
        cached['LGB_T4'] = (idx, y, p)
        task_res['models']['LGB_T4'] = {
            'description': 'LightGBM (200 trees, depth 4) with T4 feature set',
            'features': T4_feats,
            'multi_seed_mean': m, 'multi_seed_std': s,
            'bootstrap_auc': boot,
        }
        print(f'  AUC = {m:.4f} +/- {s:.4f}, bootstrap CI [{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        # Pairwise delta vs Proposed T2 (the headline model)
        print('\n-- Deltas vs Proposed T2 (the proposed headline model) --')
        ref_idx, ref_y, ref_p = cached['Proposed_T2']
        ref_pids = cohort.iloc[ref_idx]['pid'].values
        deltas = {}
        for model_name, (idx, y, p) in cached.items():
            if model_name == 'Proposed_T2':
                continue
            # We need the prediction in the same fold order - but every model used
            # seed=42 and the same splitter, so the index order matches per-fold.
            # For patient-clustered bootstrap, we align by cohort index.
            d = fast_patient_bootstrap_delta(ref_pids, ref_y, ref_p, p)
            deltas[f'{model_name}_vs_T2'] = d
            print(f'  Delta {model_name} vs Proposed T2: '
                  f'{d["mean"]:+.4f} [{d["lo"]:+.4f}, {d["hi"]:+.4f}], '
                  f'P(>0)={d["p_positive"]:.3f}')
        task_res['deltas_vs_proposed_T2'] = deltas

        results['tasks'][task_name] = task_res

    # Save
    with open(OUT_DIR / 'baselines_results.json', 'w') as f:
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
        json.dump(to_serializable(results), f, indent=2)

    print(f'\nResults saved to {OUT_DIR}/baselines_results.json')


if __name__ == '__main__':
    main()
