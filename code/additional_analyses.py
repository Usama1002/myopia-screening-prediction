"""
Additional analyses requested by the adversarial review:

1. Temporal split validation (train 2018-2019 baselines, test 2020-2021 baselines)
2. DeLong test p-values for key baseline comparisons
3. Leave-one-biometric-feature-out ablation
4. Feature correlation heatmap for the T4 feature set
5. Holm-Bonferroni multiple-testing correction for pairwise tier comparisons

All analyses use the same cohort-construction and prediction machinery as
experiment.py. Outputs are saved to ../results/additional_results.json and
../results/figures (correlation heatmap, leave-one-out plot).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from prepare import prepare_all, delong_test
from experiment import (
    build_pairs, build_high_myopia_cohort, build_onset_cohort,
    build_rapid_cohort, build_mild_to_moderate_cohort,
    T1, T2, T3, T4, make_lr,
)

warnings.filterwarnings('ignore')

OUT_DIR = Path('../results')
FIG_DIR = Path(__file__).resolve().parent.parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 2000
SEEDS = [42, 123, 456, 789, 2024]


# ============================================================================
# Shared CV helper
# ============================================================================
def cv_pooled(cohort, feats, seed=42):
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    all_idx, all_y, all_p = [], [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xtr = cohort.iloc[tr][feats].fillna(med).values
        Xte = cohort.iloc[te][feats].fillna(med).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m = make_lr(seed).fit(Xtr, ytr)
        all_idx.extend(te)
        all_y.extend(yte)
        all_p.extend(m.predict_proba(Xte)[:, 1])
    return np.array(all_idx), np.array(all_y), np.array(all_p)


def patient_bootstrap_auc(pids, y, p, n_iter=N_BOOTSTRAP, seed=42):
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


# ============================================================================
# 1. Temporal split validation (train on early years, test on late years)
# ============================================================================
def temporal_split_evaluation(clean):
    """Train on 2018+2019 baselines, test on 2020+2021 baselines."""
    print('\n' + '=' * 60)
    print('1. TEMPORAL SPLIT VALIDATION')
    print('=' * 60)

    results = {}

    for horizon in [1, 2, 3]:
        cohort = build_high_myopia_cohort(clean, horizon)
        # Split by baseline year
        train_mask = cohort['year'].isin([2018, 2019])
        test_mask = cohort['year'].isin([2020, 2021])
        train = cohort[train_mask].reset_index(drop=True)
        test = cohort[test_mask].reset_index(drop=True)

        if len(test) < 30 or test['label'].sum() < 5:
            print(f'  {horizon}-year: test set too small (n={len(test)}, '
                  f'events={test["label"].sum()}) — SKIPPED')
            continue

        print(f'\n--- high_myopia_{horizon}yr temporal split ---')
        print(f'  Train: n={len(train)}, events={train["label"].sum()} '
              f'({train["label"].mean():.3f})')
        print(f'  Test:  n={len(test)}, events={test["label"].sum()} '
              f'({test["label"].mean():.3f})')

        task_res = {
            'train_n': int(len(train)),
            'train_events': int(train['label'].sum()),
            'train_rate': float(train['label'].mean()),
            'test_n': int(len(test)),
            'test_events': int(test['label'].sum()),
            'test_rate': float(test['label'].mean()),
            'tiers': {},
        }

        for tier_name, feats in [('T2', T2), ('T4', T4)]:
            med = train[feats].median()
            Xtr = train[feats].fillna(med).values
            Xte = test[feats].fillna(med).values
            ytr = train['label'].values
            yte = test['label'].values

            model = make_lr(42).fit(Xtr, ytr)
            p_test = model.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, p_test)

            # Patient-clustered bootstrap on the temporal test set
            pids = test['pid'].values
            boot = patient_bootstrap_auc(pids, yte, p_test)

            task_res['tiers'][tier_name] = {
                'test_auc': float(auc),
                'bootstrap': boot,
            }
            print(f'  {tier_name} temporal test AUC = {auc:.4f} '
                  f'[{boot["lo"]:.3f}, {boot["hi"]:.3f}]')

        results[f'high_myopia_{horizon}yr'] = task_res

    return results


# ============================================================================
# 2. DeLong test p-values for key comparisons
# ============================================================================
def delong_tests(clean, pairs):
    """Compute DeLong p-values for T2 vs T1/T3/T4 and for T2 vs baselines."""
    print('\n' + '=' * 60)
    print('2. DELONG TEST P-VALUES')
    print('=' * 60)

    from prepare import MYOPIA_THRESHOLD

    # Build Lin et al. 2018 cohort with progression rate feature
    def build_hm_with_prograte(clean, horizon):
        clean_sorted = clean.sort_values(['pid', 'year']).reset_index(drop=True)
        samples = []
        for pid, g in clean_sorted.groupby('pid'):
            g = g.sort_values('year').reset_index(drop=True)
            for i in range(len(g)):
                b = g.iloc[i].copy()
                if b['SE_OD'] <= -3.0:
                    continue
                future = g.iloc[i + 1:]
                future = future[(future['year'] - b['year']) <= horizon]
                if len(future) == 0:
                    continue
                if i == 0:
                    b['prog_rate'] = 0.0
                else:
                    prior = g.iloc[i - 1]
                    dt = max(b['age'] - prior['age'], 0.5)
                    b['prog_rate'] = (b['SE_OD'] - prior['SE_OD']) / dt
                label = (future['SE_OD'] <= -3.0).any()
                b['label'] = int(label)
                samples.append(b)
        return pd.DataFrame(samples).reset_index(drop=True)

    from sklearn.ensemble import RandomForestClassifier

    def lin2018_classifier(seed):
        return RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            max_features='sqrt', n_jobs=4, random_state=seed,
            class_weight='balanced')

    results = {}
    lin_feats = ['age', 'SE_OD', 'prog_rate']

    for horizon in [1, 2, 3]:
        cohort = build_hm_with_prograte(clean, horizon)
        task_name = f'high_myopia_{horizon}yr'
        print(f'\n--- {task_name} DeLong tests ---')

        # Proposed T2 predictions
        _, y_t2, p_t2 = cv_pooled(cohort, T2, seed=42)

        # Lin et al. 2018 predictions (Random Forest with age+SE+prograte)
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
        all_y_lin, all_p_lin = [], []
        for tr, te in splits:
            med = cohort[lin_feats].median()
            Xtr = cohort.iloc[tr][lin_feats].fillna(med).values
            Xte = cohort.iloc[te][lin_feats].fillna(med).values
            ytr = cohort.iloc[tr]['label'].values
            yte = cohort.iloc[te]['label'].values
            m = lin2018_classifier(42).fit(Xtr, ytr)
            all_y_lin.extend(yte)
            all_p_lin.extend(m.predict_proba(Xte)[:, 1])
        y_lin = np.array(all_y_lin)
        p_lin = np.array(all_p_lin)

        # DeLong: T2 vs Lin. We need paired data on the same test instances.
        # The CV folds differ slightly between cohorts because Lin cohort has
        # the same structure as T2 cohort (same inclusion criteria), and we
        # used the same seed=42. The order is deterministic, so the predictions
        # are aligned.
        if len(y_t2) == len(y_lin):
            z, p = delong_test(y_t2, p_lin, p_t2)
            results[f'{task_name}_T2_vs_Lin2018'] = {
                'z': float(z), 'p': float(p),
                'auc_T2': float(roc_auc_score(y_t2, p_t2)),
                'auc_Lin2018': float(roc_auc_score(y_lin, p_lin)),
            }
            print(f'  T2 vs Lin2018: DeLong z={z:.3f}, p={p:.4f} '
                  f'(AUC T2={roc_auc_score(y_t2, p_t2):.4f}, '
                  f'Lin={roc_auc_score(y_lin, p_lin):.4f})')

        # T4 vs T2 (on T2 cohort, no prog_rate needed)
        from experiment import build_high_myopia_cohort as build_hm_std
        cohort_std = build_hm_std(clean, horizon)
        _, y2, p2 = cv_pooled(cohort_std, T2, seed=42)
        _, y4, p4 = cv_pooled(cohort_std, T4, seed=42)
        z, p = delong_test(y2, p2, p4)
        results[f'{task_name}_T4_vs_T2'] = {
            'z': float(z), 'p': float(p),
            'auc_T2': float(roc_auc_score(y2, p2)),
            'auc_T4': float(roc_auc_score(y4, p4)),
        }
        print(f'  T4 vs T2: DeLong z={z:.3f}, p={p:.4f}')

        # T3 vs T2
        _, y3, p3 = cv_pooled(cohort_std, T3, seed=42)
        z, p = delong_test(y2, p2, p3)
        results[f'{task_name}_T3_vs_T2'] = {
            'z': float(z), 'p': float(p),
            'auc_T2': float(roc_auc_score(y2, p2)),
            'auc_T3': float(roc_auc_score(y3, p3)),
        }
        print(f'  T3 vs T2: DeLong z={z:.3f}, p={p:.4f}')

    return results


# ============================================================================
# 3. Holm-Bonferroni correction for multiple-testing family
# ============================================================================
def holm_bonferroni(delong_results, family='primary'):
    """Apply Holm-Bonferroni to a family of p-values."""
    print('\n' + '=' * 60)
    print('3. HOLM-BONFERRONI CORRECTION')
    print('=' * 60)

    if family == 'primary':
        # Family: all pairwise tier comparisons on the 3 high myopia horizons
        # That is: T3 vs T2, T4 vs T2, T2 vs Lin2018 at 1/2/3-year = 9 tests
        keys = [k for k in delong_results.keys()
                if 'high_myopia' in k and ('T2' in k or 'Lin2018' in k)]
    pvals = [(k, delong_results[k]['p']) for k in keys]
    pvals_sorted = sorted(pvals, key=lambda x: x[1])
    m = len(pvals_sorted)
    holm = []
    for i, (k, p) in enumerate(pvals_sorted):
        alpha_adj = 0.05 / (m - i)
        significant = p < alpha_adj
        holm.append({'comparison': k, 'p_raw': p,
                     'threshold': alpha_adj,
                     'significant_holm05': significant})
        print(f'  {k}: p={p:.4f}, Holm threshold={alpha_adj:.4f}, '
              f'significant={significant}')
    return {'family': family, 'n_tests': m, 'results': holm}


# ============================================================================
# 4. Leave-one-biometric-feature-out ablation
# ============================================================================
def leave_one_out_biometric(clean):
    """For T4 on 2-year high myopia, remove each biometric feature one at a time."""
    print('\n' + '=' * 60)
    print('4. LEAVE-ONE-BIOMETRIC-FEATURE-OUT ABLATION')
    print('=' * 60)

    BIOMETRIC_FEATS = ['pupil_OD', 'pupil_OS', 'IPD',
                        'gaze_mag_OD', 'gaze_mag_OS', 'pupil_asym']

    cohort = build_high_myopia_cohort(clean, 2)
    results = {'task': 'high_myopia_2yr', 'cohort_size': int(len(cohort)),
               'events': int(cohort['label'].sum())}

    # Baseline: full T4
    _, y, p = cv_pooled(cohort, T4, seed=42)
    pids = cohort.iloc[:]['pid'].values[np.argsort(
        _)] if False else cohort['pid'].values
    # Simpler: recompute the index
    idx, y, p = cv_pooled(cohort, T4, seed=42)
    pids = cohort.iloc[idx]['pid'].values
    t4_boot = patient_bootstrap_auc(pids, y, p)
    results['T4_full'] = {
        'auc': float(roc_auc_score(y, p)),
        'bootstrap': t4_boot,
    }
    print(f'  T4 full: AUC={roc_auc_score(y, p):.4f} '
          f'[{t4_boot["lo"]:.3f}, {t4_boot["hi"]:.3f}]')

    # Remove each biometric feature
    results['leave_one_out'] = {}
    for removed in BIOMETRIC_FEATS:
        feats = [f for f in T4 if f != removed]
        idx2, y2, p2 = cv_pooled(cohort, feats, seed=42)
        pids2 = cohort.iloc[idx2]['pid'].values
        boot = patient_bootstrap_auc(pids2, y2, p2)
        delta = float(roc_auc_score(y2, p2)) - results['T4_full']['auc']
        results['leave_one_out'][removed] = {
            'auc': float(roc_auc_score(y2, p2)),
            'bootstrap': boot,
            'delta_vs_T4': delta,
        }
        print(f'  T4 - {removed}: AUC={roc_auc_score(y2, p2):.4f} '
              f'[{boot["lo"]:.3f}, {boot["hi"]:.3f}], delta={delta:+.4f}')

    return results


def plot_leave_one_out(loo_results):
    """Bar plot of leave-one-out results with distinct colours per feature."""
    feats = list(loo_results['leave_one_out'].keys())
    deltas = [loo_results['leave_one_out'][f]['delta_vs_T4'] for f in feats]
    errors_lo = [loo_results['leave_one_out'][f]['bootstrap']['lo']
                  - loo_results['T4_full']['auc'] for f in feats]
    errors_hi = [loo_results['leave_one_out'][f]['bootstrap']['hi']
                  - loo_results['T4_full']['auc'] for f in feats]
    err_neg = [d - lo for d, lo in zip(deltas, errors_lo)]
    err_pos = [hi - d for d, hi in zip(deltas, errors_hi)]

    display_names = {
        'pupil_OD': 'Pupil (OD)', 'pupil_OS': 'Pupil (OS)',
        'IPD': 'Interpupil distance',
        'gaze_mag_OD': 'Gaze magnitude (OD)',
        'gaze_mag_OS': 'Gaze magnitude (OS)',
        'pupil_asym': 'Pupil asymmetry',
    }
    labels = [display_names[f] for f in feats]

    # Distinct colour per feature
    palette = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, deltas, xerr=[err_neg, err_pos],
                    color=palette[:len(feats)], edgecolor='black',
                    linewidth=0.6, capsize=4, height=0.6,
                    error_kw={'elinewidth': 1.2, 'color': '#444444'})
    ax.axvline(x=0, color='black', lw=0.8)
    ax.set_xlabel('$\\Delta$ AUC vs full T4 model (95% bootstrap CI)')
    ax.set_title('Leave-one-biometric-feature-out ablation\n'
                  '2-year high myopia prediction', loc='left', fontweight='bold')

    # Place annotations outside the error bars so they never overlap
    for bar, d, elo, ehi in zip(bars, deltas, err_neg, err_pos):
        # Position label at the far end of the error bar + padding
        if d >= 0:
            label_x = d + ehi + 0.004
            ha = 'left'
        else:
            label_x = d - elo - 0.004
            ha = 'right'
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{d:+.4f}', va='center', ha=ha, fontsize=9,
                fontweight='medium')

    # Pad x-axis so labels fit
    xmin = min(d - elo for d, elo in zip(deltas, err_neg)) - 0.015
    xmax = max(d + ehi for d, ehi in zip(deltas, err_pos)) + 0.015
    ax.set_xlim(xmin, xmax)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_leave_one_out.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig_leave_one_out.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig_leave_one_out.png/pdf')


# ============================================================================
# 5. Feature correlation heatmap
# ============================================================================
def feature_correlation_heatmap(clean):
    print('\n' + '=' * 60)
    print('5. FEATURE CORRELATION HEATMAP')
    print('=' * 60)

    cohort = build_high_myopia_cohort(clean, 2)
    feats_to_plot = [
        'age', 'SE_OD', 'SE_OS',
        'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
        'pupil_OD', 'pupil_OS', 'IPD',
        'gaze_mag_OD', 'gaze_mag_OS', 'pupil_asym',
    ]
    display_names = {
        'age': 'Age', 'SE_OD': 'SE (OD)', 'SE_OS': 'SE (OS)',
        'M_OD': 'M (OD)', 'J0_OD': '$J_0$ (OD)', 'J45_OD': '$J_{45}$ (OD)',
        'cyl_mag_OD': '|cyl| (OD)',
        'pupil_OD': 'Pupil (OD)', 'pupil_OS': 'Pupil (OS)',
        'IPD': 'Interpupil distance',
        'gaze_mag_OD': 'Gaze mag (OD)', 'gaze_mag_OS': 'Gaze mag (OS)',
        'pupil_asym': 'Pupil asymmetry',
    }

    df = cohort[feats_to_plot].fillna(cohort[feats_to_plot].median())
    corr = df.corr().values
    labels = [display_names[f] for f in feats_to_plot]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    # Mask upper triangle for clarity
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if i >= j:
                val = corr[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color)
            else:
                ax.text(j, i, '', ha='center', va='center')
    # Hide upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = np.where(mask, np.nan, corr)
    im = ax.imshow(corr_masked, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title('Feature correlation matrix (2-year high myopia cohort)',
                  loc='left', fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson correlation')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig_correlation.pdf', bbox_inches='tight')
    plt.close()
    print('  Saved fig_correlation.png/pdf')

    return {
        'feature_order': labels,
        'feature_keys': feats_to_plot,
        'correlations': corr.tolist(),
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    clean, _, _ = prepare_all()
    pairs = build_pairs(clean)
    all_results = {}

    all_results['temporal_split'] = temporal_split_evaluation(clean)
    all_results['delong_tests'] = delong_tests(clean, pairs)
    all_results['holm_bonferroni'] = holm_bonferroni(all_results['delong_tests'])
    all_results['leave_one_out'] = leave_one_out_biometric(clean)
    plot_leave_one_out(all_results['leave_one_out'])
    all_results['correlation_matrix'] = feature_correlation_heatmap(clean)

    # Save JSON
    def to_ser(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_ser(x) for x in obj]
        return obj

    with open(OUT_DIR / 'additional_results.json', 'w') as f:
        json.dump(to_ser(all_results), f, indent=2)
    print(f'\nResults saved to {OUT_DIR}/additional_results.json')


if __name__ == '__main__':
    main()
