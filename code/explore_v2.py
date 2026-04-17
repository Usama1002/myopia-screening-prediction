"""
Deeper exploration:
1. Patient-clustered bootstrap (proper CIs)
2. Aggressive feature engineering (interactions, ratios, normative features)
3. Use full longitudinal sequence (mean, slope, variance per patient)
"""
import numpy as np
import pandas as pd
from prepare import prepare_all, MYOPIA_THRESHOLD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

clean, _, _ = prepare_all()
clean = clean.sort_values(['pid', 'year']).reset_index(drop=True)

# === BUILD ENRICHED FEATURE SET ===
# For each baseline observation, compute features from prior history (if available)
def build_enriched_pairs(clean):
    pairs = []
    for pid, group in clean.groupby('pid'):
        group = group.sort_values('year').reset_index(drop=True)
        for i in range(len(group) - 1):
            b = group.iloc[i].copy()
            f = group.iloc[i + 1]

            # Future labels
            b['SE_OD_next'] = f['SE_OD']
            b['SE_OS_next'] = f['SE_OS']
            b['delta_t'] = f['year'] - b['year']

            # Prior history (all observations BEFORE this baseline)
            prior = group.iloc[:i]
            if len(prior) > 0:
                b['has_history'] = 1
                b['n_prior_visits'] = len(prior)
                b['hist_age_min'] = prior['age'].min()
                b['hist_se_min'] = prior['SE_OD'].min()
                b['hist_se_max'] = prior['SE_OD'].max()
                b['hist_se_range'] = b['hist_se_max'] - b['hist_se_min']
                b['hist_se_slope'] = (b['SE_OD'] - prior['SE_OD'].iloc[0]) / max(b['age'] - prior['age'].iloc[0], 0.5)
                b['hist_pupil_min'] = prior['pupil_OD'].min()
                b['hist_pupil_max'] = prior['pupil_OD'].max()
                b['hist_pupil_range'] = b['hist_pupil_max'] - b['hist_pupil_min']
                b['hist_pupil_slope'] = (b['pupil_OD'] - prior['pupil_OD'].iloc[0]) / max(b['age'] - prior['age'].iloc[0], 0.5)
                b['hist_ipd_slope'] = (b['IPD'] - prior['IPD'].iloc[0]) / max(b['age'] - prior['age'].iloc[0], 0.5)
            else:
                b['has_history'] = 0
                for col in ['n_prior_visits', 'hist_age_min', 'hist_se_min', 'hist_se_max',
                            'hist_se_range', 'hist_se_slope', 'hist_pupil_min', 'hist_pupil_max',
                            'hist_pupil_range', 'hist_pupil_slope', 'hist_ipd_slope']:
                    b[col] = np.nan

            # Engineered features
            b['cyclopean_SE'] = (b['SE_OD'] + b['SE_OS']) / 2
            b['SE_diff'] = b['SE_OD'] - b['SE_OS']
            b['pupil_avg'] = (b['pupil_OD'] + b['pupil_OS']) / 2
            b['pupil_per_age'] = b['pupil_avg'] / max(b['age'], 1)
            b['IPD_per_age'] = b['IPD'] / max(b['age'], 1)
            b['pupil_x_age'] = b['pupil_avg'] * b['age']
            b['IPD_x_age'] = b['IPD'] * b['age']
            b['pupil_x_SE'] = b['pupil_avg'] * b['SE_OD']
            b['IPD_x_SE'] = b['IPD'] * b['SE_OD']
            b['gaze_avg'] = (b['gaze_mag_OD'] + b['gaze_mag_OS']) / 2

            pairs.append(b)
    return pd.DataFrame(pairs).reset_index(drop=True)


pairs = build_enriched_pairs(clean)
print(f'Total pairs: {len(pairs)}')
print(f'Pairs with prior history: {pairs["has_history"].sum()}')

# Build rapid progression cohort
rapid = pairs[pairs['SE_OD'] <= -0.5].copy()
rapid['rate'] = (rapid['SE_OD_next'] - rapid['SE_OD']) / rapid['delta_t']
rapid['label'] = (rapid['rate'] <= -0.5).astype(int)
print(f'Rapid cohort: n={len(rapid)}, event rate={rapid["label"].mean():.3f}')


def cv_pooled(cohort, feats, seed=42):
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    all_y, all_p = [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xtr = cohort.iloc[tr][feats].fillna(med).values
        Xte = cohort.iloc[te][feats].fillna(med).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m = Pipeline([('s', StandardScaler()),
                       ('c', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'))]).fit(Xtr, ytr)
        all_y.extend(yte)
        all_p.extend(m.predict_proba(Xte)[:, 1])
    return np.array(all_y), np.array(all_p)


def multi_seed(cohort, feats):
    aucs = []
    for s in [42, 123, 456, 789, 2024]:
        y, p = cv_pooled(cohort, feats, seed=s)
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs), np.std(aucs)


# Feature sets
T2 = ['age', 'gender_M', 'SE_OD', 'SE_OS']
T3 = ['age', 'gender_M', 'SE_OD', 'SE_OS', 'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
      'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS', 'aniso_SE']
T4 = T3 + ['pupil_OD', 'pupil_OS', 'IPD', 'gaze_mag_OD', 'gaze_mag_OS']

# Engineered features
T2_eng = T2 + ['cyclopean_SE', 'SE_diff']
T4_eng = T4 + ['pupil_avg', 'pupil_per_age', 'IPD_per_age',
                'pupil_x_age', 'IPD_x_age', 'pupil_x_SE', 'IPD_x_SE', 'gaze_avg']

# Longitudinal features (only for has_history=1 subset)
T_long = T4_eng + ['n_prior_visits', 'hist_se_slope', 'hist_pupil_slope', 'hist_ipd_slope',
                    'hist_se_range', 'hist_pupil_range']

print('\n=== Rapid progression: feature engineering ===')
for name, fs in [('T2 (clinical)', T2), ('T3 (powervec)', T3), ('T4 (biometric)', T4),
                  ('T2 + engineered', T2_eng), ('T4 + engineered', T4_eng)]:
    m, s = multi_seed(rapid, fs)
    print(f'  {name:30s}: AUC={m:.4f} +/- {s:.4f}, n_features={len(fs)}')

# Subset with longitudinal history
rapid_hist = rapid[rapid['has_history'] == 1].reset_index(drop=True)
print(f'\nRapid with prior history: n={len(rapid_hist)}, event rate={rapid_hist["label"].mean():.3f}')
for name, fs in [('T2', T2), ('T4', T4), ('T4_eng', T4_eng), ('T_longitudinal', T_long)]:
    if len(rapid_hist) >= 30:
        m, s = multi_seed(rapid_hist, fs)
        print(f'  {name:30s}: AUC={m:.4f} +/- {s:.4f}')


# Patient-clustered bootstrap CI for T4_eng vs T2
def patient_bootstrap_delta(cohort, fs_a, fs_b, n_iter=2000, seed=42):
    """Bootstrap by sampling patients (not individual records)."""
    rng = np.random.RandomState(seed)
    # Get pooled predictions ONCE
    y_a, p_a = cv_pooled(cohort, fs_a, seed=42)
    y_b, p_b = cv_pooled(cohort, fs_b, seed=42)
    # Build a map from cohort row to (y, p) values
    # cv_pooled returns predictions in fold order, so we need indices
    # Easier: re-do this with index tracking
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    all_idx, all_ya, all_pa, all_pb = [], [], [], []
    for tr, te in splits:
        med = cohort[fs_a].median()
        Xtr_a = cohort.iloc[tr][fs_a].fillna(med).values
        Xte_a = cohort.iloc[te][fs_a].fillna(med).values
        med_b = cohort[fs_b].median()
        Xtr_b = cohort.iloc[tr][fs_b].fillna(med_b).values
        Xte_b = cohort.iloc[te][fs_b].fillna(med_b).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m_a = Pipeline([('s', StandardScaler()),
                         ('c', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'))]).fit(Xtr_a, ytr)
        m_b = Pipeline([('s', StandardScaler()),
                         ('c', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'))]).fit(Xtr_b, ytr)
        all_idx.extend(te)
        all_ya.extend(yte)
        all_pa.extend(m_a.predict_proba(Xte_a)[:, 1])
        all_pb.extend(m_b.predict_proba(Xte_b)[:, 1])

    df = pd.DataFrame({
        'idx': all_idx,
        'pid': cohort.iloc[all_idx]['pid'].values,
        'y': all_ya,
        'p_a': all_pa,
        'p_b': all_pb,
    })

    unique_pids = df['pid'].unique()
    deltas = []
    for _ in range(n_iter):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        sub = pd.concat([df[df['pid'] == pid] for pid in sampled_pids])
        if len(np.unique(sub['y'])) < 2:
            continue
        try:
            d = roc_auc_score(sub['y'], sub['p_b']) - roc_auc_score(sub['y'], sub['p_a'])
            deltas.append(d)
        except Exception:
            continue
    return deltas


print('\n=== Patient-clustered bootstrap CI ===')
print('Rapid cohort, T4 (full biometric) vs T2 (clinical):')
deltas = patient_bootstrap_delta(rapid, T2, T4)
print(f'  Mean: {np.mean(deltas):.4f}')
print(f'  95% CI: [{np.percentile(deltas, 2.5):.4f}, {np.percentile(deltas, 97.5):.4f}]')
print(f'  P(positive): {(np.array(deltas) > 0).mean():.4f}')

print('\nRapid cohort, T4_eng vs T2:')
deltas = patient_bootstrap_delta(rapid, T2, T4_eng)
print(f'  Mean: {np.mean(deltas):.4f}')
print(f'  95% CI: [{np.percentile(deltas, 2.5):.4f}, {np.percentile(deltas, 97.5):.4f}]')
print(f'  P(positive): {(np.array(deltas) > 0).mean():.4f}')
