"""
Bilateral data augmentation: each child contributes 2 records (OD and OS)
to double sample size for eye-specific outcomes.
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


def make_bilateral_pairs(clean):
    pairs = []
    clean_sorted = clean.sort_values(['pid', 'year']).reset_index(drop=True)
    for pid, group in clean_sorted.groupby('pid'):
        group = group.sort_values('year').reset_index(drop=True)
        for i in range(len(group) - 1):
            b = group.iloc[i]
            f = group.iloc[i + 1]
            for eye in ['OD', 'OS']:
                other = 'OS' if eye == 'OD' else 'OD'
                rec = {
                    'pid': b['pid'],
                    'eye': eye,
                    'year': b['year'],
                    'age': b['age'],
                    'gender_M': b['gender_M'],
                    'SE_eye': b[f'SE_{eye}'],
                    'SE_other': b[f'SE_{other}'],
                    'DS_eye': b[f'DS_{eye}'],
                    'DC_eye': b[f'DC_{eye}'],
                    'M_eye': b[f'M_{eye}'],
                    'J0_eye': b[f'J0_{eye}'],
                    'J45_eye': b[f'J45_{eye}'],
                    'cyl_mag_eye': b[f'cyl_mag_{eye}'],
                    'pupil_eye': b[f'pupil_{eye}'],
                    'pupil_other': b[f'pupil_{other}'],
                    'IPD': b['IPD'],
                    'gaze_mag_eye': b[f'gaze_mag_{eye}'],
                    'gaze_mag_other': b[f'gaze_mag_{other}'],
                    'pupil_diff': b[f'pupil_{eye}'] - b[f'pupil_{other}'],
                    'SE_diff': b[f'SE_{eye}'] - b[f'SE_{other}'],
                    'delta_t': f['year'] - b['year'],
                    'SE_eye_next': f[f'SE_{eye}'],
                    'cyl_mag_eye_next': abs(f[f'DC_{eye}']),
                }
                pairs.append(rec)
    return pd.DataFrame(pairs).reset_index(drop=True)


bil = make_bilateral_pairs(clean)
key = bil[['pid', 'eye']].drop_duplicates()
print(f'Bilateral pairs: {len(bil)}, unique patient-eyes: {len(key)}')

# Build the rapid progression cohort using bilateral
bil_rapid = bil[bil['SE_eye'] <= -0.5].copy()
bil_rapid['rate'] = (bil_rapid['SE_eye_next'] - bil_rapid['SE_eye']) / bil_rapid['delta_t']
bil_rapid['label'] = (bil_rapid['rate'] <= -0.5).astype(int)
print(f'Bilateral rapid cohort: n={len(bil_rapid)}, rapid_rate={bil_rapid["label"].mean():.3f}')


def cv_pooled_bil(cohort, feats, seed=42):
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    all_y, all_p = [], []
    for tr, te in splits:
        Xtr = cohort.iloc[tr][feats].fillna(cohort[feats].median()).values
        Xte = cohort.iloc[te][feats].fillna(cohort[feats].median()).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m = Pipeline([('s', StandardScaler()),
                       ('c', LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced'))]).fit(Xtr, ytr)
        all_y.extend(yte)
        all_p.extend(m.predict_proba(Xte)[:, 1])
    return np.array(all_y), np.array(all_p)


def multi_seed_bil(cohort, feats):
    aucs = []
    for s in [42, 123, 456, 789, 2024]:
        y, p = cv_pooled_bil(cohort, feats, seed=s)
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs), np.std(aucs)


T1 = ['age', 'gender_M']
T2 = ['age', 'gender_M', 'SE_eye', 'SE_other']
T3 = ['age', 'gender_M', 'SE_eye', 'SE_other', 'DS_eye', 'DC_eye', 'cyl_mag_eye', 'J0_eye', 'J45_eye']
T4 = T3 + ['pupil_eye', 'pupil_other', 'IPD', 'gaze_mag_eye', 'gaze_mag_other', 'pupil_diff']

print('\n=== Bilateral rapid progression task ===')
for name, fs in [('T1', T1), ('T2', T2), ('T3', T3), ('T4', T4)]:
    m, s = multi_seed_bil(bil_rapid, fs)
    print(f'  {name}: AUC={m:.4f} +/- {s:.4f}')

# Per-seed delta T4 vs T2
deltas = []
for seed in [42, 123, 456, 789, 2024]:
    y2, p2 = cv_pooled_bil(bil_rapid, T2, seed=seed)
    y4, p4 = cv_pooled_bil(bil_rapid, T4, seed=seed)
    deltas.append(roc_auc_score(y2, p4) - roc_auc_score(y2, p2))
print(f'\nPer-seed delta T4-T2: {[round(d, 4) for d in deltas]}')
print(f'Mean: {np.mean(deltas):.4f}, std: {np.std(deltas):.4f}')

# Bootstrap CI
y2, p2 = cv_pooled_bil(bil_rapid, T2, seed=42)
y4, p4 = cv_pooled_bil(bil_rapid, T4, seed=42)
np.random.seed(42)
boot_deltas = []
for _ in range(2000):
    idx = np.random.randint(0, len(y2), size=len(y2))
    if len(np.unique(y2[idx])) < 2:
        continue
    try:
        d = roc_auc_score(y2[idx], p4[idx]) - roc_auc_score(y2[idx], p2[idx])
        boot_deltas.append(d)
    except Exception:
        continue
print(f'\nBootstrap CI for T4-T2 delta (seed 42):')
print(f'  Mean: {np.mean(boot_deltas):.4f}')
print(f'  95% CI: [{np.percentile(boot_deltas, 2.5):.4f}, {np.percentile(boot_deltas, 97.5):.4f}]')
print(f'  P(positive): {(np.array(boot_deltas) > 0).mean():.4f}')

# High myopia development with bilateral
bil_hm = bil.copy()
bil_hm = bil_hm[bil_hm['SE_eye'] > -3.0].copy()
bil_hm['label'] = (bil_hm['SE_eye_next'] <= -3.0).astype(int)
print(f'\n=== Bilateral high myopia development ===')
print(f'n={len(bil_hm)}, event_rate={bil_hm["label"].mean():.3f}')
for name, fs in [('T1', T1), ('T2', T2), ('T3', T3), ('T4', T4)]:
    m, s = multi_seed_bil(bil_hm, fs)
    print(f'  {name}: AUC={m:.4f} +/- {s:.4f}')

# Bootstrap CI for HM
y2, p2 = cv_pooled_bil(bil_hm, T2, seed=42)
y4, p4 = cv_pooled_bil(bil_hm, T4, seed=42)
np.random.seed(42)
boot_deltas = []
for _ in range(2000):
    idx = np.random.randint(0, len(y2), size=len(y2))
    if len(np.unique(y2[idx])) < 2:
        continue
    try:
        d = roc_auc_score(y2[idx], p4[idx]) - roc_auc_score(y2[idx], p2[idx])
        boot_deltas.append(d)
    except Exception:
        continue
print(f'\nHM Bootstrap T4-T2 delta:')
print(f'  Mean: {np.mean(boot_deltas):.4f}')
print(f'  95% CI: [{np.percentile(boot_deltas, 2.5):.4f}, {np.percentile(boot_deltas, 97.5):.4f}]')
