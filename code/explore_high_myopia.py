"""
Deep exploration of HIGH MYOPIA development as the primary task.
This is the most clinically important outcome (sight-threatening myopia
leading to retinal detachment, glaucoma, etc).

Tests:
1. Multi-horizon prediction (1, 2, 3 years)
2. Non-linear models (gradient boosting)
3. Subgroup analyses (by age, by baseline SE)
4. Calibration, DCA, NRI
5. Patient-clustered bootstrap CIs
6. Whether biometric features help for THIS task in any subgroup
"""
import numpy as np
import pandas as pd
from prepare import prepare_all, MYOPIA_THRESHOLD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

clean, _, _ = prepare_all()
clean = clean.sort_values(['pid', 'year']).reset_index(drop=True)

# Build multi-horizon high myopia cohort
def build_hm_cohort(clean, horizon):
    samples = []
    for pid, group in clean.groupby('pid'):
        group = group.sort_values('year').reset_index(drop=True)
        for i in range(len(group)):
            b = group.iloc[i]
            if b['SE_OD'] <= -3.0:
                continue  # Already high myopia
            future = group.iloc[i + 1:]
            future = future[(future['year'] - b['year']) <= horizon]
            if len(future) == 0:
                continue
            label = (future['SE_OD'] <= -3.0).any()
            sample = b.copy()
            sample['label'] = int(label)
            sample['horizon'] = horizon
            samples.append(sample)
    return pd.DataFrame(samples).reset_index(drop=True)


def cv_pooled(cohort, feats, model_fn, seed=42):
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    all_idx, all_y, all_p = [], [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xtr = cohort.iloc[tr][feats].fillna(med).values
        Xte = cohort.iloc[te][feats].fillna(med).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m = model_fn(seed)
        m.fit(Xtr, ytr)
        all_idx.extend(te)
        all_y.extend(yte)
        if hasattr(m, 'predict_proba'):
            all_p.extend(m.predict_proba(Xte)[:, 1])
        else:
            all_p.extend(m.decision_function(Xte))
    return np.array(all_idx), np.array(all_y), np.array(all_p)


def make_lr(seed):
    return Pipeline([('s', StandardScaler()),
                     ('c', LogisticRegression(max_iter=2000, C=1.0,
                                                class_weight='balanced', random_state=seed))])

def make_gbm(seed):
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=20,
        random_state=seed)


def multi_seed(cohort, feats, model_fn):
    aucs = []
    for s in [42, 123, 456, 789, 2024]:
        _, y, p = cv_pooled(cohort, feats, model_fn, seed=s)
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs), np.std(aucs)


def patient_bootstrap_ci(cohort, feats_a, feats_b, model_fn_a, model_fn_b, n_iter=2000, seed=42):
    idx_a, y_a, p_a = cv_pooled(cohort, feats_a, model_fn_a, seed=42)
    idx_b, y_b, p_b = cv_pooled(cohort, feats_b, model_fn_b, seed=42)
    df = pd.DataFrame({
        'pid': cohort.iloc[idx_a]['pid'].values,
        'y': y_a,
        'p_a': p_a,
        'p_b': p_b,
    })
    rng = np.random.RandomState(seed)
    unique_pids = df['pid'].unique()
    deltas = []
    auc_a_boots = []
    auc_b_boots = []
    for _ in range(n_iter):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        sub = pd.concat([df[df['pid'] == pid] for pid in sampled])
        if len(np.unique(sub['y'])) < 2:
            continue
        try:
            a = roc_auc_score(sub['y'], sub['p_a'])
            b = roc_auc_score(sub['y'], sub['p_b'])
            auc_a_boots.append(a)
            auc_b_boots.append(b)
            deltas.append(b - a)
        except Exception:
            continue
    return {
        'delta_mean': np.mean(deltas),
        'delta_lo': np.percentile(deltas, 2.5),
        'delta_hi': np.percentile(deltas, 97.5),
        'p_positive': (np.array(deltas) > 0).mean(),
        'auc_a_mean': np.mean(auc_a_boots),
        'auc_a_lo': np.percentile(auc_a_boots, 2.5),
        'auc_a_hi': np.percentile(auc_a_boots, 97.5),
        'auc_b_mean': np.mean(auc_b_boots),
        'auc_b_lo': np.percentile(auc_b_boots, 2.5),
        'auc_b_hi': np.percentile(auc_b_boots, 97.5),
    }


T1 = ['age', 'gender_M']
T2 = ['age', 'gender_M', 'SE_OD', 'SE_OS']
T3 = ['age', 'gender_M', 'SE_OD', 'SE_OS', 'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
      'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS', 'aniso_SE']
T4 = T3 + ['pupil_OD', 'pupil_OS', 'IPD', 'gaze_mag_OD', 'gaze_mag_OS']

print('=' * 70)
print('HIGH MYOPIA DEVELOPMENT - MULTI-HORIZON ANALYSIS')
print('=' * 70)
for h in [1, 2, 3]:
    cohort = build_hm_cohort(clean, h)
    print(f'\n--- {h}-year horizon ---')
    print(f'n={len(cohort)}, events={cohort["label"].sum()}, rate={cohort["label"].mean():.3f}')

    print('\nLogistic regression (multi-seed):')
    for name, fs in [('T1', T1), ('T2', T2), ('T3', T3), ('T4', T4)]:
        m, s = multi_seed(cohort, fs, make_lr)
        print(f'  {name}: AUC={m:.4f} +/- {s:.4f}')

    if cohort['label'].sum() >= 100:
        print('\nGradient boosting (multi-seed):')
        for name, fs in [('T2', T2), ('T4', T4)]:
            m, s = multi_seed(cohort, fs, make_gbm)
            print(f'  {name}: AUC={m:.4f} +/- {s:.4f}')

# Focus on 2-year horizon (best balance of n and clinical relevance)
print('\n' + '=' * 70)
print('2-YEAR HORIZON DEEP DIVE')
print('=' * 70)
hm2 = build_hm_cohort(clean, 2)
print(f'n={len(hm2)}, events={hm2["label"].sum()}, rate={hm2["label"].mean():.3f}')

# Subgroup by baseline SE: who benefits from biometric features?
print('\n--- Subgroup by baseline SE ---')
hm2['se_group'] = pd.cut(hm2['SE_OD'],
                          bins=[-3.0, -2.0, -1.0, 0.0, 1.0, 5.0],
                          labels=['(-3,-2)', '(-2,-1)', '(-1,0)', '(0,1)', '(1+)'])
for grp in hm2['se_group'].cat.categories:
    sub = hm2[hm2['se_group'] == grp].reset_index(drop=True)
    if len(sub) < 50 or sub['label'].sum() < 5 or (sub['label'] == 0).sum() < 5:
        print(f'  SE {grp}: n={len(sub)}, events={sub["label"].sum()} - SKIP')
        continue
    try:
        m_t2, s_t2 = multi_seed(sub, T2, make_lr)
        m_t4, s_t4 = multi_seed(sub, T4, make_lr)
        print(f'  SE {grp}: n={len(sub)}, events={sub["label"].sum()}, T2={m_t2:.4f}, T4={m_t4:.4f}, delta={m_t4-m_t2:+.4f}')
    except Exception as e:
        print(f'  SE {grp}: failed - {e}')

# Subgroup by age
print('\n--- Subgroup by age ---')
hm2['age_group'] = pd.cut(hm2['age'], bins=[0, 8, 11, 20], labels=['under_8', '8_to_10', '11_plus'])
for grp in hm2['age_group'].cat.categories:
    sub = hm2[hm2['age_group'] == grp].reset_index(drop=True)
    if len(sub) < 50 or sub['label'].sum() < 5 or (sub['label'] == 0).sum() < 5:
        print(f'  Age {grp}: n={len(sub)}, events={sub["label"].sum()} - SKIP')
        continue
    try:
        m_t2, s_t2 = multi_seed(sub, T2, make_lr)
        m_t4, s_t4 = multi_seed(sub, T4, make_lr)
        print(f'  Age {grp}: n={len(sub)}, events={sub["label"].sum()}, T2={m_t2:.4f}, T4={m_t4:.4f}, delta={m_t4-m_t2:+.4f}')
    except Exception as e:
        print(f'  Age {grp}: failed - {e}')

# Patient-clustered bootstrap on T4 vs T2 for 2-year HM
print('\n--- Patient-clustered bootstrap T4 vs T2 (2-year HM) ---')
res = patient_bootstrap_ci(hm2, T2, T4, make_lr, make_lr)
print(f'  T2 AUC: {res["auc_a_mean"]:.4f} [{res["auc_a_lo"]:.4f}, {res["auc_a_hi"]:.4f}]')
print(f'  T4 AUC: {res["auc_b_mean"]:.4f} [{res["auc_b_lo"]:.4f}, {res["auc_b_hi"]:.4f}]')
print(f'  Delta: {res["delta_mean"]:.4f} [{res["delta_lo"]:.4f}, {res["delta_hi"]:.4f}], P(>0)={res["p_positive"]:.4f}')
