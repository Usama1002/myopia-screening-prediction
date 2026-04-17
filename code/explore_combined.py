"""
Comprehensive exploration with FAST patient-clustered bootstrap.
"""
import numpy as np
import pandas as pd
from prepare import prepare_all
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

clean, _, _ = prepare_all()
clean = clean.sort_values(['pid', 'year']).reset_index(drop=True)

T1 = ['age', 'gender_M']
T2 = ['age', 'gender_M', 'SE_OD', 'SE_OS']
T3 = ['age', 'gender_M', 'SE_OD', 'SE_OS', 'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
      'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS', 'aniso_SE']
T4 = T3 + ['pupil_OD', 'pupil_OS', 'IPD', 'gaze_mag_OD', 'gaze_mag_OS']


def cv_pooled_indexed(cohort, feats, seed=42):
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    out_idx, out_y, out_p = [], [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xtr = cohort.iloc[tr][feats].fillna(med).values
        Xte = cohort.iloc[te][feats].fillna(med).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values
        m = Pipeline([('s', StandardScaler()),
                       ('c', LogisticRegression(max_iter=2000, C=1.0,
                                                  class_weight='balanced'))]).fit(Xtr, ytr)
        out_idx.extend(te)
        out_y.extend(yte)
        out_p.extend(m.predict_proba(Xte)[:, 1])
    return np.array(out_idx), np.array(out_y), np.array(out_p)


def fast_patient_bootstrap(pids, y, p, n_iter=1000, seed=42):
    """Vectorized patient-clustered bootstrap. Pre-builds patient->indices map."""
    rng = np.random.RandomState(seed)
    pid_to_indices = {}
    for i, pid in enumerate(pids):
        pid_to_indices.setdefault(pid, []).append(i)
    unique_pids = np.array(list(pid_to_indices.keys()))
    pid_arrays = [np.array(pid_to_indices[pid]) for pid in unique_pids]

    aucs = []
    n_pids = len(unique_pids)
    for _ in range(n_iter):
        sampled_idx = rng.randint(0, n_pids, size=n_pids)
        idx_lists = [pid_arrays[i] for i in sampled_idx]
        idx = np.concatenate(idx_lists)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(ys, p[idx]))
        except Exception:
            continue
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def fast_bootstrap_delta(pids, y, p_a, p_b, n_iter=1000, seed=42):
    rng = np.random.RandomState(seed)
    pid_to_indices = {}
    for i, pid in enumerate(pids):
        pid_to_indices.setdefault(pid, []).append(i)
    unique_pids = np.array(list(pid_to_indices.keys()))
    pid_arrays = [np.array(pid_to_indices[pid]) for pid in unique_pids]

    deltas = []
    n_pids = len(unique_pids)
    for _ in range(n_iter):
        sampled_idx = rng.randint(0, n_pids, size=n_pids)
        idx = np.concatenate([pid_arrays[i] for i in sampled_idx])
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            d = roc_auc_score(ys, p_b[idx]) - roc_auc_score(ys, p_a[idx])
            deltas.append(d)
        except Exception:
            continue
    return np.mean(deltas), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5), (np.array(deltas) > 0).mean()


def evaluate_task(cohort, label):
    print(f'\n--- {label} ---')
    print(f'n={len(cohort)}, events={cohort["label"].sum()}, rate={cohort["label"].mean():.3f}')
    results = {}
    for name, fs in [('T1', T1), ('T2', T2), ('T3', T3), ('T4', T4)]:
        idx, y, p = cv_pooled_indexed(cohort, fs)
        pids = cohort.iloc[idx]['pid'].values
        m, lo, hi = fast_patient_bootstrap(pids, y, p)
        results[name] = (m, lo, hi, idx, y, p)
        print(f'  {name}: AUC={m:.4f} [{lo:.4f}, {hi:.4f}]')

    # Delta T4 vs T2
    _, y2, p2 = results['T2'][3], results['T2'][4], results['T2'][5]
    _, y4, p4 = results['T4'][3], results['T4'][4], results['T4'][5]
    pids = cohort.iloc[results['T2'][3]]['pid'].values
    dm, dlo, dhi, dpos = fast_bootstrap_delta(pids, y2, p2, p4)
    print(f'  Delta T4-T2: {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}], P(>0)={dpos:.3f}')

    # Delta T3 vs T2
    _, y3, p3 = results['T3'][3], results['T3'][4], results['T3'][5]
    dm, dlo, dhi, dpos = fast_bootstrap_delta(pids, y2, p2, p3)
    print(f'  Delta T3-T2: {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}], P(>0)={dpos:.3f}')
    return results


def build_pairs(clean):
    pairs = []
    for pid, group in clean.groupby('pid'):
        group = group.sort_values('year').reset_index(drop=True)
        for i in range(len(group) - 1):
            b = group.iloc[i].copy()
            f = group.iloc[i + 1]
            b['SE_OD_next'] = f['SE_OD']
            b['delta_t'] = f['year'] - b['year']
            pairs.append(b)
    return pd.DataFrame(pairs).reset_index(drop=True)


def build_hm(clean, h):
    samples = []
    for pid, g in clean.groupby('pid'):
        g = g.sort_values('year').reset_index(drop=True)
        for i in range(len(g)):
            b = g.iloc[i]
            if b['SE_OD'] <= -3.0:
                continue
            future = g.iloc[i + 1:]
            future = future[(future['year'] - b['year']) <= h]
            if len(future) == 0:
                continue
            label = (future['SE_OD'] <= -3.0).any()
            s = b.copy()
            s['label'] = int(label)
            samples.append(s)
    return pd.DataFrame(samples).reset_index(drop=True)


pairs = build_pairs(clean)

# === HIGH MYOPIA at 1, 2, 3 years ===
for h in [1, 2, 3]:
    cohort = build_hm(clean, h)
    evaluate_task(cohort, f'HIGH MYOPIA development ({h}-year horizon)')

# === Onset ===
onset = pairs[pairs['SE_OD'] > -0.5].copy()
onset['label'] = (onset['SE_OD_next'] <= -0.5).astype(int)
evaluate_task(onset, 'MYOPIA ONSET (1-year)')

# === Rapid progression ===
rapid = pairs[pairs['SE_OD'] <= -0.5].copy()
rapid['rate'] = (rapid['SE_OD_next'] - rapid['SE_OD']) / rapid['delta_t']
rapid['label'] = (rapid['rate'] <= -0.5).astype(int)
evaluate_task(rapid, 'RAPID PROGRESSION (myopic at baseline)')

# === Mild-to-moderate transition ===
m2m = pairs[(pairs['SE_OD'] <= -0.5) & (pairs['SE_OD'] > -3.0)].copy()
m2m['label'] = (m2m['SE_OD_next'] <= -3.0).astype(int)
if m2m['label'].sum() >= 30:
    evaluate_task(m2m, 'MILD-TO-MODERATE PROGRESSION')

print('\n' + '=' * 70)
print('DONE')
