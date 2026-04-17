"""
Compute decision curve analysis on isotonic-calibrated predictions.
The raw T2 predictions are under-confident because of class weighting
on an imbalanced outcome, which gives misleading DCA curves. Isotonic
recalibration fixes the probability scale without changing AUC.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from prepare import prepare_all, decision_curve_net_benefit, calibration_metrics
from experiment import (
    build_high_myopia_cohort, build_pairs, build_onset_cohort,
    build_mild_to_moderate_cohort, build_rapid_cohort,
    T2, T4, make_lr,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


OUT_DIR = Path('../results')


def cv_pooled_calibrated(cohort, feats, seed=42):
    """5-fold CV with isotonic calibration fit on the training fold only."""
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
        np.zeros(len(cohort)), cohort['label'].values, groups=cohort['pid'].values)
    out_idx, out_y = [], []
    out_p_raw, out_p_cal = [], []
    for tr, te in splits:
        med = cohort[feats].median()
        Xtr = cohort.iloc[tr][feats].fillna(med).values
        Xte = cohort.iloc[te][feats].fillna(med).values
        ytr = cohort.iloc[tr]['label'].values
        yte = cohort.iloc[te]['label'].values

        # Inner CV: train on 4/5 of tr, fit isotonic on 1/5 of tr
        inner_idx = np.arange(len(tr))
        rng = np.random.RandomState(seed)
        rng.shuffle(inner_idx)
        split_at = int(len(inner_idx) * 0.8)
        inner_tr = inner_idx[:split_at]
        inner_cal = inner_idx[split_at:]

        m = Pipeline([
            ('s', StandardScaler()),
            ('c', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                       class_weight='balanced', random_state=seed))
        ]).fit(Xtr[inner_tr], ytr[inner_tr])
        p_cal_train = m.predict_proba(Xtr[inner_cal])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip').fit(p_cal_train, ytr[inner_cal])

        # Refit on full train fold and then calibrate
        m_full = Pipeline([
            ('s', StandardScaler()),
            ('c', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                       class_weight='balanced', random_state=seed))
        ]).fit(Xtr, ytr)
        p_raw_test = m_full.predict_proba(Xte)[:, 1]
        p_cal_test = iso.predict(p_raw_test)

        out_idx.extend(te)
        out_y.extend(yte)
        out_p_raw.extend(p_raw_test)
        out_p_cal.extend(p_cal_test)
    return (np.array(out_idx), np.array(out_y),
            np.array(out_p_raw), np.array(out_p_cal))


def main():
    clean, _, _ = prepare_all()
    pairs = build_pairs(clean)
    tasks = {
        'high_myopia_1yr': build_high_myopia_cohort(clean, 1),
        'high_myopia_2yr': build_high_myopia_cohort(clean, 2),
        'high_myopia_3yr': build_high_myopia_cohort(clean, 3),
        'mild_to_moderate': build_mild_to_moderate_cohort(pairs),
        'onset': build_onset_cohort(pairs),
    }

    out = {}
    for task_name, cohort in tasks.items():
        print(f'\n=== {task_name} ===')
        print(f'  n={len(cohort)}, events={cohort["label"].sum()}, rate={cohort["label"].mean():.3f}')

        # Calibrated T2 predictions
        idx_t2, y_t2, p_raw_t2, p_cal_t2 = cv_pooled_calibrated(cohort, T2, seed=42)
        # Calibrated T4 predictions
        idx_t4, y_t4, p_raw_t4, p_cal_t4 = cv_pooled_calibrated(cohort, T4, seed=42)

        # Calibration metrics before and after
        cal_raw_t2 = calibration_metrics(y_t2, p_raw_t2, n_bins=10)
        cal_cal_t2 = calibration_metrics(y_t2, p_cal_t2, n_bins=10)
        print(f'  T2 raw calibration slope: {cal_raw_t2["cal_slope"]:.3f}, MAE: {cal_raw_t2["mean_abs_cal_error"]:.3f}')
        print(f'  T2 isotonic slope: {cal_cal_t2["cal_slope"]:.3f}, MAE: {cal_cal_t2["mean_abs_cal_error"]:.3f}')

        # DCA on calibrated predictions
        thresholds = np.arange(0.02, 0.85, 0.02)
        dca_t2 = decision_curve_net_benefit(y_t2, p_cal_t2, thresholds=thresholds)
        dca_t4 = decision_curve_net_benefit(y_t4, p_cal_t4, thresholds=thresholds)

        out[task_name] = {
            'n': int(len(cohort)),
            'events': int(cohort['label'].sum()),
            'rate': float(cohort['label'].mean()),
            'cal_raw_t2': cal_raw_t2,
            'cal_iso_t2': cal_cal_t2,
            'dca_t2_calibrated': dca_t2,
            'dca_t4_calibrated': dca_t4,
            'y_t2': y_t2.tolist(),
            'p_raw_t2': p_raw_t2.tolist(),
            'p_cal_t2': p_cal_t2.tolist(),
            'p_cal_t4': p_cal_t4.tolist(),
        }

    with open(OUT_DIR / 'calibrated_dca.json', 'w') as f:
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
        json.dump(to_ser(out), f, indent=2)
    print(f'\nSaved to {OUT_DIR}/calibrated_dca.json')


if __name__ == '__main__':
    main()
