"""
IMMUTABLE evaluation harness for Paper 1 experiments.
Do NOT modify this file after Stage 6.

Provides:
- Data loading and cleaning from raw .xls files
- Cohort construction (onset / progression)
- Train/test/CV splits with patient-level grouping
- Metric computation (AUC, MAE, calibration, NRI, DCA)
- Bootstrap confidence intervals
"""
import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, mean_absolute_error,
    mean_squared_error, r2_score, f1_score, confusion_matrix
)
from scipy.stats import pearsonr, norm

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RANDOM_SEED = 42
MYOPIA_THRESHOLD = -0.50  # diopters
N_FOLDS = 5
BOOTSTRAP_N = 1000

ANONYMIZED_CSV = DATA_DIR / 'paediatric_myopia_screening_anonymized.csv'

# Feature sets for the model comparison.
# Tiers progressively add information to test incremental value:
#   T1: demographics only (floor)
#   T2: + spherical equivalent (current clinical practice)
#   T3: + sphere/cylinder/axis decomposed into power vectors
#   T4: + biometric features (pupil, IPD, gaze)
#   T5: + longitudinal trajectories (subcohort with prior visit)
FEATURE_SETS = {
    'T1_demographic': ['age', 'gender_M'],
    'T2_clinical': ['age', 'gender_M', 'SE_OD', 'SE_OS'],
    'T3_powervec': ['age', 'gender_M',
                     'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
                     'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS',
                     'aniso_SE'],
    'T4_biometric': ['age', 'gender_M',
                      'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
                      'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS',
                      'aniso_SE',
                      'pupil_OD', 'pupil_OS', 'IPD',
                      'gaze_mag_OD', 'gaze_mag_OS', 'pupil_asym'],
    'T5_longitudinal': ['age', 'gender_M',
                         'M_OD', 'J0_OD', 'J45_OD', 'cyl_mag_OD',
                         'M_OS', 'J0_OS', 'J45_OS', 'cyl_mag_OS',
                         'aniso_SE',
                         'pupil_OD', 'pupil_OS', 'IPD',
                         'gaze_mag_OD', 'gaze_mag_OS', 'pupil_asym',
                         'd_SE_OD', 'd_SE_OS', 'd_pupil_OD', 'd_pupil_OS',
                         'd_IPD', 'd_gaze_mag_OD',
                         'SE_OD_prior', 'pupil_OD_prior', 'IPD_prior',
                         'years_since_prior'],
}


# ============================================================================
# DATA LOADING
# ============================================================================
def parse_gaze(val):
    """Parse gaze string like '←2°' or '↑2°' into signed magnitude."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == '0°' or s == '0':
        return 0.0
    try:
        num = float(''.join(c for c in s if c.isdigit() or c == '.'))
        return num
    except (ValueError, TypeError):
        return np.nan


def load_raw_data():
    """Load the anonymized screening dataset released with this repository.

    The published dataset is a de-identified export of the original raw
    records: direct identifiers (date of birth, device record number) are
    removed, and an opaque keyed hash is substituted as the stable
    longitudinal patient identifier (`patient_id`). The hash key is not
    shipped, so the mapping back to individuals is not recoverable.
    """
    df = pd.read_csv(ANONYMIZED_CSV)
    df['year'] = df['Screening_Year'].astype(int)
    # Coerce object columns that should be numeric
    for col in ['Formatted Od DS', 'Od SE']:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def parse_axis(val):
    """Parse axis like '@95°' or '@180°' into integer degrees."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lstrip('@').rstrip('°')
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def power_vectors(ds, dc, axis_deg):
    """
    Decompose sphero-cylindrical refraction into power vectors (Thibos 1997):
        M  = DS + DC/2          (spherical equivalent)
        J0 = -(DC/2) * cos(2A)  (with-the-rule when positive)
        J45 = -(DC/2) * sin(2A) (oblique component)
    where A is axis in radians. Power vectors are clinically standard and
    handle the circularity of axis values properly.
    """
    a = np.radians(axis_deg)
    M = ds + dc / 2.0
    J0 = -(dc / 2.0) * np.cos(2 * a)
    J45 = -(dc / 2.0) * np.sin(2 * a)
    return M, J0, J45


def clean_and_engineer(df):
    """
    Standardize column names and engineer features.

    The released dataset already ships with the longitudinal patient identifier
    as an opaque keyed-hash column `patient_id`. This preserves the within-patient
    grouping structure required for patient-level cross-validation and bootstrap
    inference, while making re-identification infeasible because the hash key
    is not published alongside the data.
    """
    out = pd.DataFrame()
    out['year'] = df['year'].astype(int)
    out['gender'] = df['Gender']
    out['pid'] = df['patient_id'].astype(str)
    out['Id'] = out['pid']  # alias for code that expects 'Id'

    # Age at screening: use the precomputed fractional age from the release
    out['age'] = pd.to_numeric(df['Age_precise'], errors='coerce')
    out['gender_M'] = (out['gender'] == 'M').astype(int)

    # Refraction
    out['DS_OD'] = pd.to_numeric(df['Formatted Od DS'], errors='coerce')
    out['DC_OD'] = pd.to_numeric(df['Formatted Od DC'], errors='coerce')
    out['SE_OD'] = pd.to_numeric(df['Od SE'], errors='coerce')
    out['DS_OS'] = pd.to_numeric(df['Formatted Os DS'], errors='coerce')
    out['DC_OS'] = pd.to_numeric(df['Formatted Os DC'], errors='coerce')
    out['SE_OS'] = pd.to_numeric(df['OS SE'], errors='coerce')

    # Cylindrical axis (degrees) and power vector decomposition
    out['axis_OD'] = df['Formatted Od Axis'].apply(parse_axis)
    out['axis_OS'] = df['Formatted Os Axis'].apply(parse_axis)
    M_od, J0_od, J45_od = power_vectors(out['DS_OD'], out['DC_OD'], out['axis_OD'])
    M_os, J0_os, J45_os = power_vectors(out['DS_OS'], out['DC_OS'], out['axis_OS'])
    out['M_OD'] = M_od  # equivalent to SE_OD up to rounding
    out['J0_OD'] = J0_od
    out['J45_OD'] = J45_od
    out['M_OS'] = M_os
    out['J0_OS'] = J0_os
    out['J45_OS'] = J45_os
    # Astigmatism magnitude (always positive)
    out['cyl_mag_OD'] = out['DC_OD'].abs()
    out['cyl_mag_OS'] = out['DC_OS'].abs()
    # Anisometropia (always positive)
    out['aniso_SE'] = (out['SE_OD'] - out['SE_OS']).abs()

    # Biometric (the unique features)
    out['pupil_OD'] = pd.to_numeric(df['Formatted Od Pupil Size'], errors='coerce')
    out['pupil_OS'] = pd.to_numeric(df['Formatted Os Pupil Size'], errors='coerce')
    out['IPD'] = pd.to_numeric(df['Formatted Interpupil'], errors='coerce')
    out['pupil_asym'] = (out['pupil_OD'] - out['pupil_OS']).abs()

    # Gaze magnitude (unsigned, since we care about deviation from center)
    out['gaze_OD_X'] = df['Formatted Od Gaze X'].apply(parse_gaze)
    out['gaze_OD_Y'] = df['Formatted Od Gaze Y'].apply(parse_gaze)
    out['gaze_OS_X'] = df['Formatted Os Gaze X'].apply(parse_gaze)
    out['gaze_OS_Y'] = df['Formatted Os Gaze Y'].apply(parse_gaze)
    out['gaze_mag_OD'] = np.sqrt(out['gaze_OD_X']**2 + out['gaze_OD_Y']**2)
    out['gaze_mag_OS'] = np.sqrt(out['gaze_OS_X']**2 + out['gaze_OS_Y']**2)

    # Drop rows with missing core measurements
    core_cols = ['SE_OD', 'SE_OS', 'pupil_OD', 'pupil_OS', 'IPD', 'age', 'pid',
                  'M_OD', 'J0_OD', 'J45_OD']
    out = out.dropna(subset=core_cols)

    # Reasonable age filter (paediatric)
    out = out[(out['age'] >= 4) & (out['age'] <= 16)]

    # Drop duplicates: same patient (hashed patient_id) in same year, keep first
    out = out.drop_duplicates(subset=['pid', 'year'], keep='first')

    return out.reset_index(drop=True)


# ============================================================================
# COHORT CONSTRUCTION
# ============================================================================
def _compute_longitudinal_features(baseline, prior):
    """
    Compute per-year change features from baseline minus prior visit.
    Returns dict of d_* features.
    """
    if prior is None:
        return {
            'd_SE_OD': np.nan, 'd_SE_OS': np.nan,
            'd_pupil_OD': np.nan, 'd_pupil_OS': np.nan,
            'd_IPD': np.nan, 'd_gaze_mag_OD': np.nan,
            'SE_OD_prior': np.nan, 'pupil_OD_prior': np.nan,
            'IPD_prior': np.nan, 'years_since_prior': np.nan,
            'has_prior': 0,
        }
    delta_t = baseline['year'] - prior['year']
    if delta_t <= 0:
        delta_t = 1
    return {
        'd_SE_OD': (baseline['SE_OD'] - prior['SE_OD']) / delta_t,
        'd_SE_OS': (baseline['SE_OS'] - prior['SE_OS']) / delta_t,
        'd_pupil_OD': (baseline['pupil_OD'] - prior['pupil_OD']) / delta_t,
        'd_pupil_OS': (baseline['pupil_OS'] - prior['pupil_OS']) / delta_t,
        'd_IPD': (baseline['IPD'] - prior['IPD']) / delta_t,
        'd_gaze_mag_OD': (baseline['gaze_mag_OD'] - prior['gaze_mag_OD']) / delta_t,
        'SE_OD_prior': prior['SE_OD'],
        'pupil_OD_prior': prior['pupil_OD'],
        'IPD_prior': prior['IPD'],
        'years_since_prior': float(delta_t),
        'has_prior': 1,
    }


def build_onset_cohort(df):
    """
    Onset cohort: non-myopic in OD at baseline, has any future screening.
    Label: becomes_myopic = 1 if SE_OD <= -0.50 D at next visit.
    Uses OD (clinical convention) to be consistent with progression cohort
    and to avoid label dependence on which eye becomes myopic first.

    Each baseline record additionally carries longitudinal delta features
    computed from the most recent prior visit (if any).
    """
    df = df.sort_values(['pid', 'year']).reset_index(drop=True)
    samples = []

    for pid, group in df.groupby('pid'):
        group = group.sort_values('year').reset_index(drop=True)
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            baseline = group.iloc[i]
            followup = group.iloc[i + 1]

            # Inclusion: OD non-myopic at baseline
            if baseline['SE_OD'] <= MYOPIA_THRESHOLD:
                continue

            # Label: OD myopic at follow-up
            myopic_next = followup['SE_OD'] <= MYOPIA_THRESHOLD

            # Most recent prior visit (if any)
            prior = group.iloc[i - 1] if i > 0 else None
            long_feats = _compute_longitudinal_features(baseline, prior)

            sample = baseline.copy()
            sample['delta_t'] = followup['year'] - baseline['year']
            sample['label'] = int(myopic_next)
            sample['SE_OD_followup'] = followup['SE_OD']
            for k, v in long_feats.items():
                sample[k] = v
            samples.append(sample)

    cohort = pd.DataFrame(samples).reset_index(drop=True)
    return cohort


def build_rapid_progressor_cohort(df, threshold_d_per_year=-0.50):
    """
    Rapid progressor cohort.
    For each currently myopic baseline (OD SE <= -0.50), label = 1 if SE
    progresses at <= threshold D/year by next visit, else 0.
    Threshold default -0.50 D/year is the conventional cutoff in the
    paediatric myopia literature.
    """
    base = build_progression_cohort(df)
    if len(base) == 0:
        return base
    base = base.copy()
    base['rate'] = base['label']  # the original continuous label
    base['label'] = (base['rate'] <= threshold_d_per_year).astype(int)
    return base


def build_progression_cohort(df):
    """
    Progression cohort: myopic in OD at baseline, has future screening.
    Label: OD SE change rate (D/year). OD is the clinical convention.
    Uses OD only (not worse eye) to avoid swap artifacts when eyes have
    different SE values that change between visits.
    """
    df = df.sort_values(['pid', 'year']).reset_index(drop=True)
    samples = []

    for pid, group in df.groupby('pid'):
        group = group.sort_values('year').reset_index(drop=True)
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            baseline = group.iloc[i]
            followup = group.iloc[i + 1]

            # Inclusion: OD myopic at baseline
            if baseline['SE_OD'] > MYOPIA_THRESHOLD:
                continue

            delta_years = followup['year'] - baseline['year']
            if delta_years <= 0:
                continue

            rate = (followup['SE_OD'] - baseline['SE_OD']) / delta_years

            prior = group.iloc[i - 1] if i > 0 else None
            long_feats = _compute_longitudinal_features(baseline, prior)

            sample = baseline.copy()
            sample['delta_t'] = delta_years
            sample['label'] = rate
            sample['SE_OD_followup'] = followup['SE_OD']
            for k, v in long_feats.items():
                sample[k] = v
            samples.append(sample)

    cohort = pd.DataFrame(samples).reset_index(drop=True)
    return cohort


def get_features(cohort, tier_name):
    """Extract feature matrix for a given tier."""
    feats = FEATURE_SETS[tier_name]
    X = cohort[feats].copy()
    # Robust imputation: median for numeric features
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    # Final safety: replace any remaining NaN with 0 (e.g. all-NaN columns)
    X = X.fillna(0.0)
    return X.values, feats


def get_longitudinal_subcohort(cohort):
    """Restrict cohort to records with at least one prior visit."""
    return cohort[cohort['has_prior'] == 1].reset_index(drop=True)


# ============================================================================
# CV SPLITS (patient-level grouping to prevent leakage)
# ============================================================================
def get_cv_splits(cohort, n_splits=N_FOLDS, classification=True, seed=RANDOM_SEED):
    """Generate fold indices grouped by patient pid to prevent leakage."""
    groups = cohort['pid'].values
    if classification:
        y = cohort['label'].values
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(cohort)), y, groups=groups))
    else:
        rng = np.random.RandomState(seed)
        unique_pids = cohort['pid'].unique()
        rng.shuffle(unique_pids)
        pid_to_fold = {pid: i % n_splits for i, pid in enumerate(unique_pids)}
        fold_ids = np.array([pid_to_fold[pid] for pid in groups])
        splits = []
        for fold in range(n_splits):
            train_idx = np.where(fold_ids != fold)[0]
            test_idx = np.where(fold_ids == fold)[0]
            splits.append((train_idx, test_idx))
        return splits


def get_temporal_split(cohort):
    """Train on 2018-2020 baseline year, test on 2021 baseline year."""
    train_idx = cohort.index[cohort['year'] < 2021].values
    test_idx = cohort.index[cohort['year'] == 2021].values
    return train_idx, test_idx


# ============================================================================
# METRICS
# ============================================================================
def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """All classification metrics for one set of predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    if len(np.unique(y_true)) >= 2:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc'] = np.nan
    metrics['brier'] = brier_score_loss(y_true, y_prob)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return metrics


def compute_regression_metrics(y_true, y_pred):
    metrics = {}
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)
    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics['pearson_r'], _ = pearsonr(y_true, y_pred)
    else:
        metrics['pearson_r'] = np.nan
    return metrics


def bootstrap_ci(metric_fn, y_true, y_pred, n_iter=BOOTSTRAP_N, ci=0.95, seed=42):
    """Bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    estimates = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        try:
            est = metric_fn(y_true[idx], y_pred[idx])
            if not np.isnan(est):
                estimates.append(est)
        except (ValueError, ZeroDivisionError):
            continue
    if not estimates:
        return (np.nan, np.nan, np.nan)
    estimates = np.array(estimates)
    point = np.mean(estimates)
    lo = np.percentile(estimates, (1 - ci) / 2 * 100)
    hi = np.percentile(estimates, (1 + ci) / 2 * 100)
    return (point, lo, hi)


def delong_test(y_true, prob_a, prob_b):
    """
    DeLong test for two correlated AUCs.
    Returns z-score and two-sided p-value.
    """
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return 0.0, 1.0

    def compute_v(probs):
        pos_scores = probs[pos_mask]
        neg_scores = probs[neg_mask]
        # V10 and V01 components
        v10 = np.zeros(n_pos)
        v01 = np.zeros(n_neg)
        for i in range(n_pos):
            v10[i] = np.mean((pos_scores[i] > neg_scores).astype(float)
                             + 0.5 * (pos_scores[i] == neg_scores).astype(float))
        for j in range(n_neg):
            v01[j] = np.mean((pos_scores > neg_scores[j]).astype(float)
                             + 0.5 * (pos_scores == neg_scores[j]).astype(float))
        return v10, v01

    v10_a, v01_a = compute_v(prob_a)
    v10_b, v01_b = compute_v(prob_b)

    auc_a = np.mean(v10_a)
    auc_b = np.mean(v10_b)

    s10 = np.cov(np.stack([v10_a, v10_b]))
    s01 = np.cov(np.stack([v01_a, v01_b]))
    if s10.ndim == 0:
        s10 = np.array([[s10, 0], [0, s10]])
    if s01.ndim == 0:
        s01 = np.array([[s01, 0], [0, s01]])

    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n_pos + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n_neg

    if var <= 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def net_reclassification_improvement(y_true, prob_old, prob_new,
                                       low_threshold=0.10, high_threshold=0.30):
    """
    Three-category NRI: Low (<0.10), Med (0.10-0.30), High (>=0.30).
    Returns NRI = NRI_events + NRI_nonevents.
    """
    def to_cat(p):
        cat = np.zeros_like(p, dtype=int)
        cat[p >= low_threshold] = 1
        cat[p >= high_threshold] = 2
        return cat
    cat_old = to_cat(prob_old)
    cat_new = to_cat(prob_new)
    events = y_true == 1
    nonevents = y_true == 0

    if events.sum() == 0 or nonevents.sum() == 0:
        return np.nan

    up_events = ((cat_new > cat_old) & events).sum()
    down_events = ((cat_new < cat_old) & events).sum()
    up_nonevents = ((cat_new > cat_old) & nonevents).sum()
    down_nonevents = ((cat_new < cat_old) & nonevents).sum()

    nri_events = (up_events - down_events) / events.sum()
    nri_nonevents = (down_nonevents - up_nonevents) / nonevents.sum()
    return nri_events + nri_nonevents


def calibration_metrics(y_true, y_prob, n_bins=10):
    """Hosmer-Lemeshow style calibration."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    obs_freq = []
    pred_freq = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            obs_freq.append(y_true[mask].mean())
            pred_freq.append(y_prob[mask].mean())
    if not obs_freq:
        return {'cal_slope': np.nan, 'cal_intercept': np.nan, 'mean_abs_cal_error': np.nan}
    obs_freq = np.array(obs_freq)
    pred_freq = np.array(pred_freq)
    # Calibration slope via linear regression of obs on pred
    if len(obs_freq) >= 2 and np.std(pred_freq) > 0:
        slope, intercept = np.polyfit(pred_freq, obs_freq, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        'cal_slope': float(slope),
        'cal_intercept': float(intercept),
        'mean_abs_cal_error': float(np.mean(np.abs(obs_freq - pred_freq))),
    }


def decision_curve_net_benefit(y_true, y_prob, thresholds=None):
    """Compute net benefit across threshold probabilities for DCA."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)
    n = len(y_true)
    nb_model = []
    nb_treat_all = []
    for pt in thresholds:
        pred_pos = y_prob >= pt
        tp = ((pred_pos == 1) & (y_true == 1)).sum()
        fp = ((pred_pos == 1) & (y_true == 0)).sum()
        nb = tp / n - (fp / n) * (pt / (1 - pt))
        nb_model.append(nb)
        # Treat all
        prev = y_true.mean()
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        nb_treat_all.append(nb_all)
    return {
        'thresholds': thresholds.tolist(),
        'nb_model': nb_model,
        'nb_treat_all': nb_treat_all,
    }


# ============================================================================
# MAIN HELPERS
# ============================================================================
def prepare_all():
    """One-shot: load, clean, and build both cohorts."""
    raw = load_raw_data()
    clean = clean_and_engineer(raw)
    onset = build_onset_cohort(clean)
    progression = build_progression_cohort(clean)
    return clean, onset, progression


def report_metrics(name, value):
    """Standard metric reporting format for stdout parsing."""
    print(f"{name}: {value}")


if __name__ == '__main__':
    # Quick sanity check when run directly
    clean, onset, prog = prepare_all()
    print(f"clean_records: {len(clean)}")
    print(f"clean_unique_ids: {clean['Id'].nunique()}")
    print(f"onset_cohort_size: {len(onset)}")
    print(f"onset_event_rate: {onset['label'].mean():.4f}")
    print(f"progression_cohort_size: {len(prog)}")
    print(f"progression_mean_rate: {prog['label'].mean():.4f}")
