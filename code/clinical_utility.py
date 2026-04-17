"""
Clinical utility analysis: operating characteristics, risk stratification,
and patient characteristics table for TRIPOD compliance.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                                roc_curve)
from prepare import prepare_all
from experiment import build_high_myopia_cohort, cv_pooled, T2, T4

OUT_DIR = Path('../results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def operating_characteristics(y_true, y_prob, thresholds):
    out = []
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        out.append({
            'threshold': float(thr),
            'sens': float(sens),
            'spec': float(spec),
            'ppv': float(ppv),
            'npv': float(npv),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        })
    return out


def risk_stratification(y_true, y_prob):
    """Three-category risk stratification: low <10%, moderate 10-30%, high >30%."""
    cat = np.zeros_like(y_prob, dtype=int)
    cat[y_prob >= 0.10] = 1
    cat[y_prob >= 0.30] = 2
    results = {}
    for lbl, c in [('low', 0), ('moderate', 1), ('high', 2)]:
        mask = cat == c
        n = int(mask.sum())
        events = int(y_true[mask].sum()) if mask.any() else 0
        rate = float(events / n) if n else 0.0
        results[lbl] = {'n': n, 'events': events, 'event_rate': rate}
    return results


def patient_table(cohort, label_col='label'):
    """TRIPOD-style Table 1 by outcome status."""
    out = {}
    for name, sub in [('overall', cohort),
                        ('events', cohort[cohort[label_col] == 1]),
                        ('non_events', cohort[cohort[label_col] == 0])]:
        out[name] = {
            'n': int(len(sub)),
            'age_mean': float(sub['age'].mean()),
            'age_sd': float(sub['age'].std()),
            'age_min': float(sub['age'].min()),
            'age_max': float(sub['age'].max()),
            'male_n': int(sub['gender_M'].sum()),
            'male_pct': float(sub['gender_M'].mean() * 100),
            'SE_OD_mean': float(sub['SE_OD'].mean()),
            'SE_OD_sd': float(sub['SE_OD'].std()),
            'SE_OS_mean': float(sub['SE_OS'].mean()),
            'SE_OS_sd': float(sub['SE_OS'].std()),
            'DC_OD_mean': float(sub['DC_OD'].mean()),
            'DC_OD_sd': float(sub['DC_OD'].std()),
            'aniso_mean': float(sub['aniso_SE'].mean()),
            'pupil_OD_mean': float(sub['pupil_OD'].mean()),
            'pupil_OD_sd': float(sub['pupil_OD'].std()),
            'IPD_mean': float(sub['IPD'].mean()),
            'IPD_sd': float(sub['IPD'].std()),
        }
    return out


def main():
    clean, _, _ = prepare_all()
    out = {}

    for horizon in [1, 2, 3]:
        cohort = build_high_myopia_cohort(clean, horizon)
        task_name = f'high_myopia_{horizon}yr'

        # Get T2 predictions
        idx, y, p = cv_pooled(cohort, T2, seed=42)

        # Operating characteristics at clinically meaningful thresholds
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        op_char = operating_characteristics(y, p, thresholds)

        # Risk stratification
        strat = risk_stratification(y, p)

        # Patient characteristics
        pt_table = patient_table(cohort.iloc[idx])

        out[task_name] = {
            'operating_characteristics': op_char,
            'risk_stratification': strat,
            'patient_characteristics': pt_table,
        }

    with open(OUT_DIR / 'clinical_utility.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved to {OUT_DIR}/clinical_utility.json')

    # Print 2-year summary
    print('\n=== Clinical Utility Summary: 2-year high myopia ===')
    hm2 = out['high_myopia_2yr']
    print('\nOperating characteristics (T2 model):')
    print(f'{"Thresh":<8}{"Sens":<8}{"Spec":<8}{"PPV":<8}{"NPV":<8}{"TP":<6}{"FP":<6}')
    for op in hm2['operating_characteristics']:
        print(f'{op["threshold"]:<8.2f}{op["sens"]:<8.3f}{op["spec"]:<8.3f}'
              f'{op["ppv"]:<8.3f}{op["npv"]:<8.3f}{op["tp"]:<6}{op["fp"]:<6}')
    print('\nRisk stratification:')
    for lbl, s in hm2['risk_stratification'].items():
        print(f'  {lbl}: n={s["n"]}, events={s["events"]} ({s["event_rate"]*100:.1f}%)')
    print('\nPatient characteristics:')
    o = hm2['patient_characteristics']['overall']
    e = hm2['patient_characteristics']['events']
    ne = hm2['patient_characteristics']['non_events']
    print(f'  Overall: n={o["n"]}, age={o["age_mean"]:.1f}+/-{o["age_sd"]:.1f}, male={o["male_pct"]:.1f}%')
    print(f'  SE_OD: {o["SE_OD_mean"]:.2f}+/-{o["SE_OD_sd"]:.2f}')
    print(f'  Events: n={e["n"]}, age={e["age_mean"]:.1f}, SE_OD={e["SE_OD_mean"]:.2f}')
    print(f'  Non-events: n={ne["n"]}, age={ne["age_mean"]:.1f}, SE_OD={ne["SE_OD_mean"]:.2f}')


if __name__ == '__main__':
    main()
