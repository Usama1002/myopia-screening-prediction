"""
Comprehensive EDA for Paediatric Myopia Screening Dataset
==========================================================
Data from paediatric department (autorefractor screening), 2018-2021.
Analogous to the Xinjiang myopia screening dataset from Eric-project-13.

Key variables:
- OS = Oculus Sinister (Left Eye)
- OD = Oculus Dexter (Right Eye)
- DS = Spherical power (Diopters Sphere)
- DC = Cylindrical power (Diopters Cylinder)
- SE = Spherical Equivalent = DS + 0.5*DC
- Myopia: SE <= -0.50 D
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

DATA_DIR = Path('../data')
OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)

YEAR_FILES = {
    2018: '2018-A.xls',
    2019: '2019-A.xls',
    2020: '2020-A.xls',
    2021: '2021-A.xls',
}

# ============================================================================
# 1. LOAD & CLEAN DATA
# ============================================================================
print("="*80)
print("1. LOADING AND CLEANING DATA")
print("="*80)

dfs = {}
for year, fname in YEAR_FILES.items():
    df = pd.read_excel(DATA_DIR / fname)
    df['Screening_Year'] = year
    # Ensure numeric types for key columns
    for col in ['Formatted Od DS', 'Od SE']:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    dfs[year] = df
    print(f"\n{year}: {len(df)} records, {df['Gender'].nunique()} genders")
    print(f"  DOB range: {df['Date of Birth'].min().date()} to {df['Date of Birth'].max().date()}")
    print(f"  Age range at screening: {year - df['Date of Birth'].dt.year.max()} to {year - df['Date of Birth'].dt.year.min()} years")

# Combine all years
all_data = pd.concat(dfs.values(), ignore_index=True)
print(f"\n{'='*40}")
print(f"TOTAL RECORDS: {len(all_data)}")
print(f"UNIQUE IDs: {all_data['Id'].nunique()}")
print(f"Date of Birth range: {all_data['Date of Birth'].min().date()} to {all_data['Date of Birth'].max().date()}")

# Calculate age at screening
all_data['Age_at_screening'] = all_data['Screening_Year'] - all_data['Date of Birth'].dt.year
# More precise age
all_data['Age_precise'] = (pd.Timestamp(f'{all_data["Screening_Year"].iloc[0]}-09-01') - all_data['Date of Birth']).dt.days / 365.25
# Recalculate per row
all_data['Age_precise'] = all_data.apply(
    lambda r: (pd.Timestamp(f'{int(r["Screening_Year"])}-09-01') - r['Date of Birth']).days / 365.25, axis=1
)

# ============================================================================
# 2. MYOPIA CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("2. MYOPIA CLASSIFICATION")
print("="*80)

# Myopia definitions (using SE)
# Myopia: SE <= -0.50 D
# Mild myopia: -3.00 < SE <= -0.50
# Moderate myopia: -6.00 < SE <= -3.00
# High myopia: SE <= -6.00
# Hyperopia: SE >= +2.00
# Emmetropia: -0.50 < SE < +2.00

def classify_refractive(se):
    if pd.isna(se):
        return 'Unknown'
    if se <= -6.0:
        return 'High Myopia'
    elif se <= -3.0:
        return 'Moderate Myopia'
    elif se <= -0.50:
        return 'Mild Myopia'
    elif se < 2.0:
        return 'Emmetropia'
    else:
        return 'Hyperopia'

for eye, col in [('OD', 'Od SE'), ('OS', 'OS SE')]:
    all_data[f'{eye}_class'] = all_data[col].apply(classify_refractive)

# Myopia flags
all_data['Myopia_OD'] = all_data['Od SE'] <= -0.50
all_data['Myopia_OS'] = all_data['OS SE'] <= -0.50
all_data['Myopia_either'] = all_data['Myopia_OD'] | all_data['Myopia_OS']

# Astigmatism flags (|DC| >= 0.75 D is clinically significant)
all_data['Astig_OD'] = all_data['Formatted Od DC'].abs() >= 0.75
all_data['Astig_OS'] = all_data['Formatted Os DC'].abs() >= 0.75

# Anisometropia (|SE_OD - SE_OS| >= 1.0 D)
all_data['Anisometropia'] = (all_data['Od SE'] - all_data['OS SE']).abs() >= 1.0

print("\nOverall Refractive Error Distribution (Right Eye):")
print(all_data['OD_class'].value_counts().to_string())
print(f"\nOverall Myopia Prevalence:")
for year in sorted(YEAR_FILES.keys()):
    mask = all_data['Screening_Year'] == year
    n = mask.sum()
    myopia_n = all_data.loc[mask, 'Myopia_either'].sum()
    print(f"  {year}: {myopia_n}/{n} = {myopia_n/n*100:.1f}%")

total_myopia = all_data['Myopia_either'].sum()
print(f"  Overall: {total_myopia}/{len(all_data)} = {total_myopia/len(all_data)*100:.1f}%")

# ============================================================================
# 3. DEVICE DIAGNOSIS vs COMPUTED DIAGNOSIS
# ============================================================================
print("\n" + "="*80)
print("3. DEVICE DIAGNOSIS ANALYSIS")
print("="*80)

print("\nResult Status Distribution:")
print(all_data['Formatted Result Status Text'].value_counts().to_string())

print("\nTop 20 Result Combined Text:")
print(all_data['Formatted Result Combined Text'].value_counts().head(20).to_string())

# Parse the combined result text for conditions
conditions = ['Myopia', 'Hyperopia', 'Astigmatism', 'Anisometropia', 'Gaze']
for cond in conditions:
    all_data[f'Device_{cond}'] = all_data['Formatted Result Combined Text'].str.contains(cond, na=False)
    print(f"  Device flagged {cond}: {all_data[f'Device_{cond}'].sum()} ({all_data[f'Device_{cond}'].mean()*100:.1f}%)")

# ============================================================================
# FIGURE 1: Dataset Overview
# ============================================================================
print("\n" + "="*80)
print("GENERATING FIGURES...")
print("="*80)

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

# 1a. Records per year
ax1 = fig.add_subplot(gs[0, 0])
year_counts = all_data.groupby('Screening_Year').size()
colors_year = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars = ax1.bar(year_counts.index, year_counts.values, color=colors_year, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, year_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, str(val),
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.set_xlabel('Screening Year')
ax1.set_ylabel('Number of Records')
ax1.set_title('(a) Records per Screening Year')
ax1.set_xticks(year_counts.index)

# 1b. Gender distribution per year
ax2 = fig.add_subplot(gs[0, 1])
gender_year = all_data.groupby(['Screening_Year', 'Gender']).size().unstack(fill_value=0)
gender_year.plot(kind='bar', ax=ax2, color=['#E91E63', '#2196F3'], edgecolor='white', linewidth=1.5)
ax2.set_xlabel('Screening Year')
ax2.set_ylabel('Count')
ax2.set_title('(b) Gender Distribution by Year')
ax2.legend(title='Gender')
ax2.tick_params(axis='x', rotation=0)

# 1c. Age distribution
ax3 = fig.add_subplot(gs[0, 2])
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    ages = all_data.loc[mask, 'Age_at_screening']
    ax3.hist(ages, bins=range(4, 18), alpha=0.5, label=str(year), color=color, edgecolor='white')
ax3.set_xlabel('Age at Screening (years)')
ax3.set_ylabel('Count')
ax3.set_title('(c) Age Distribution by Year')
ax3.legend()

# 1d. SE distribution (Right Eye)
ax4 = fig.add_subplot(gs[1, 0])
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    se = all_data.loc[mask, 'Od SE'].dropna()
    ax4.hist(se, bins=50, alpha=0.5, label=f'{year} (n={len(se)})', color=color, density=True)
ax4.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, label='Myopia threshold (-0.5D)')
ax4.set_xlabel('Spherical Equivalent - Right Eye (D)')
ax4.set_ylabel('Density')
ax4.set_title('(d) SE Distribution (Right Eye)')
ax4.legend(fontsize=8)

# 1e. SE distribution (Left Eye)
ax5 = fig.add_subplot(gs[1, 1])
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    se = all_data.loc[mask, 'OS SE'].dropna()
    ax5.hist(se, bins=50, alpha=0.5, label=f'{year} (n={len(se)})', color=color, density=True)
ax5.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, label='Myopia threshold (-0.5D)')
ax5.set_xlabel('Spherical Equivalent - Left Eye (D)')
ax5.set_ylabel('Density')
ax5.set_title('(e) SE Distribution (Left Eye)')
ax5.legend(fontsize=8)

# 1f. Myopia prevalence by year
ax6 = fig.add_subplot(gs[1, 2])
prev_data = []
for year in sorted(YEAR_FILES.keys()):
    mask = all_data['Screening_Year'] == year
    n = mask.sum()
    for eye, col in [('Right Eye', 'Myopia_OD'), ('Left Eye', 'Myopia_OS'), ('Either Eye', 'Myopia_either')]:
        prev = all_data.loc[mask, col].mean() * 100
        prev_data.append({'Year': year, 'Eye': eye, 'Prevalence': prev})
prev_df = pd.DataFrame(prev_data)
for eye, marker, color in [('Right Eye', 'o', '#2196F3'), ('Left Eye', 's', '#4CAF50'), ('Either Eye', 'D', '#F44336')]:
    subset = prev_df[prev_df['Eye'] == eye]
    ax6.plot(subset['Year'], subset['Prevalence'], marker=marker, color=color, label=eye, linewidth=2, markersize=8)
ax6.set_xlabel('Year')
ax6.set_ylabel('Myopia Prevalence (%)')
ax6.set_title('(f) Myopia Prevalence Trend')
ax6.legend()
ax6.set_xticks(sorted(YEAR_FILES.keys()))

# 1g. Refractive error classification
ax7 = fig.add_subplot(gs[2, 0])
class_order = ['High Myopia', 'Moderate Myopia', 'Mild Myopia', 'Emmetropia', 'Hyperopia']
class_colors = ['#b71c1c', '#e53935', '#ef9a9a', '#81c784', '#4fc3f7']
class_counts = all_data['OD_class'].value_counts().reindex(class_order, fill_value=0)
bars = ax7.barh(class_order, class_counts.values, color=class_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, class_counts.values):
    ax7.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f'{val} ({val/len(all_data)*100:.1f}%)', va='center', fontsize=9)
ax7.set_xlabel('Count')
ax7.set_title('(g) Refractive Error Classification (Right Eye)')

# 1h. Pass/Fail rate by year
ax8 = fig.add_subplot(gs[2, 1])
pass_fail = all_data.groupby(['Screening_Year', 'Formatted Result Status Text']).size().unstack(fill_value=0)
if 'Passed' in pass_fail.columns and 'Failed' in pass_fail.columns:
    pass_rate = pass_fail['Passed'] / (pass_fail['Passed'] + pass_fail['Failed']) * 100
    fail_rate = pass_fail['Failed'] / (pass_fail['Passed'] + pass_fail['Failed']) * 100
    ax8.bar(pass_fail.index, pass_rate, color='#81c784', label='Passed', edgecolor='white')
    ax8.bar(pass_fail.index, fail_rate, bottom=pass_rate, color='#e57373', label='Failed', edgecolor='white')
    ax8.set_ylabel('Percentage (%)')
    ax8.set_xlabel('Year')
    ax8.set_title('(h) Screening Pass/Fail Rate')
    ax8.legend()
    ax8.set_xticks(sorted(YEAR_FILES.keys()))

# 1i. Binocular correlation (OD vs OS SE)
ax9 = fig.add_subplot(gs[2, 2])
valid = all_data[['Od SE', 'OS SE']].dropna()
ax9.scatter(valid['Od SE'], valid['OS SE'], alpha=0.15, s=8, c='#1565C0')
ax9.plot([-8, 8], [-8, 8], 'r--', alpha=0.5, label='y=x')
corr = valid['Od SE'].corr(valid['OS SE'])
ax9.set_xlabel('SE Right Eye (D)')
ax9.set_ylabel('SE Left Eye (D)')
ax9.set_title(f'(i) Binocular SE Correlation (r={corr:.3f})')
ax9.legend()

plt.suptitle('Paediatric Myopia Screening Dataset: Overview', fontsize=16, fontweight='bold', y=1.01)
plt.savefig(OUT_DIR / 'fig01_dataset_overview.png')
plt.close()
print("  Saved fig01_dataset_overview.png")

# ============================================================================
# FIGURE 2: Age-stratified myopia prevalence
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 2a. Myopia prevalence by age (all years combined)
ax = axes[0, 0]
age_prev = all_data.groupby('Age_at_screening').agg(
    n=('Myopia_either', 'size'),
    myopia=('Myopia_either', 'sum')
).reset_index()
age_prev['prevalence'] = age_prev['myopia'] / age_prev['n'] * 100
# Filter to ages with reasonable sample size
age_prev = age_prev[age_prev['n'] >= 10]
ax.bar(age_prev['Age_at_screening'], age_prev['prevalence'],
       color='#1565C0', edgecolor='white', alpha=0.8)
for _, row in age_prev.iterrows():
    ax.text(row['Age_at_screening'], row['prevalence'] + 1,
            f'n={int(row["n"])}', ha='center', fontsize=7, rotation=45)
ax.set_xlabel('Age at Screening (years)')
ax.set_ylabel('Myopia Prevalence (%)')
ax.set_title('(a) Myopia Prevalence by Age')

# 2b. Myopia prevalence by age and year
ax = axes[0, 1]
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    ag = all_data[mask].groupby('Age_at_screening').agg(
        n=('Myopia_either', 'size'),
        myopia=('Myopia_either', 'sum')
    ).reset_index()
    ag['prevalence'] = ag['myopia'] / ag['n'] * 100
    ag = ag[ag['n'] >= 5]
    ax.plot(ag['Age_at_screening'], ag['prevalence'], marker='o', label=str(year),
            color=color, linewidth=2, markersize=6)
ax.set_xlabel('Age at Screening (years)')
ax.set_ylabel('Myopia Prevalence (%)')
ax.set_title('(b) Myopia Prevalence by Age & Year')
ax.legend()

# 2c. Myopia prevalence by age and gender
ax = axes[1, 0]
for gender, color, marker in [('M', '#2196F3', 'o'), ('F', '#E91E63', 's')]:
    mask = all_data['Gender'] == gender
    ag = all_data[mask].groupby('Age_at_screening').agg(
        n=('Myopia_either', 'size'),
        myopia=('Myopia_either', 'sum')
    ).reset_index()
    ag['prevalence'] = ag['myopia'] / ag['n'] * 100
    ag = ag[ag['n'] >= 5]
    ax.plot(ag['Age_at_screening'], ag['prevalence'], marker=marker, label=gender,
            color=color, linewidth=2, markersize=6)
ax.set_xlabel('Age at Screening (years)')
ax.set_ylabel('Myopia Prevalence (%)')
ax.set_title('(c) Myopia Prevalence by Age & Gender')
ax.legend()

# 2d. Mean SE by age
ax = axes[1, 1]
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    ag = all_data[mask].groupby('Age_at_screening').agg(
        mean_se=('Od SE', 'mean'),
        std_se=('Od SE', 'std'),
        n=('Od SE', 'size')
    ).reset_index()
    ag = ag[ag['n'] >= 5]
    ax.plot(ag['Age_at_screening'], ag['mean_se'], marker='o', label=str(year),
            color=color, linewidth=2, markersize=6)
    ax.fill_between(ag['Age_at_screening'],
                     ag['mean_se'] - ag['std_se'],
                     ag['mean_se'] + ag['std_se'],
                     alpha=0.1, color=color)
ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Myopia threshold')
ax.set_xlabel('Age at Screening (years)')
ax.set_ylabel('Mean SE - Right Eye (D)')
ax.set_title('(d) Mean SE by Age & Year (shaded = 1 SD)')
ax.legend(fontsize=8)

plt.suptitle('Age-Stratified Myopia Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig02_age_stratified.png')
plt.close()
print("  Saved fig02_age_stratified.png")

# ============================================================================
# FIGURE 3: Unique Features - Pupil Size, Interpupil Distance, Gaze
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 3a. Pupil size distribution (OD)
ax = axes[0, 0]
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    vals = all_data.loc[mask, 'Formatted Od Pupil Size'].dropna()
    ax.hist(vals, bins=30, alpha=0.5, label=str(year), color=color, density=True)
ax.set_xlabel('Pupil Size - Right Eye (mm)')
ax.set_ylabel('Density')
ax.set_title('(a) Pupil Size Distribution (OD)')
ax.legend()

# 3b. Interpupil distance distribution
ax = axes[0, 1]
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    vals = all_data.loc[mask, 'Formatted Interpupil'].dropna()
    ax.hist(vals, bins=30, alpha=0.5, label=str(year), color=color, density=True)
ax.set_xlabel('Interpupil Distance (mm)')
ax.set_ylabel('Density')
ax.set_title('(b) Interpupil Distance Distribution')
ax.legend()

# 3c. Pupil size vs SE
ax = axes[0, 2]
valid = all_data[['Formatted Od Pupil Size', 'Od SE']].dropna()
ax.scatter(valid['Formatted Od Pupil Size'], valid['Od SE'], alpha=0.1, s=5, c='#1565C0')
# Add trend line
z = np.polyfit(valid['Formatted Od Pupil Size'], valid['Od SE'], 1)
p = np.poly1d(z)
x_line = np.linspace(valid['Formatted Od Pupil Size'].min(), valid['Formatted Od Pupil Size'].max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
corr = valid['Formatted Od Pupil Size'].corr(valid['Od SE'])
ax.set_xlabel('Pupil Size - Right Eye (mm)')
ax.set_ylabel('SE - Right Eye (D)')
ax.set_title(f'(c) Pupil Size vs SE (r={corr:.3f})')
ax.legend()

# 3d. Interpupil distance vs age
ax = axes[1, 0]
valid = all_data[['Age_at_screening', 'Formatted Interpupil']].dropna()
ag = valid.groupby('Age_at_screening').agg(
    mean_ipd=('Formatted Interpupil', 'mean'),
    std_ipd=('Formatted Interpupil', 'std'),
    n=('Formatted Interpupil', 'size')
).reset_index()
ag = ag[ag['n'] >= 5]
ax.errorbar(ag['Age_at_screening'], ag['mean_ipd'], yerr=ag['std_ipd'],
            fmt='o-', color='#1565C0', capsize=3, linewidth=2, markersize=6)
ax.set_xlabel('Age at Screening (years)')
ax.set_ylabel('Mean Interpupil Distance (mm)')
ax.set_title('(d) Interpupil Distance vs Age')

# 3e. Pupil size asymmetry (OD - OS)
ax = axes[1, 1]
all_data['Pupil_asymmetry'] = all_data['Formatted Od Pupil Size'] - all_data['Formatted Os Pupil Size']
ax.hist(all_data['Pupil_asymmetry'].dropna(), bins=50, color='#7B1FA2', alpha=0.7, edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
mean_asym = all_data['Pupil_asymmetry'].mean()
ax.set_xlabel('Pupil Size Asymmetry OD - OS (mm)')
ax.set_ylabel('Count')
ax.set_title(f'(e) Pupil Size Asymmetry (mean={mean_asym:.3f}mm)')

# 3f. Pupil size by myopia status
ax = axes[1, 2]
myopic = all_data[all_data['Myopia_OD'] == True]['Formatted Od Pupil Size'].dropna()
non_myopic = all_data[all_data['Myopia_OD'] == False]['Formatted Od Pupil Size'].dropna()
ax.hist(non_myopic, bins=30, alpha=0.5, color='#81c784', label=f'Non-myopic (n={len(non_myopic)})', density=True)
ax.hist(myopic, bins=30, alpha=0.5, color='#e53935', label=f'Myopic (n={len(myopic)})', density=True)
ax.set_xlabel('Pupil Size - Right Eye (mm)')
ax.set_ylabel('Density')
ax.set_title(f'(f) Pupil Size by Myopia Status')
ax.legend()

plt.suptitle('Unique Features: Pupil Size, Interpupil Distance & Gaze', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig03_unique_features.png')
plt.close()
print("  Saved fig03_unique_features.png")

# ============================================================================
# FIGURE 4: Astigmatism and Cylindrical Power Analysis
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 4a. DC distribution
ax = axes[0, 0]
ax.hist(all_data['Formatted Od DC'].dropna(), bins=50, alpha=0.6, color='#2196F3', label='OD', density=True)
ax.hist(all_data['Formatted Os DC'].dropna(), bins=50, alpha=0.6, color='#E91E63', label='OS', density=True)
ax.axvline(x=-0.75, color='red', linestyle='--', alpha=0.7, label='Clinically significant')
ax.set_xlabel('Cylindrical Power (D)')
ax.set_ylabel('Density')
ax.set_title('(a) Cylindrical Power Distribution')
ax.legend()

# 4b. Astigmatism prevalence by age
ax = axes[0, 1]
for eye, col, color in [('OD', 'Astig_OD', '#2196F3'), ('OS', 'Astig_OS', '#E91E63')]:
    ag = all_data.groupby('Age_at_screening').agg(
        n=(col, 'size'),
        astig=(col, 'sum')
    ).reset_index()
    ag['prevalence'] = ag['astig'] / ag['n'] * 100
    ag = ag[ag['n'] >= 5]
    ax.plot(ag['Age_at_screening'], ag['prevalence'], marker='o', label=eye,
            color=color, linewidth=2, markersize=6)
ax.set_xlabel('Age at Screening (years)')
ax.set_ylabel('Astigmatism Prevalence (%)')
ax.set_title('(b) Astigmatism Prevalence by Age')
ax.legend()

# 4c. Axis distribution (polar plot for OD)
ax = axes[0, 2]
# Parse axis values
axis_vals_od = all_data['Formatted Od Axis'].dropna()
axis_numeric = axis_vals_od.str.extract(r'@(\d+)°')[0].dropna().astype(float)
ax.hist(axis_numeric, bins=36, color='#FF9800', alpha=0.7, edgecolor='white')
ax.set_xlabel('Axis (degrees)')
ax.set_ylabel('Count')
ax.set_title('(c) Cylindrical Axis Distribution (OD)')

# 4d. DC vs DS (right eye)
ax = axes[1, 0]
valid = all_data[['Formatted Od DS', 'Formatted Od DC']].dropna()
ax.scatter(valid['Formatted Od DS'], valid['Formatted Od DC'], alpha=0.1, s=5, c='#1565C0')
ax.set_xlabel('Spherical Power DS - OD (D)')
ax.set_ylabel('Cylindrical Power DC - OD (D)')
ax.set_title('(d) DS vs DC Relationship (OD)')

# 4e. Anisometropia prevalence by age
ax = axes[1, 1]
ag = all_data.groupby('Age_at_screening').agg(
    n=('Anisometropia', 'size'),
    aniso=('Anisometropia', 'sum')
).reset_index()
ag['prevalence'] = ag['aniso'] / ag['n'] * 100
ag = ag[ag['n'] >= 5]
ax.plot(ag['Age_at_screening'], ag['prevalence'], marker='o', color='#7B1FA2', linewidth=2, markersize=8)
ax.set_xlabel('Age at Screening (years)')
ax.set_ylabel('Anisometropia Prevalence (%)')
ax.set_title('(e) Anisometropia Prevalence by Age')

# 4f. Co-occurrence of conditions
ax = axes[1, 2]
conditions_data = {
    'Myopia': all_data['Myopia_either'].sum(),
    'Astigmatism\n(OD)': all_data['Astig_OD'].sum(),
    'Astigmatism\n(OS)': all_data['Astig_OS'].sum(),
    'Anisometropia': all_data['Anisometropia'].sum(),
    'Myopia+Astig': (all_data['Myopia_either'] & (all_data['Astig_OD'] | all_data['Astig_OS'])).sum(),
}
bars = ax.bar(conditions_data.keys(), conditions_data.values(),
              color=['#e53935', '#2196F3', '#E91E63', '#7B1FA2', '#FF9800'], edgecolor='white')
for bar, val in zip(bars, conditions_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val}\n({val/len(all_data)*100:.1f}%)', ha='center', fontsize=9)
ax.set_ylabel('Count')
ax.set_title('(f) Condition Prevalence')

plt.suptitle('Astigmatism & Cylindrical Power Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig04_astigmatism.png')
plt.close()
print("  Saved fig04_astigmatism.png")

# ============================================================================
# FIGURE 5: Longitudinal Analysis (repeated IDs across years)
# ============================================================================
print("\n" + "="*80)
print("4. LONGITUDINAL ANALYSIS")
print("="*80)

# Check for repeated IDs across years
id_years = all_data.groupby('Id')['Screening_Year'].apply(list).reset_index()
id_years['n_years'] = id_years['Screening_Year'].apply(len)
id_years['years_set'] = id_years['Screening_Year'].apply(lambda x: sorted(set(x)))
id_years['n_unique_years'] = id_years['years_set'].apply(len)

print(f"\nTotal unique IDs: {len(id_years)}")
print(f"IDs appearing in multiple years:")
for n in range(1, 5):
    count = (id_years['n_unique_years'] == n).sum()
    print(f"  {n} year(s): {count} students")

# Get students with 2+ years
multi_year = id_years[id_years['n_unique_years'] >= 2]
print(f"\nStudents with 2+ years of data: {len(multi_year)}")

if len(multi_year) > 0:
    # For students with multiple years, compute SE change
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Get longitudinal data
    long_ids = multi_year['Id'].values
    long_data = all_data[all_data['Id'].isin(long_ids)].sort_values(['Id', 'Screening_Year'])

    # Compute year-over-year change
    changes = []
    for sid, group in long_data.groupby('Id'):
        group = group.sort_values('Screening_Year')
        if len(group) >= 2:
            for i in range(len(group) - 1):
                row1 = group.iloc[i]
                row2 = group.iloc[i + 1]
                delta_se_od = row2['Od SE'] - row1['Od SE']
                delta_se_os = row2['OS SE'] - row1['OS SE']
                delta_years = row2['Screening_Year'] - row1['Screening_Year']
                changes.append({
                    'Id': sid,
                    'Year_from': row1['Screening_Year'],
                    'Year_to': row2['Screening_Year'],
                    'SE_OD_from': row1['Od SE'],
                    'SE_OD_to': row2['Od SE'],
                    'SE_OS_from': row1['OS SE'],
                    'SE_OS_to': row2['OS SE'],
                    'Delta_SE_OD': delta_se_od,
                    'Delta_SE_OS': delta_se_os,
                    'Delta_years': delta_years,
                    'Rate_OD': delta_se_od / delta_years if delta_years > 0 else np.nan,
                    'Rate_OS': delta_se_os / delta_years if delta_years > 0 else np.nan,
                    'Age_from': row1['Age_at_screening'],
                    'Gender': row1['Gender'],
                })
    changes_df = pd.DataFrame(changes)

    if len(changes_df) > 0:
        print(f"\nYear-over-year transitions: {len(changes_df)}")
        print(f"Mean SE change rate (OD): {changes_df['Rate_OD'].mean():.3f} D/year")
        print(f"Mean SE change rate (OS): {changes_df['Rate_OS'].mean():.3f} D/year")

        # 5a. Distribution of SE change rate
        ax = axes[0, 0]
        ax.hist(changes_df['Rate_OD'].dropna(), bins=50, color='#1565C0', alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.75, color='red', linestyle='--', alpha=0.7, label='Rapid progression')
        ax.axvline(x=changes_df['Rate_OD'].mean(), color='green', linestyle='-', alpha=0.7,
                   label=f'Mean={changes_df["Rate_OD"].mean():.3f}')
        ax.set_xlabel('SE Change Rate (D/year) - OD')
        ax.set_ylabel('Count')
        ax.set_title('(a) Distribution of SE Change Rate (OD)')
        ax.legend()

        # 5b. SE trajectory spaghetti plot
        ax = axes[0, 1]
        sample_ids = multi_year.sample(min(100, len(multi_year)), random_state=42)['Id'].values
        for sid in sample_ids:
            student = long_data[long_data['Id'] == sid].sort_values('Screening_Year')
            ax.plot(student['Screening_Year'], student['Od SE'], alpha=0.3, linewidth=0.8, color='#1565C0')
        # Add mean trajectory
        mean_traj = long_data.groupby('Screening_Year')['Od SE'].mean()
        ax.plot(mean_traj.index, mean_traj.values, 'r-', linewidth=3, label='Mean', zorder=5)
        ax.set_xlabel('Year')
        ax.set_ylabel('SE - Right Eye (D)')
        ax.set_title(f'(b) Individual SE Trajectories (n={min(100, len(multi_year))} sample)')
        ax.legend()

        # 5c. Progression by baseline SE
        ax = axes[1, 0]
        changes_df['Baseline_category'] = pd.cut(changes_df['SE_OD_from'],
            bins=[-np.inf, -3, -0.5, 0.5, 2, np.inf],
            labels=['High Myopia\n(<-3D)', 'Myopia\n(-3 to -0.5D)',
                    'Emmetropia\n(-0.5 to 0.5D)', 'Low Hyperopia\n(0.5 to 2D)',
                    'Hyperopia\n(>2D)'])
        box_data = []
        box_labels = []
        for cat in changes_df['Baseline_category'].cat.categories:
            vals = changes_df[changes_df['Baseline_category'] == cat]['Rate_OD'].dropna()
            if len(vals) >= 3:
                box_data.append(vals)
                box_labels.append(cat)
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors_box = ['#b71c1c', '#e53935', '#81c784', '#4fc3f7', '#2196F3'][:len(box_data)]
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('SE Change Rate (D/year)')
        ax.set_title('(c) Progression Rate by Baseline SE')
        ax.tick_params(axis='x', rotation=15)

        # 5d. Progression by gender
        ax = axes[1, 1]
        for gender, color in [('M', '#2196F3'), ('F', '#E91E63')]:
            vals = changes_df[changes_df['Gender'] == gender]['Rate_OD'].dropna()
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=f'{gender} (n={len(vals)}, mean={vals.mean():.3f})',
                    density=True)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('SE Change Rate (D/year) - OD')
        ax.set_ylabel('Density')
        ax.set_title('(d) SE Change Rate by Gender')
        ax.legend()

    plt.suptitle('Longitudinal SE Progression Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig05_longitudinal.png')
    plt.close()
    print("  Saved fig05_longitudinal.png")
else:
    print("  No longitudinal data found (no repeated IDs across years)")

# ============================================================================
# FIGURE 6: COVID Impact Analysis (2020 data is notably smaller)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 6a. Sample size context
ax = axes[0]
year_sizes = all_data.groupby('Screening_Year').size()
colors_bar = ['#2196F3' if y != 2020 else '#FF9800' for y in year_sizes.index]
bars = ax.bar(year_sizes.index, year_sizes.values, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, year_sizes.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, str(val),
            ha='center', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Records')
ax.set_title('(a) Records per Year\n(2020 = COVID impact?)')

# 6b. Mean SE per year
ax = axes[1]
mean_se_year = all_data.groupby('Screening_Year').agg(
    mean_od=('Od SE', 'mean'),
    mean_os=('OS SE', 'mean'),
    std_od=('Od SE', 'std')
).reset_index()
ax.errorbar(mean_se_year['Screening_Year'], mean_se_year['mean_od'],
            yerr=mean_se_year['std_od'], fmt='o-', color='#1565C0', capsize=5,
            linewidth=2, markersize=8, label='OD')
ax.errorbar(mean_se_year['Screening_Year'], mean_se_year['mean_os'],
            yerr=mean_se_year['std_od'], fmt='s-', color='#E91E63', capsize=5,
            linewidth=2, markersize=8, label='OS')
ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3, label='Myopia threshold')
ax.set_xlabel('Year')
ax.set_ylabel('Mean SE (D)')
ax.set_title('(b) Mean SE by Year')
ax.legend()
ax.set_xticks(sorted(YEAR_FILES.keys()))

# 6c. Age distribution shift across years
ax = axes[2]
for year, color in zip(sorted(YEAR_FILES.keys()), colors_year):
    mask = all_data['Screening_Year'] == year
    ages = all_data.loc[mask, 'Age_at_screening'].dropna()
    ax.hist(ages, bins=range(4, 18), alpha=0.4, label=f'{year} (mean={ages.mean():.1f})',
            color=color, density=True)
ax.set_xlabel('Age at Screening')
ax.set_ylabel('Density')
ax.set_title('(c) Age Distribution Shift Across Years')
ax.legend()

plt.suptitle('Temporal Trends & COVID Impact', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig06_temporal.png')
plt.close()
print("  Saved fig06_temporal.png")

# ============================================================================
# FIGURE 7: Gaze Analysis (unique to this dataset)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Parse gaze values
def parse_gaze(val):
    """Parse gaze string like '←2°' into direction and magnitude."""
    if pd.isna(val) or val == '0°':
        return 0
    try:
        num = int(''.join(c for c in val if c.isdigit()))
        if '←' in val or '→' in val:
            return num if '→' in val else -num
        if '↑' in val or '↓' in val:
            return num if '↑' in val else -num
    except:
        return np.nan
    return np.nan

# Gaze X = horizontal, Gaze Y = vertical
all_data['Gaze_OD_X'] = all_data['Formatted Od Gaze X'].apply(parse_gaze)
all_data['Gaze_OD_Y'] = all_data['Formatted Od Gaze Y'].apply(parse_gaze)
all_data['Gaze_OS_X'] = all_data['Formatted Os Gaze X'].apply(parse_gaze)
all_data['Gaze_OS_Y'] = all_data['Formatted Os Gaze Y'].apply(parse_gaze)
all_data['Gaze_magnitude_OD'] = np.sqrt(all_data['Gaze_OD_X']**2 + all_data['Gaze_OD_Y']**2)

# 7a. Gaze magnitude distribution
ax = axes[0]
gaze_mag = all_data['Gaze_magnitude_OD'].dropna()
ax.hist(gaze_mag, bins=20, color='#7B1FA2', alpha=0.7, edgecolor='white')
ax.set_xlabel('Gaze Deviation Magnitude (degrees) - OD')
ax.set_ylabel('Count')
ax.set_title('(a) Gaze Deviation Magnitude')

# 7b. Gaze vs SE
ax = axes[1]
valid = all_data[['Gaze_magnitude_OD', 'Od SE']].dropna()
ax.scatter(valid['Gaze_magnitude_OD'], valid['Od SE'], alpha=0.1, s=5, c='#7B1FA2')
corr = valid['Gaze_magnitude_OD'].corr(valid['Od SE'])
# Bin-averaged trend
bins = pd.cut(valid['Gaze_magnitude_OD'], bins=10)
binned = valid.groupby(bins, observed=True)['Od SE'].mean()
ax.set_xlabel('Gaze Deviation (degrees) - OD')
ax.set_ylabel('SE - Right Eye (D)')
ax.set_title(f'(b) Gaze Deviation vs SE (r={corr:.3f})')

# 7c. Gaze flagged by device vs SE
ax = axes[2]
gaze_flagged = all_data[all_data['Device_Gaze'] == True]['Od SE'].dropna()
no_gaze = all_data[all_data['Device_Gaze'] == False]['Od SE'].dropna()
ax.hist(no_gaze, bins=50, alpha=0.5, color='#81c784', label=f'No gaze issue (n={len(no_gaze)})', density=True)
if len(gaze_flagged) > 0:
    ax.hist(gaze_flagged, bins=50, alpha=0.5, color='#e53935', label=f'Gaze flagged (n={len(gaze_flagged)})', density=True)
ax.set_xlabel('SE - Right Eye (D)')
ax.set_ylabel('Density')
ax.set_title('(c) SE Distribution: Gaze Flagged vs Normal')
ax.legend()

plt.suptitle('Gaze Direction Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig07_gaze.png')
plt.close()
print("  Saved fig07_gaze.png")

# ============================================================================
# FIGURE 8: Correlation Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 10))

numeric_cols = [
    'Age_at_screening', 'Formatted Od Pupil Size', 'Formatted Os Pupil Size',
    'Formatted Interpupil', 'Formatted Od DS', 'Formatted Od DC', 'Od SE',
    'Formatted Os DS', 'Formatted Os DC', 'OS SE',
    'Gaze_magnitude_OD', 'Pupil_asymmetry'
]
corr_data = all_data[numeric_cols].dropna()
corr_matrix = corr_data.corr()

# Rename for readability
rename = {
    'Age_at_screening': 'Age',
    'Formatted Od Pupil Size': 'Pupil OD',
    'Formatted Os Pupil Size': 'Pupil OS',
    'Formatted Interpupil': 'IPD',
    'Formatted Od DS': 'DS OD',
    'Formatted Od DC': 'DC OD',
    'Od SE': 'SE OD',
    'Formatted Os DS': 'DS OS',
    'Formatted Os DC': 'DC OS',
    'OS SE': 'SE OS',
    'Gaze_magnitude_OD': 'Gaze Mag OD',
    'Pupil_asymmetry': 'Pupil Asym',
}
corr_matrix = corr_matrix.rename(index=rename, columns=rename)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig08_correlations.png')
plt.close()
print("  Saved fig08_correlations.png")

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================
print("\n" + "="*80)
print("5. DETAILED SUMMARY STATISTICS")
print("="*80)

print("\n--- Per-Year Summary ---")
for year in sorted(YEAR_FILES.keys()):
    mask = all_data['Screening_Year'] == year
    d = all_data[mask]
    n = len(d)
    print(f"\n{year} (n={n}):")
    print(f"  Gender: M={sum(d['Gender']=='M')}/{n} ({sum(d['Gender']=='M')/n*100:.1f}%), "
          f"F={sum(d['Gender']=='F')}/{n} ({sum(d['Gender']=='F')/n*100:.1f}%)")
    print(f"  Age: {d['Age_at_screening'].mean():.1f} +/- {d['Age_at_screening'].std():.1f} "
          f"(range: {d['Age_at_screening'].min()}-{d['Age_at_screening'].max()})")
    print(f"  SE OD: {d['Od SE'].mean():.3f} +/- {d['Od SE'].std():.3f} D")
    print(f"  SE OS: {d['OS SE'].mean():.3f} +/- {d['OS SE'].std():.3f} D")
    print(f"  Myopia prevalence (either eye): {d['Myopia_either'].mean()*100:.1f}%")
    print(f"  Astigmatism OD (|DC|>=0.75D): {d['Astig_OD'].mean()*100:.1f}%")
    print(f"  Anisometropia (|SE diff|>=1D): {d['Anisometropia'].mean()*100:.1f}%")
    print(f"  Device Pass rate: {sum(d['Formatted Result Status Text']=='Passed')/n*100:.1f}%")

print("\n--- Severity Breakdown (All years, Right Eye) ---")
for cat in ['High Myopia', 'Moderate Myopia', 'Mild Myopia', 'Emmetropia', 'Hyperopia']:
    n = sum(all_data['OD_class'] == cat)
    print(f"  {cat}: {n} ({n/len(all_data)*100:.1f}%)")

print("\n--- Key Correlations ---")
pairs = [
    ('Age_at_screening', 'Od SE'),
    ('Formatted Od Pupil Size', 'Od SE'),
    ('Formatted Interpupil', 'Od SE'),
    ('Formatted Od DC', 'Od SE'),
    ('Gaze_magnitude_OD', 'Od SE'),
    ('Od SE', 'OS SE'),
]
for c1, c2 in pairs:
    valid = all_data[[c1, c2]].dropna()
    r = valid[c1].corr(valid[c2])
    name1 = rename.get(c1, c1)
    name2 = rename.get(c2, c2)
    print(f"  {name1} vs {name2}: r = {r:.3f}")

# ============================================================================
# COMPARISON TABLE WITH XINJIANG
# ============================================================================
print("\n" + "="*80)
print("6. COMPARISON WITH XINJIANG DATASET")
print("="*80)

print("""
FEATURE COMPARISON:
+---------------------------+---------------------------+---------------------------+
| Feature                   | Xinjiang Dataset          | Paediatric Dataset        |
+---------------------------+---------------------------+---------------------------+
| Total Records             | 288,160                   | ~5,546                    |
| Unique Students           | 155,083                   | ~4,800 (est.)             |
| Years                     | 2020-2023 (4 years)       | 2018-2021 (4 years)       |
| Schools                   | 289 schools               | Not specified (1 clinic?) |
| Ethnic groups             | 96.5% Uyghur, 2.6% Han   | Not available             |
| UCVA                      | Yes (OD, OS)              | No                        |
| Spherical Power (DS)      | Yes (OD, OS)              | Yes (OD, OS)              |
| Cylindrical Power (DC)    | Yes (OD, OS)              | Yes (OD, OS)              |
| Axis                      | Not mentioned             | Yes (OD, OS)              |
| Spherical Equivalent (SE) | Yes (OD, OS)              | Yes (OD, OS)              |
| Axial Length              | Yes                       | No                        |
| BCVA                      | Partial (~2.8%)           | No                        |
| Pupil Size                | No                        | Yes (OD, OS)              |
| Interpupil Distance       | No                        | Yes                       |
| Gaze Direction            | No                        | Yes (X, Y for OD, OS)     |
| School Environment        | Yes (MSAVI, Albedo, etc.) | No                        |
| Questionnaire             | Yes (3,508 students)      | No                        |
| Device Diagnosis          | No                        | Yes (Pass/Fail + details) |
+---------------------------+---------------------------+---------------------------+
""")

# ============================================================================
# SAVE CLEANED COMBINED DATA
# ============================================================================
output_path = Path('../data/combined_cleaned.csv')
all_data.to_csv(output_path, index=False)
print(f"\nSaved cleaned combined data to: {output_path}")
print(f"Shape: {all_data.shape}")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
