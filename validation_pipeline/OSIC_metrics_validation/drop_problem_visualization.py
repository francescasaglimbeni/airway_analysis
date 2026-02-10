"""
VISUAL EXAMPLE: Why Drop Prediction Fails
Shows 5 example patients for two different metrics to demonstrate the parallel slopes problem
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Load data
DATA_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\analyzis_base_results")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\drop_results")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "integrated_dataset.csv")

def create_per_patient_dataset(feature_name):
    """Create per-patient dataset with specified feature"""
    print(f"\nCreating per-patient dataset for {feature_name}...")
    patients = []
    for patient, group in df.groupby('patient'):
        early = group[group['week'] <= 20].copy()
        if len(early) == 0:
            continue
        early['dist_to_0'] = abs(early['week'] - 0)
        week0_row = early.loc[early['dist_to_0'].idxmin()]
        
        late = group[group['week'] >= 35].copy()
        if len(late) == 0:
            continue
        late['dist_to_52'] = abs(late['week'] - 52)
        week52_row = late.loc[late['dist_to_52'].idxmin()]
        
        patients.append({
            'patient': patient,
            feature_name: week0_row[feature_name],
            'FVC_percent_week0': week0_row['Percent'],
            'FVC_percent_week52': week52_row['Percent'],
            'FVC_drop': week0_row['Percent'] - week52_row['Percent']
        })
    
    df_patients = pd.DataFrame(patients)
    df_patients = df_patients.dropna()
    print(f"  Total patients with complete data: {len(df_patients)}")
    return df_patients

def select_representative_patients(df_patients, feature_name, labels_low_high):
    """Select 5 representative patients at different percentiles"""
    df_sorted = df_patients.sort_values(feature_name)
    n = len(df_sorted)
    indices = [
        int(n * 0.1),  # 10th percentile
        int(n * 0.3),  # 30th percentile
        int(n * 0.5),  # 50th percentile (median)
        int(n * 0.7),  # 70th percentile
        int(n * 0.9)   # 90th percentile
    ]
    selected_patients = df_sorted.iloc[indices].copy()
    selected_patients['patient_label'] = [
        f'Patient A\n({labels_low_high[0]})', 
        'Patient B', 
        'Patient C\n(Median)', 
        'Patient D', 
        f'Patient E\n({labels_low_high[1]})'
    ]
    
    print(f"\n  Selected 5 patients:")
    print(selected_patients[['patient', feature_name, 'FVC_percent_week0', 'FVC_percent_week52', 'FVC_drop']])
    return selected_patients

def create_visualization(df_patients, feature_name, feature_label, title_suffix, output_filename):
    """Create 3-plot visualization for a given feature"""
    print(f"\n{'='*80}")
    print(f"Creating visualization for {feature_name}")
    print(f"{'='*80}")
    
    # Fit regression models on ALL data
    X = df_patients[feature_name].values.reshape(-1, 1)
    
    model_w0 = LinearRegression()
    model_w0.fit(X, df_patients['FVC_percent_week0'])
    print(f"Week0 model: FVC = {model_w0.coef_[0]:.2f} × {feature_name} + {model_w0.intercept_:.2f}")
    
    model_w52 = LinearRegression()
    model_w52.fit(X, df_patients['FVC_percent_week52'])
    print(f"Week52 model: FVC = {model_w52.coef_[0]:.2f} × {feature_name} + {model_w52.intercept_:.2f}")
    
    drop_coef = model_w0.coef_[0] - model_w52.coef_[0]
    drop_intercept = model_w0.intercept_ - model_w52.intercept_
    print(f"Expected Drop: Drop = {drop_coef:.2f} × {feature_name} + {drop_intercept:.2f}")
    
    model_drop = LinearRegression()
    model_drop.fit(X, df_patients['FVC_drop'])
    print(f"Actual Drop model: Drop = {model_drop.coef_[0]:.2f} × {feature_name} + {model_drop.intercept_:.2f}")
    
    # Select representative patients
    if 'entropy' in feature_name.lower():
        labels = ['Low entropy', 'High entropy']
    else:
        labels = ['Low density', 'High density']
    selected_patients = select_representative_patients(df_patients, feature_name, labels)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color palette for patients
    patient_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Calculate predictions for all patients
    pred_w0_all = model_w0.predict(X)
    actual_w0_all = df_patients['FVC_percent_week0'].values
    
    pred_w52_all = model_w52.predict(X)
    actual_w52_all = df_patients['FVC_percent_week52'].values
    
    pred_drop_all = model_w0.predict(X) - model_w52.predict(X)
    actual_drop_all = df_patients['FVC_drop'].values
    
    # Calculate predictions for selected patients
    selected_patients['pred_w0'] = model_w0.predict(selected_patients[[feature_name]])
    selected_patients['pred_w52'] = model_w52.predict(selected_patients[[feature_name]])
    selected_patients['pred_drop'] = selected_patients['pred_w0'] - selected_patients['pred_w52']
    
    # ========================================================================
    # Plot 1: Week 0 (Predicted vs Actual)
    # ========================================================================
    ax1.scatter(actual_w0_all, pred_w0_all, 
               alpha=0.3, s=40, color='gray', label='All patients')
    
    min_val = min(actual_w0_all.min(), pred_w0_all.min())
    max_val = max(actual_w0_all.max(), pred_w0_all.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2.5, alpha=0.8, label='Perfect')
    
    z = np.polyfit(actual_w0_all, pred_w0_all, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min_val, max_val, 100)
    ax1.plot(x_trend, p(x_trend), 'b-', linewidth=2.5, alpha=0.8, label='Regression')
    
    for idx, (_, patient) in enumerate(selected_patients.iterrows()):
        ax1.scatter(patient['FVC_percent_week0'], patient['pred_w0'], 
                   s=150, color=patient_colors[idx], edgecolors='black', linewidth=2, 
                   marker='o', zorder=5, label=patient['patient_label'])
    
    ax1.set_xlabel('Actual FVC % at Week 0', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted FVC % at Week 0', fontsize=12, fontweight='bold')
    ax1.set_title('Week 0 Prediction', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    r2_w0 = r2_score(actual_w0_all, pred_w0_all)
    ax1.text(0.95, 0.05, f'R² = {r2_w0:.3f}', transform=ax1.transAxes, 
            fontsize=11, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # Plot 2: Week 52 (Predicted vs Actual)
    # ========================================================================
    ax2.scatter(actual_w52_all, pred_w52_all, 
               alpha=0.3, s=40, color='gray', label='All patients')
    
    min_val = min(actual_w52_all.min(), pred_w52_all.min())
    max_val = max(actual_w52_all.max(), pred_w52_all.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2.5, alpha=0.8, label='Perfect')
    
    z = np.polyfit(actual_w52_all, pred_w52_all, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min_val, max_val, 100)
    ax2.plot(x_trend, p(x_trend), 'orange', linewidth=2.5, alpha=0.8, label='Regression')
    
    for idx, (_, patient) in enumerate(selected_patients.iterrows()):
        ax2.scatter(patient['FVC_percent_week52'], patient['pred_w52'], 
                   s=150, color=patient_colors[idx], edgecolors='black', linewidth=2, 
                   marker='s', zorder=5)
    
    ax2.set_xlabel('Actual FVC % at Week 52', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted FVC % at Week 52', fontsize=12, fontweight='bold')
    ax2.set_title('Week 52 Prediction', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    r2_w52 = r2_score(actual_w52_all, pred_w52_all)
    ax2.text(0.95, 0.05, f'R² = {r2_w52:.3f}', transform=ax2.transAxes, 
            fontsize=11, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # Plot 3: Drop (Predicted vs Actual)
    # ========================================================================
    ax3.scatter(actual_drop_all, pred_drop_all, 
               alpha=0.3, s=40, color='gray', label='All patients')
    
    min_val = min(actual_drop_all.min(), pred_drop_all.min())
    max_val = max(actual_drop_all.max(), pred_drop_all.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2.5, alpha=0.8, label='Perfect')
    
    z = np.polyfit(actual_drop_all, pred_drop_all, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min_val, max_val, 100)
    ax3.plot(x_trend, p(x_trend), 'green', linewidth=2.5, alpha=0.8, label='Regression')
    
    for idx, (_, patient) in enumerate(selected_patients.iterrows()):
        ax3.scatter(patient['FVC_drop'], patient['pred_drop'], 
                   s=150, color=patient_colors[idx], edgecolors='black', linewidth=2, 
                   marker='D', zorder=5)
    
    ax3.set_xlabel('Actual FVC Drop', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted FVC Drop', fontsize=12, fontweight='bold')
    ax3.set_title('Drop Prediction (POOR!)', fontsize=14, fontweight='bold', color='red')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    r2_drop = r2_score(actual_drop_all, pred_drop_all)
    ax3.text(0.95, 0.05, f'R² = {r2_drop:.3f}', transform=ax3.transAxes, 
            fontsize=11, fontweight='bold', color='red', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    
    fig.suptitle(f'Why Drop Prediction Fails with {feature_label}: Week 0 and Week 52 Predict Well, but Drop Does Not', 
                fontsize=16, fontweight='bold', y=1.02)
    
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("DROP PREDICTION PROBLEM VISUALIZATION")
print("Creating visualizations for 2 different metrics")
print("="*80)

# Metric 1: Histogram Entropy
df_entropy = create_per_patient_dataset('histogram_entropy')
create_visualization(
    df_entropy, 
    'histogram_entropy', 
    'Histogram Entropy',
    'Histogram Entropy',
    'drop_problem_histogram_entropy.png'
)

# Metric 2: Mean Lung Density
df_density = create_per_patient_dataset('mean_lung_density_HU')
create_visualization(
    df_density, 
    'mean_lung_density_HU', 
    'Mean Lung Density',
    'Mean Lung Density',
    'drop_problem_mean_lung_density.png'
)

print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print(f"Both figures saved in: {OUTPUT_DIR}")
print("  1. drop_problem_histogram_entropy.png")
print("  2. drop_problem_mean_lung_density.png")
