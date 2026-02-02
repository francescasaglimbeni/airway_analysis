"""
IMPROVED FVC PREDICTION - BALANCED APPROACH
More flexible than strict windows, but more controlled than original
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation.csv")
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_combined")
TRAIN_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\train.csv")
TEST_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\test.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\improved_prediction")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ORIGINAL FUNCTIONS (KEPT FOR COMPATIBILITY)
# ============================================================================

def load_clinical_data():
    """Load and combine train.csv and test.csv"""
    print("Loading clinical data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    clinical = pd.concat([train, test], ignore_index=True)
    
    print(f"  Loaded {len(clinical)} clinical records")
    print(f"  Unique patients: {clinical['Patient'].nunique()}")
    
    return clinical

def load_validation_results():
    """Load validation results and filter RELIABLE cases"""
    print("\nLoading validation results...")
    validation = pd.read_csv(VALIDATION_CSV)
    reliable = validation[validation['status'] == 'RELIABLE'].copy()
    
    print(f"  Total cases: {len(validation)}")
    print(f"  RELIABLE cases: {len(reliable)}")
    
    return reliable

def load_advanced_metrics(case_name):
    """Load advanced metrics JSON for a specific case"""
    json_path = RESULTS_ROOT / case_name / "step4_analysis" / "advanced_metrics.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def load_parenchymal_metrics(case_name):
    """Load parenchymal metrics JSON for a specific case"""
    json_path = RESULTS_ROOT / case_name / "step5_parenchymal_metrics" / "parenchymal_metrics.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def extract_patient_id(case_name):
    """Extract patient ID from case name"""
    return case_name.replace("_gaussian", "")

def leave_one_out_predict(df, feature_name, target_name):
    """Leave-one-out cross-validation for single-feature prediction"""
    
    valid_data = df[[feature_name, target_name]].dropna()
    if len(valid_data) < 5:
        return None
    
    X = valid_data[feature_name].values.reshape(-1, 1)
    y = valid_data[target_name].values
    
    predictions, actuals, errors = [], [], []
    
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i].reshape(1, -1)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        
        predictions.append(y_pred)
        actuals.append(y[i])
        errors.append(y_pred - y[i])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.array(errors)
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    pearson_r, pearson_p = pearsonr(actuals, predictions)
    
    return {
        'feature': feature_name,
        'target': target_name,
        'n_samples': len(X),
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }

def create_single_feature_plots(df_quality, features, output_dir):
    """
    Create plots for each feature showing predictions for:
    1) FVC@week0
    2) FVC@week52  
    3) %FVC drop at 1year
    Includes correlation plots and Bland-Altman plots
    """
    plots_dir = output_dir / "single_feature_plots"
    plots_dir.mkdir(exist_ok=True)
    
    targets = ['FVC_percent_week0', 'FVC_percent_week52', 'FVC_drop_percent']
    
    all_results = []
    
    for feature in features:
        if feature not in df_quality.columns:
            continue
            
        print(f"\n{'='*60}")
        print(f"Analyzing feature: {feature}")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Single-Feature Prediction: {feature}', fontsize=16, fontweight='bold')
        
        for col, target in enumerate(targets):
            if target not in df_quality.columns:
                continue
                
            result = leave_one_out_predict(df_quality, feature, target)
            if result is None or result['n_samples'] < 5:
                continue
                
            all_results.append(result)
            
            predictions = result['predictions']
            actuals = result['actuals']
            errors = result['errors']
            
            # Row 1: Correlation plots
            ax_corr = axes[0, col]
            
            # Scatter plot
            ax_corr.scatter(actuals, predictions, alpha=0.7, s=60, 
                           edgecolors='black', linewidth=0.5, color='steelblue')
            
            # Perfect prediction line
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            ax_corr.plot([min_val, max_val], [min_val, max_val], 
                        'r--', alpha=0.8, linewidth=2, label='Perfect')
            
            # Regression line
            z = np.polyfit(actuals, predictions, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val, max_val, 100)
            ax_corr.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Regression')
            
            ax_corr.set_xlabel(f'Actual {target}', fontsize=12)
            ax_corr.set_ylabel(f'Predicted {target}', fontsize=12)
            ax_corr.set_title(f'{target}\nR² = {result["r2"]:.3f}, MAE = {result["mae"]:.2f}', 
                            fontsize=12, fontweight='bold')
            ax_corr.legend(loc='best', fontsize=10)
            ax_corr.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = f'n = {result["n_samples"]}\nMAE = {result["mae"]:.2f}\nRMSE = {result["rmse"]:.2f}'
            ax_corr.text(0.05, 0.95, metrics_text, transform=ax_corr.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Row 2: Bland-Altman plots
            ax_ba = axes[1, col]
            
            # Calculate mean and difference
            means = (actuals + predictions) / 2
            differences = predictions - actuals
            
            # Scatter plot
            ax_ba.scatter(means, differences, alpha=0.7, s=60, 
                         edgecolors='black', linewidth=0.5, color='coral')
            
            # Mean difference line
            mean_diff = np.mean(differences)
            ax_ba.axhline(y=mean_diff, color='blue', linestyle='-', linewidth=2, 
                         label=f'Mean diff: {mean_diff:.2f}')
            
            # ±1.96 SD lines
            sd_diff = np.std(differences)
            upper_limit = mean_diff + 1.96 * sd_diff
            lower_limit = mean_diff - 1.96 * sd_diff
            
            ax_ba.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5,
                         label=f'+1.96 SD: {upper_limit:.2f}')
            ax_ba.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1.5,
                         label=f'-1.96 SD: {lower_limit:.2f}')
            
            ax_ba.set_xlabel('Mean of Actual and Predicted', fontsize=12)
            ax_ba.set_ylabel('Predicted - Actual', fontsize=12)
            ax_ba.set_title(f'Bland-Altman: {target}', fontsize=12, fontweight='bold')
            ax_ba.legend(loc='best', fontsize=9)
            ax_ba.grid(True, alpha=0.3)
            
            # Print results
            print(f"  {target}: R²={result['r2']:.3f}, MAE={result['mae']:.2f}, n={result['n_samples']}")
        
        plt.tight_layout()
        plot_path = plots_dir / f"{feature}_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save summary of results
    if all_results:
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Feature': result['feature'],
                'Target': result['target'],
                'n_samples': result['n_samples'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'Pearson_r': result['pearson_r'],
                'Pearson_p': result['pearson_p']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values(['Feature', 'Target'])
        df_summary.to_csv(output_dir / "single_feature_predictions_summary.csv", index=False)
        
        print(f"\n✓ Saved prediction summary: single_feature_predictions_summary.csv")
    
    return all_results

# ============================================================================
# NEW BALANCED INTERPOLATION FUNCTIONS
# ============================================================================

def interpolate_fvc_percent_balanced(patient_data, target_week, week_type='week0'):
    """
    BALANCED INTERPOLATION - MORE FLEXIBLE THAN NEW, MORE CONTROLLED THAN OLD
    
    For WEEK 0:
    - Preferred window: [-5, 10] weeks → use nearest (quality based on distance)
    - Extended window: [15, 30] weeks → linear regression if ≥2 points
    - Beyond 30 weeks: no value (too far)
    
    For WEEK 52:
    - Preferred window: [40, 65] weeks → use nearest (quality based on distance)
    - Extended cases: linear regression if measurements are reasonable
    """
    patient_data = patient_data.sort_values('Weeks').reset_index(drop=True)
    
    if len(patient_data) == 0:
        return {
            'value': np.nan,
            'actual_week': np.nan,
            'method': 'no_data',
            'quality': 'failed',
            'n_points_used': 0,
            'distance': np.nan
        }
    
    if week_type == 'week0':
        # ============ WEEK 0 LOGIC ============
        # 1. Check for measurements in PREFERRED window [-5, 15]
        in_pref_window = patient_data[
            (patient_data['Weeks'] >= -5) & (patient_data['Weeks'] <= 10)
        ]
        
        if len(in_pref_window) > 0:
            # Use nearest measurement in preferred window
            distances = abs(in_pref_window['Weeks'] - target_week)
            closest_idx = distances.idxmin()
            closest = in_pref_window.loc[closest_idx]
            min_distance = distances.min()
            
            # Quality based on distance
            if min_distance <= 3:
                quality = 'high'
            elif min_distance <= 8:
                quality = 'medium'
            elif min_distance <= 10:
                quality = 'low'  # Still included but marked as low
            else:
                quality = 'very_low'  # Excluded from quality-filtered
            
            return {
                'value': closest['Percent'],
                'actual_week': closest['Weeks'],
                'method': 'nearest_pref_window',
                'quality': quality,
                'n_points_used': 1,
                'distance': min_distance
            }
        
        # 2. Check for measurements in EXTENDED window [10, 30]
        in_ext_window = patient_data[
            (patient_data['Weeks'] >= 10) & (patient_data['Weeks'] <= 30)
        ]
        
        if len(in_ext_window) >= 2:
            # Linear regression on extended window points
            weeks = in_ext_window['Weeks'].values
            percents = in_ext_window['Percent'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
            estimated_value = slope * target_week + intercept
            
            # Quality assessment - more lenient than new script
            avg_distance = np.mean(abs(weeks - target_week))
            
            if len(in_ext_window) >= 3 and abs(r_value) > 0.6 and avg_distance <= 20:
                quality = 'medium'
            elif len(in_ext_window) >= 2 and avg_distance <= 25:
                quality = 'low'
            else:
                quality = 'very_low'
            
            return {
                'value': estimated_value,
                'actual_week': np.mean(weeks),
                'method': 'regression_ext_window',
                'quality': quality,
                'n_points_used': len(in_ext_window),
                'r_value': r_value,
                'distance': avg_distance
            }
        
        # 3. Last resort: Check original logic for backward compatibility
        # If we have at least 2 measurements anywhere, try regression
        if len(patient_data) >= 2:
            weeks = patient_data['Weeks'].values
            percents = patient_data['Percent'].values
            
            # Only proceed if max week ≤ 40 (not too far)
            if weeks.max() <= 40:
                slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
                estimated_value = slope * target_week + intercept
                
                # Quality will be low or very_low
                avg_distance = np.mean(abs(weeks - target_week))
                
                if avg_distance <= 30 and abs(r_value) > 0.5:
                    quality = 'low'
                else:
                    quality = 'very_low'
                
                return {
                    'value': estimated_value,
                    'actual_week': np.mean(weeks),
                    'method': 'regression_last_resort',
                    'quality': quality,
                    'n_points_used': len(patient_data),
                    'r_value': r_value,
                    'distance': avg_distance
                }
        
        # No suitable data
        return {
            'value': np.nan,
            'actual_week': np.nan,
            'method': 'no_suitable_data',
            'quality': 'failed',
            'n_points_used': 0,
            'distance': np.nan
        }
    
    elif week_type == 'week52':
        # ============ WEEK 52 LOGIC ============
        # 1. Check for measurements in PREFERRED window [40, 65]
        in_pref_window = patient_data[
            (patient_data['Weeks'] >= 40) & (patient_data['Weeks'] <= 65)
        ]
        
        if len(in_pref_window) > 0:
            # Use nearest measurement in preferred window
            distances = abs(in_pref_window['Weeks'] - target_week)
            closest_idx = distances.idxmin()
            closest = in_pref_window.loc[closest_idx]
            min_distance = distances.min()
            
            # Quality based on distance
            if min_distance <= 4:
                quality = 'high'
            elif min_distance <= 10:
                quality = 'medium'
            elif min_distance <= 10:
                quality = 'low'  # Still included but marked as low
            else:
                quality = 'very_low'  # Excluded from quality-filtered
            
            return {
                'value': closest['Percent'],
                'actual_week': closest['Weeks'],
                'method': 'nearest_pref_window',
                'quality': quality,
                'n_points_used': 1,
                'distance': min_distance
            }
        
        # 2. For week 52, allow regression more liberally
        if len(patient_data) >= 2:
            weeks = patient_data['Weeks'].values
            percents = patient_data['Percent'].values
            
            # Only proceed if we have measurements that make sense
            # i.e., at least some measurements after week 20 OR ≥3 measurements
            if weeks.max() >= 20 or len(patient_data) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
                estimated_value = slope * target_week + intercept
                
                # Calculate extrapolation distance
                max_week = weeks.max()
                extrapolation_distance = target_week - max_week if target_week > max_week else 0
                avg_distance = np.mean(abs(weeks - target_week))
                
                # Quality assessment
                if len(patient_data) >= 3 and abs(r_value) > 0.5 and extrapolation_distance <= 25:
                    quality = 'medium'
                elif len(patient_data) >= 2 and extrapolation_distance <= 35:
                    quality = 'low'
                else:
                    quality = 'very_low'
                
                # Determine method
                if target_week > max_week:
                    method = 'extrapolation'
                elif target_week < weeks.min():
                    method = 'backward_extrapolation'
                else:
                    method = 'interpolation'
                
                return {
                    'value': estimated_value,
                    'actual_week': np.mean(weeks),
                    'method': method,
                    'quality': quality,
                    'n_points_used': len(patient_data),
                    'r_value': r_value,
                    'distance': avg_distance,
                    'extrapolation_distance': extrapolation_distance
                }
        
        # No suitable data
        return {
            'value': np.nan,
            'actual_week': np.nan,
            'method': 'no_suitable_data',
            'quality': 'failed',
            'n_points_used': 0,
            'distance': np.nan
        }
    
    else:
        raise ValueError(f"Invalid week_type: {week_type}. Use 'week0' or 'week52'")

def create_interpolated_fvc_dataset_balanced(clinical_data):    
    results = []
    
    patient_ids = clinical_data['Patient'].unique()
    print(f"\nProcessing {len(patient_ids)} patients with BALANCED LOGIC...")
    print("Week 0: Preferred [-5, 15], Extended [15, 30], Last resort ≤40 weeks")
    print("Week 52: Preferred [40, 65], Regression if measurements reasonable")
    
    stats_week0 = {'pref_window': 0, 'ext_window': 0, 'last_resort': 0, 'failed': 0}
    stats_week52 = {'pref_window': 0, 'regression': 0, 'failed': 0}
    
    for patient_id in patient_ids:
        patient_data = clinical_data[clinical_data['Patient'] == patient_id]
        demographics = patient_data.iloc[0]
        
        # Week 0 interpolation
        week0_result = interpolate_fvc_percent_balanced(
            patient_data, 
            target_week=0,
            week_type='week0'
        )
        
        # Week 52 interpolation
        week52_result = interpolate_fvc_percent_balanced(
            patient_data,
            target_week=52,
            week_type='week52'
        )
        
        # Track statistics
        if week0_result['method'] == 'nearest_pref_window':
            stats_week0['pref_window'] += 1
        elif week0_result['method'] == 'regression_ext_window':
            stats_week0['ext_window'] += 1
        elif week0_result['method'] == 'regression_last_resort':
            stats_week0['last_resort'] += 1
        else:
            stats_week0['failed'] += 1
            
        if week52_result['method'] == 'nearest_pref_window':
            stats_week52['pref_window'] += 1
        elif week52_result['method'] in ['extrapolation', 'interpolation', 'backward_extrapolation']:
            stats_week52['regression'] += 1
        else:
            stats_week52['failed'] += 1
        
        # Calculate drop
        if not np.isnan(week0_result['value']) and not np.isnan(week52_result['value']):
            fvc_drop_absolute = week0_result['value'] - week52_result['value']
            fvc_drop_percent = (fvc_drop_absolute / week0_result['value']) * 100 if week0_result['value'] > 0 else np.nan
        else:
            fvc_drop_absolute = np.nan
            fvc_drop_percent = np.nan
        
        results.append({
            'Patient': patient_id,
            'Age': demographics['Age'],
            'Sex': demographics['Sex'],
            'SmokingStatus': demographics['SmokingStatus'],
            
            # Week 0 data
            'FVC_percent_week0': week0_result['value'],
            'week0_actual_week': week0_result['actual_week'],
            'week0_method': week0_result['method'],
            'week0_quality': week0_result['quality'],
            'week0_distance': week0_result['distance'],
            'week0_n_points': week0_result['n_points_used'],
            
            # Week 52 data
            'FVC_percent_week52': week52_result['value'],
            'week52_actual_week': week52_result['actual_week'],
            'week52_method': week52_result['method'],
            'week52_quality': week52_result['quality'],
            'week52_distance': week52_result['distance'],
            'week52_n_points': week52_result['n_points_used'],
            
            # Calculated values
            'FVC_drop_absolute': fvc_drop_absolute,
            'FVC_drop_percent': fvc_drop_percent,
            
            # Data completeness
            'n_measurements': len(patient_data)
        })
    
    df = pd.DataFrame(results)
    
    # Report statistics
    print(f"\n{'='*60}")
    print("BALANCED INTERPOLATION STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nWEEK 0 Results (total {len(df)} patients):")
    print(f"  Preferred window [-5, 15]: {stats_week0['pref_window']} patients")
    print(f"  Extended window [15, 30]: {stats_week0['ext_window']} patients")
    print(f"  Last resort (≤40 weeks): {stats_week0['last_resort']} patients")
    print(f"  Failed/no data: {stats_week0['failed']} patients")
    
    print(f"\nWEEK 52 Results (total {len(df)} patients):")
    print(f"  Preferred window [40, 65]: {stats_week52['pref_window']} patients")
    print(f"  Regression/other: {stats_week52['regression']} patients")
    print(f"  Failed/no data: {stats_week52['failed']} patients")
    
    complete = df.dropna(subset=['FVC_percent_week0', 'FVC_percent_week52'])
    print(f"\nComplete cases (both week0 and week52): {len(complete)} ({100*len(complete)/len(df):.1f}%)")
    
    # Quality distribution
    week0_quality_counts = df['week0_quality'].value_counts()
    week52_quality_counts = df['week52_quality'].value_counts()
    
    print(f"\nWeek 0 Quality distribution:")
    for quality in ['high', 'medium', 'low', 'very_low', 'failed']:
        count = week0_quality_counts.get(quality, 0)
        print(f"  {quality}: {count} ({100*count/len(df):.1f}%)")
    
    print(f"\nWeek 52 Quality distribution:")
    for quality in ['high', 'medium', 'low', 'very_low', 'failed']:
        count = week52_quality_counts.get(quality, 0)
        print(f"  {quality}: {count} ({100*count/len(df):.1f}%)")
    
    # Show how many would be included in quality-filtered dataset
    high_medium_count = len(df[
        (df['week0_quality'].isin(['high', 'medium'])) & 
        (df['week52_quality'].isin(['high', 'medium']))
    ])
    
    high_medium_low_count = len(df[
        (df['week0_quality'].isin(['high', 'medium', 'low'])) & 
        (df['week52_quality'].isin(['high', 'medium', 'low']))
    ])
    
    print(f"\nQuality-filtered dataset sizes:")
    print(f"  High/Medium only: {high_medium_count} patients")
    print(f"  High/Medium/Low: {high_medium_low_count} patients")
    
    return df

def integrate_airway_and_fvc_balanced(reliable_cases, fvc_df, quality_level='high_medium'):
    """
    Integrate airway metrics with interpolated FVC data.
    
    Args:
        quality_level: 'high_medium' (default), 'high_medium_low', or 'all'
    """
    print(f"\n{'='*80}")
    print(f"INTEGRATING AIRWAY METRICS WITH FVC DATA")
    print(f"Quality level: {quality_level}")
    print(f"{'='*80}")
    
    rows = []
    
    for idx, case_row in reliable_cases.iterrows():
        case_name = case_row['case']
        patient_id = extract_patient_id(case_name)
        
        # Find FVC data
        patient_fvc = fvc_df[fvc_df['Patient'] == patient_id]
        if len(patient_fvc) == 0:
            continue
        
        patient_fvc = patient_fvc.iloc[0]
        
        # Load metrics
        advanced = load_advanced_metrics(case_name)
        if advanced is None:
            continue
        
        parenchymal = load_parenchymal_metrics(case_name)
        
        # Create row
        row = {
            'patient': patient_id,
            'case': case_name,
            
            # Demographics
            'Age': patient_fvc['Age'],
            'Sex': patient_fvc['Sex'],
            'SmokingStatus': patient_fvc['SmokingStatus'],
            
            # FVC targets
            'FVC_percent_week0': patient_fvc['FVC_percent_week0'],
            'FVC_percent_week52': patient_fvc['FVC_percent_week52'],
            'FVC_drop_absolute': patient_fvc['FVC_drop_absolute'],
            'FVC_drop_percent': patient_fvc['FVC_drop_percent'],
            
            # Quality info
            'week0_quality': patient_fvc['week0_quality'],
            'week52_quality': patient_fvc['week52_quality'],
            'week0_method': patient_fvc['week0_method'],
            'week52_method': patient_fvc['week52_method'],
            
            # Airway metrics
            'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
            'peripheral_branch_density': advanced.get('peripheral_branch_density'),
            'mean_peripheral_diameter_mm': advanced.get('mean_peripheral_diameter_mm'),
            'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
        }
        
        # Add parenchymal metrics
        if parenchymal is not None:
            row.update({
                'mean_lung_density_HU': parenchymal.get('mean_lung_density_HU'),
                'histogram_entropy': parenchymal.get('histogram_entropy'),
            })
        
        rows.append(row)
    
    df_all = pd.DataFrame(rows)
    
    # Filter based on quality level
    if quality_level == 'high_medium':
        df_filtered = df_all[
            (df_all['week0_quality'].isin(['high', 'medium'])) & 
            (df_all['week52_quality'].isin(['high', 'medium']))
        ].copy()
    elif quality_level == 'high_medium_low':
        df_filtered = df_all[
            (df_all['week0_quality'].isin(['high', 'medium', 'low'])) & 
            (df_all['week52_quality'].isin(['high', 'medium', 'low']))
        ].copy()
    else:  # 'all'
        df_filtered = df_all.copy()
    
    print(f"\nIntegrated dataset:")
    print(f"  All patients: {len(df_all)}")
    print(f"  Filtered ({quality_level}): {len(df_filtered)}")
    
    return df_all, df_filtered

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("SINGLE-FEATURE FVC PREDICTION ANALYSIS - BALANCED APPROACH")
    print("="*80)
    print("Week 0: Preferred [-5, 15], Extended [15, 30], Last resort ≤40")
    print("Week 52: Preferred [40, 65], Regression if reasonable")
    print("Quality levels: high/medium (strict), high/medium/low (balanced)")
    print("="*80)
    
    # Load data
    clinical = load_clinical_data()
    
    # Create interpolated dataset with BALANCED LOGIC
    fvc_df = create_interpolated_fvc_dataset_balanced(clinical)
    
    # Save interpolated data
    fvc_output = OUTPUT_DIR / "interpolated_fvc_data_BALANCED.csv"
    fvc_df.to_csv(fvc_output, index=False)
    print(f"\n✓ Interpolated FVC data saved: {fvc_output}")
    
    # Load airway metrics
    reliable = load_validation_results()
    
    # Create datasets with DIFFERENT QUALITY LEVELS
    print(f"\n{'='*80}")
    print("CREATING DATASETS WITH DIFFERENT QUALITY LEVELS")
    print(f"{'='*80}")
    
    # 1. High/Medium only (strict)
    df_all_strict, df_strict = integrate_airway_and_fvc_balanced(reliable, fvc_df, 'high_medium')
    
    # 2. High/Medium/Low (balanced - recommended)
    df_all_balanced, df_balanced = integrate_airway_and_fvc_balanced(reliable, fvc_df, 'high_medium_low')
    
    # 3. All data (for comparison)
    df_all_all, df_all_data = integrate_airway_and_fvc_balanced(reliable, fvc_df, 'all')
    
    # Save all datasets
    df_strict.to_csv(OUTPUT_DIR / "quality_strict_dataset.csv", index=False)
    df_balanced.to_csv(OUTPUT_DIR / "quality_balanced_dataset.csv", index=False)
    df_all_data.to_csv(OUTPUT_DIR / "all_data_dataset.csv", index=False)
    
    print(f"\n✓ Saved datasets:")
    print(f"  Strict (high/medium only): {len(df_strict)} patients")
    print(f"  Balanced (high/medium/low): {len(df_balanced)} patients")
    print(f"  All data: {len(df_all_data)} patients")
    
    # Define features to analyze
    features = [
        'mean_peripheral_branch_volume_mm3',
        'peripheral_branch_density',
        'mean_peripheral_diameter_mm',
        'central_to_peripheral_diameter_ratio',
        'mean_lung_density_HU',
        'histogram_entropy'
    ]
    
    # Create single-feature plots using BALANCED dataset (recommended)
    print(f"\n{'='*80}")
    print("CREATING SINGLE-FEATURE PREDICTION PLOTS")
    print(f"Using BALANCED dataset (high/medium/low quality)")
    print(f"{'='*80}")
    
    all_results = create_single_feature_plots(df_balanced, features, OUTPUT_DIR)
    
    # Also create plots for strict dataset for comparison
    print(f"\n{'='*80}")
    print("ADDITIONAL: Creating plots for STRICT dataset")
    print(f"{'='*80}")
    
    if len(df_strict) >= 10:  # Only if we have enough patients
        strict_dir = OUTPUT_DIR / "strict_analysis"
        strict_dir.mkdir(exist_ok=True)
        strict_results = create_single_feature_plots(df_strict, features, strict_dir)
    
    # Final summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE (BALANCED APPROACH)")
    print(f"{'='*80}")
    print(f"✓ Main balanced dataset: {len(df_balanced)} patients")
    print(f"✓ Strict dataset: {len(df_strict)} patients (if >10)")
    print(f"✓ All datasets saved in: {OUTPUT_DIR}")
    
    # Show best performing features from balanced dataset
    if all_results:
        print(f"\nTOP PERFORMING FEATURES (balanced dataset):")
        
        # Group by feature and calculate average R²
        feature_performance = {}
        for result in all_results:
            feature = result['feature']
            if feature not in feature_performance:
                feature_performance[feature] = []
            feature_performance[feature].append(result['r2'])
        
        # Calculate and display averages
        avg_performance = []
        for feature, r2_values in feature_performance.items():
            avg_r2 = np.mean(r2_values)
            avg_performance.append((feature, avg_r2, len(r2_values)))
        
        # Sort by average R²
        avg_performance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, avg_r2, n_targets in avg_performance[:5]:  # Top 5
            print(f"  {feature:35s}: Avg R² = {avg_r2:.3f} (across {n_targets} targets)")

if __name__ == "__main__":
    main()