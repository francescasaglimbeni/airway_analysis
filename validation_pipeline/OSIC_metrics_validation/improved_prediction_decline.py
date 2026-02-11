"""
IMPROVED FVC PREDICTION - CON DECLINO DIRETTO
Struttura unificata: tutti i plot in un'unica directory
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
plt.rcParams['font.size'] = 11

# ============================================================================
# PATHS
# ============================================================================
VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation.csv")
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_combined")
TRAIN_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\train.csv")
TEST_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\test.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\improved_prediction")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING FUNCTIONS
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

# ============================================================================
# BALANCED INTERPOLATION FUNCTIONS
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
                quality = 'low'
            else:
                quality = 'very_low'
            
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
        if len(patient_data) >= 2:
            weeks = patient_data['Weeks'].values
            percents = patient_data['Percent'].values
            
            if weeks.max() <= 40:
                slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
                estimated_value = slope * target_week + intercept
                
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
            distances = abs(in_pref_window['Weeks'] - target_week)
            closest_idx = distances.idxmin()
            closest = in_pref_window.loc[closest_idx]
            min_distance = distances.min()
            
            if min_distance <= 4:
                quality = 'high'
            elif min_distance <= 10:
                quality = 'medium'
            elif min_distance <= 10:
                quality = 'low'
            else:
                quality = 'very_low'
            
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
            
            if weeks.max() >= 20 or len(patient_data) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
                estimated_value = slope * target_week + intercept
                
                max_week = weeks.max()
                extrapolation_distance = target_week - max_week if target_week > max_week else 0
                avg_distance = np.mean(abs(weeks - target_week))
                
                if len(patient_data) >= 3 and abs(r_value) > 0.5 and extrapolation_distance <= 25:
                    quality = 'medium'
                elif len(patient_data) >= 2 and extrapolation_distance <= 35:
                    quality = 'low'
                else:
                    quality = 'very_low'
                
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
        
        # Calculate drop
        if not np.isnan(week0_result['value']) and not np.isnan(week52_result['value']):
            fvc_drop_percent = week0_result['value'] - week52_result['value']
        else:
            fvc_drop_percent = np.nan
        
        results.append({
            'Patient': patient_id,
            'Age': demographics['Age'],
            'Sex': demographics['Sex'],
            'SmokingStatus': demographics['SmokingStatus'],
            
            # Week 0 data
            'FVC_percent_week0': week0_result['value'],
            'week0_quality': week0_result['quality'],
            'week0_method': week0_result['method'],
            'week0_distance': week0_result['distance'],
            
            # Week 52 data
            'FVC_percent_week52': week52_result['value'],
            'week52_quality': week52_result['quality'],
            'week52_method': week52_result['method'],
            'week52_distance': week52_result['distance'],
            
            # Calculated drop
            'FVC_drop_percent': fvc_drop_percent,
            
            # Data completeness
            'n_measurements': len(patient_data)
        })
    
    df = pd.DataFrame(results)
    
    # Print quick summary
    complete = df.dropna(subset=['FVC_percent_week0', 'FVC_percent_week52'])
    print(f"  Complete cases (both week0 and week52): {len(complete)}/{len(df)} ({100*len(complete)/len(df):.1f}%)")
    
    return df

def integrate_airway_and_fvc_balanced(reliable_cases, fvc_df):
    """
    Integrate airway metrics with interpolated FVC data.
    Uses balanced quality (high/medium/low) by default.
    """
    print(f"\nIntegrating airway metrics with FVC data...")
    
    rows = []
    
    for idx, case_row in reliable_cases.iterrows():
        case_name = case_row['case']
        patient_id = extract_patient_id(case_name)
        
        # Find FVC data
        patient_fvc = fvc_df[fvc_df['Patient'] == patient_id]
        if len(patient_fvc) == 0:
            continue
        
        patient_fvc = patient_fvc.iloc[0]
        
        # Apply quality filter (high/medium/low only)
        if patient_fvc['week0_quality'] not in ['high', 'medium', 'low'] or \
           patient_fvc['week52_quality'] not in ['high', 'medium', 'low']:
            continue
        
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
            'FVC_drop_percent': patient_fvc['FVC_drop_percent'],
            
            # Quality info
            'week0_quality': patient_fvc['week0_quality'],
            'week52_quality': patient_fvc['week52_quality'],
            
            # Airway metrics
            'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
            'peripheral_branch_density': advanced.get('peripheral_branch_density'),
            'mean_peripheral_diameter_mm': advanced.get('mean_peripheral_diameter_mm'),
            'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
            
            # Parenchymal metrics
            'mean_lung_density_HU': parenchymal.get('mean_lung_density_HU') if parenchymal else np.nan,
            'histogram_entropy': parenchymal.get('histogram_entropy') if parenchymal else np.nan,
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"  Integrated dataset: {len(df)} patients")
    
    return df

# ============================================================================
# DIRECT DECLINE CALCULATION FUNCTIONS
# ============================================================================

def calculate_annual_fvc_decline(patient_data, min_measurements=3):
    """
    Calcola il tasso di declino annuale di FVC direttamente da tutti i 
    timepoint disponibili.
    """
    patient_data = patient_data.sort_values('Weeks').reset_index(drop=True)
    
    if len(patient_data) < min_measurements:
        return {
            'annual_decline_percent': np.nan,
            'slope_percent_per_week': np.nan,
            'r_value': np.nan,
            'n_measurements': len(patient_data),
            'quality': 'insufficient_data',
            'timespan_weeks': np.nan
        }
    
    weeks = patient_data['Weeks'].values
    percents = patient_data['Percent'].values
    
    # Calcola regressione lineare
    slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
    
    # Converti pendenza in declino annuale (52 settimane)
    annual_decline = -slope * 52  # Positivo = declino
    
    # Calcola span temporale
    timespan = weeks.max() - weeks.min()
    
    # Valutazione qualità
    quality_score = 0
    
    if len(patient_data) >= 4:
        quality_score += 2
    elif len(patient_data) >= 3:
        quality_score += 1
    
    if timespan >= 52:
        quality_score += 2
    elif timespan >= 26:
        quality_score += 1
    
    if abs(r_value) > 0.7:
        quality_score += 2
    elif abs(r_value) > 0.5:
        quality_score += 1
    
    if quality_score >= 4:
        quality = 'high'
    elif quality_score >= 2:
        quality = 'medium'
    elif quality_score >= 1:
        quality = 'low'
    else:
        quality = 'very_low'
    
    return {
        'annual_decline_percent': annual_decline,
        'slope_percent_per_week': slope,
        'r_value': r_value,
        'p_value': p_value,
        'n_measurements': len(patient_data),
        'timespan_weeks': timespan,
        'quality': quality,
        'quality_score': quality_score
    }

def create_dataset_with_direct_decline(clinical_data, min_measurements=3):
    """
    Crea un dataset con il declino annuale calcolato DIRETTAMENTE.
    """
    print(f"\nCalculating direct annual decline...")
    
    results = []
    patient_ids = clinical_data['Patient'].unique()
    
    for patient_id in patient_ids:
        patient_data = clinical_data[clinical_data['Patient'] == patient_id]
        demographics = patient_data.iloc[0]
        
        decline_info = calculate_annual_fvc_decline(patient_data, min_measurements)
        
        results.append({
            'Patient': patient_id,
            'Age': demographics['Age'],
            'Sex': demographics['Sex'],
            'SmokingStatus': demographics['SmokingStatus'],
            
            # Tasso di declino diretto
            'FVC_annual_decline_direct': decline_info['annual_decline_percent'],
            'decline_r_value': decline_info['r_value'],
            'decline_quality': decline_info['quality'],
            'decline_n_measurements': decline_info['n_measurements'],
            'decline_timespan_weeks': decline_info['timespan_weeks']
        })
    
    df = pd.DataFrame(results)
    
    valid = df['FVC_annual_decline_direct'].notna().sum()
    print(f"  Direct decline calculated for: {valid}/{len(df)} patients ({100*valid/len(df):.1f}%)")
    
    return df

def integrate_airway_and_direct_decline(reliable_cases, fvc_df, decline_df):
    """
    Integra le metriche delle airways con il declino diretto.
    Usa qualità medium+ per il declino.
    """
    print(f"\nIntegrating airway metrics with direct decline...")
    
    rows = []
    
    for idx, case_row in reliable_cases.iterrows():
        case_name = case_row['case']
        patient_id = extract_patient_id(case_name)
        
        # Trova FVC interpolato (per demografiche)
        patient_fvc = fvc_df[fvc_df['Patient'] == patient_id]
        if len(patient_fvc) == 0:
            continue
        
        # Trova declino diretto
        patient_decline = decline_df[decline_df['Patient'] == patient_id]
        if len(patient_decline) == 0:
            continue
        
        patient_fvc = patient_fvc.iloc[0]
        patient_decline = patient_decline.iloc[0]
        
        # Filtra per qualità del declino (medium+)
        if patient_decline['decline_quality'] not in ['high', 'medium']:
            continue
        
        # Carica metriche
        advanced = load_advanced_metrics(case_name)
        if advanced is None:
            continue
        
        parenchymal = load_parenchymal_metrics(case_name)
        
        # Costruisci row
        row = {
            'patient': patient_id,
            'case': case_name,
            
            # Demographics
            'Age': patient_fvc['Age'],
            'Sex': patient_fvc['Sex'],
            'SmokingStatus': patient_fvc['SmokingStatus'],
            
            # FVC declino DIRETTO
            'FVC_annual_decline_direct': patient_decline['FVC_annual_decline_direct'],
            'decline_quality': patient_decline['decline_quality'],
            'decline_n_measurements': patient_decline['decline_n_measurements'],
            'decline_timespan_weeks': patient_decline['decline_timespan_weeks'],
            'decline_r_value': patient_decline['decline_r_value'],
            
            # Airway metrics
            'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
            'peripheral_branch_density': advanced.get('peripheral_branch_density'),
            'mean_peripheral_diameter_mm': advanced.get('mean_peripheral_diameter_mm'),
            'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
            
            # Parenchymal metrics
            'mean_lung_density_HU': parenchymal.get('mean_lung_density_HU') if parenchymal else np.nan,
            'histogram_entropy': parenchymal.get('histogram_entropy') if parenchymal else np.nan,
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"  Integrated direct decline dataset: {len(df)} patients")
    
    return df

# ============================================================================
# PLOTTING FUNCTION - UNIFIED 2x2 GRID
# ============================================================================

def plot_feature_predictions_unified(df_traditional, df_direct, feature, output_dir):
    """
    Crea un unico plot 2x2 con tutti e 4 i target:
    - FVC week0
    - FVC week52
    - Drop tradizionale (week0 - week52)
    - Declino diretto (annuale)
    """
    
    # Calcola predizioni per i 4 target
    result_week0 = leave_one_out_predict(df_traditional, feature, 'FVC_percent_week0')
    result_week52 = leave_one_out_predict(df_traditional, feature, 'FVC_percent_week52')
    result_direct = leave_one_out_predict(df_direct, feature, 'FVC_annual_decline_direct')
    
    # Calcola drop tradizionale
    result_drop = None
    if result_week0 and result_week52:
        valid_data_week0 = df_traditional[[feature, 'FVC_percent_week0']].dropna()
        valid_data_week52 = df_traditional[[feature, 'FVC_percent_week52']].dropna()
        common_idx = valid_data_week0.index.intersection(valid_data_week52.index)
        
        if len(common_idx) >= 5:
            predictions = result_week0['predictions'] - result_week52['predictions']
            actuals = df_traditional.loc[common_idx, 'FVC_drop_percent'].values
            
            result_drop = {
                'feature': feature,
                'target': 'FVC_drop_percent',
                'n_samples': len(common_idx),
                'predictions': predictions,
                'actuals': actuals,
                'r2': r2_score(actuals, predictions),
                'mae': mean_absolute_error(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                'pearson_r': pearsonr(actuals, predictions)[0]
            }
    
    # Crea figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Prediction Performance: {feature}', fontsize=16, fontweight='bold')
    
    # 1. FVC Week 0
    if result_week0:
        ax = axes[0, 0]
        ax.scatter(result_week0['actuals'], result_week0['predictions'], 
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5, color='steelblue')
        
        min_val = min(result_week0['actuals'].min(), result_week0['predictions'].min())
        max_val = max(result_week0['actuals'].max(), result_week0['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Actual FVC Week 0 (% predicted)', fontsize=11)
        ax.set_ylabel('Predicted FVC Week 0 (% predicted)', fontsize=11)
        ax.set_title(f'FVC at Week 0\nR² = {result_week0["r2"]:.3f}, MAE = {result_week0["mae"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_week0["n_samples"]}\nPearson r = {result_week0["pearson_r"]:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 2. FVC Week 52
    if result_week52:
        ax = axes[0, 1]
        ax.scatter(result_week52['actuals'], result_week52['predictions'], 
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5, color='steelblue')
        
        min_val = min(result_week52['actuals'].min(), result_week52['predictions'].min())
        max_val = max(result_week52['actuals'].max(), result_week52['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Actual FVC Week 52 (% predicted)', fontsize=11)
        ax.set_ylabel('Predicted FVC Week 52 (% predicted)', fontsize=11)
        ax.set_title(f'FVC at Week 52\nR² = {result_week52["r2"]:.3f}, MAE = {result_week52["mae"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_week52["n_samples"]}\nPearson r = {result_week52["pearson_r"]:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 3. Drop Tradizionale
    if result_drop:
        ax = axes[1, 0]
        ax.scatter(result_drop['actuals'], result_drop['predictions'], 
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5, color='coral')
        
        min_val = min(result_drop['actuals'].min(), result_drop['predictions'].min())
        max_val = max(result_drop['actuals'].max(), result_drop['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Actual FVC Drop (% predicted)', fontsize=11)
        ax.set_ylabel('Predicted FVC Drop (% predicted)', fontsize=11)
        ax.set_title(f'Traditional Drop (Week0-Week52)\nR² = {result_drop["r2"]:.3f}, MAE = {result_drop["mae"]:.2f}%', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_drop["n_samples"]}\nPearson r = {result_drop["pearson_r"]:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Traditional Drop', fontsize=12, fontweight='bold')
    
    # 4. Declino Diretto
    if result_direct:
        ax = axes[1, 1]
        ax.scatter(result_direct['actuals'], result_direct['predictions'], 
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5, color='darkgreen')
        
        min_val = min(result_direct['actuals'].min(), result_direct['predictions'].min())
        max_val = max(result_direct['actuals'].max(), result_direct['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Actual Annual Decline (%/year)', fontsize=11)
        ax.set_ylabel('Predicted Annual Decline (%/year)', fontsize=11)
        ax.set_title(f'Direct Annual Decline\nR² = {result_direct["r2"]:.3f}, MAE = {result_direct["mae"]:.2f}%/year', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_direct["n_samples"]}\nPearson r = {result_direct["pearson_r"]:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Direct Annual Decline', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{feature}_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# SUMMARY FUNCTIONS
# ============================================================================

def create_unified_summary(df_traditional, df_direct, features, output_dir):
    """
    Crea un unico file CSV con tutte le performance predittive.
    """
    summary_data = []
    
    for feature in features:
        if feature not in df_traditional.columns:
            continue
        
        # Week 0
        result_week0 = leave_one_out_predict(df_traditional, feature, 'FVC_percent_week0')
        if result_week0:
            summary_data.append({
                'Feature': feature,
                'Target': 'FVC_week0',
                'Target_Description': 'FVC at baseline (% predicted)',
                'n_samples': result_week0['n_samples'],
                'R2': result_week0['r2'],
                'MAE': result_week0['mae'],
                'RMSE': result_week0['rmse'],
                'Pearson_r': result_week0['pearson_r'],
                'Pearson_p': result_week0['pearson_p']
            })
        
        # Week 52
        result_week52 = leave_one_out_predict(df_traditional, feature, 'FVC_percent_week52')
        if result_week52:
            summary_data.append({
                'Feature': feature,
                'Target': 'FVC_week52',
                'Target_Description': 'FVC at 1 year (% predicted)',
                'n_samples': result_week52['n_samples'],
                'R2': result_week52['r2'],
                'MAE': result_week52['mae'],
                'RMSE': result_week52['rmse'],
                'Pearson_r': result_week52['pearson_r'],
                'Pearson_p': result_week52['pearson_p']
            })
        
        # Traditional Drop
        result_week0_temp = leave_one_out_predict(df_traditional, feature, 'FVC_percent_week0')
        result_week52_temp = leave_one_out_predict(df_traditional, feature, 'FVC_percent_week52')
        
        if result_week0_temp and result_week52_temp:
            valid_data_week0 = df_traditional[[feature, 'FVC_percent_week0']].dropna()
            valid_data_week52 = df_traditional[[feature, 'FVC_percent_week52']].dropna()
            common_idx = valid_data_week0.index.intersection(valid_data_week52.index)
            
            if len(common_idx) >= 5:
                predictions = result_week0_temp['predictions'] - result_week52_temp['predictions']
                actuals = df_traditional.loc[common_idx, 'FVC_drop_percent'].values
                
                r2 = r2_score(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                pearson_r, pearson_p = pearsonr(actuals, predictions)
                
                summary_data.append({
                    'Feature': feature,
                    'Target': 'Drop_traditional',
                    'Target_Description': 'FVC drop (Week0 - Week52)',
                    'n_samples': len(common_idx),
                    'R2': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Pearson_r': pearson_r,
                    'Pearson_p': pearson_p
                })
        
        # Direct Decline
        if feature in df_direct.columns:
            result_direct = leave_one_out_predict(df_direct, feature, 'FVC_annual_decline_direct')
            if result_direct:
                summary_data.append({
                    'Feature': feature,
                    'Target': 'Decline_direct',
                    'Target_Description': 'Annual FVC decline (%/year)',
                    'n_samples': result_direct['n_samples'],
                    'R2': result_direct['r2'],
                    'MAE': result_direct['mae'],
                    'RMSE': result_direct['rmse'],
                    'Pearson_r': result_direct['pearson_r'],
                    'Pearson_p': result_direct['pearson_p']
                })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Riordina colonne
    df_summary = df_summary[['Feature', 'Target', 'Target_Description', 'n_samples', 
                             'R2', 'MAE', 'RMSE', 'Pearson_r', 'Pearson_p']]
    
    # Ordina per Feature e Target
    df_summary = df_summary.sort_values(['Feature', 'Target'])
    
    # Salva
    df_summary.to_csv(output_dir / "prediction_performance_summary.csv", index=False)
    
    print(f"\n✓ Saved unified summary: prediction_performance_summary.csv")
    
    # Stampa top features per declino diretto
    direct_summary = df_summary[df_summary['Target'] == 'Decline_direct'].sort_values('R2', ascending=False)
    
    if len(direct_summary) > 0:
        print(f"\n🏆 TOP FEATURES FOR DIRECT DECLINE:")
        print(f"{'Feature':<35} {'R²':<8} {'MAE':<10} {'n':<6}")
        print("-" * 60)
        for _, row in direct_summary.head(5).iterrows():
            print(f"{row['Feature'][:35]:<35} {row['R2']:.3f}    {row['MAE']:.2f}%/y   n={row['n_samples']}")
    
    return df_summary

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("FVC PREDICTION ANALYSIS - UNIFIED APPROACH")
    print("="*80)
    print("✓ 4 targets in unico plot: FVC0, FVC52, Drop tradizionale, Declino diretto")
    print("✓ Output organizzato in un'unica directory")
    print("="*80)
    
    # =========================================================================
    # 1. Load data
    # =========================================================================
    clinical = load_clinical_data()
    reliable = load_validation_results()
    
    # =========================================================================
    # 2. Create interpolated FVC dataset (balanced quality)
    # =========================================================================
    fvc_df = create_interpolated_fvc_dataset_balanced(clinical)
    fvc_df.to_csv(OUTPUT_DIR / "01_interpolated_fvc.csv", index=False)
    print(f"  Saved: 01_interpolated_fvc.csv")
    
    # =========================================================================
    # 3. Create traditional dataset (airway metrics + FVC)
    # =========================================================================
    df_traditional = integrate_airway_and_fvc_balanced(reliable, fvc_df)
    df_traditional.to_csv(OUTPUT_DIR / "02_dataset_traditional.csv", index=False)
    print(f"  Saved: 02_dataset_traditional.csv")
    
    # =========================================================================
    # 4. Create direct decline dataset
    # =========================================================================
    decline_df = create_dataset_with_direct_decline(clinical, min_measurements=3)
    decline_df.to_csv(OUTPUT_DIR / "03_direct_decline.csv", index=False)
    print(f"  Saved: 03_direct_decline.csv")
    
    # =========================================================================
    # 5. Create integrated direct decline dataset
    # =========================================================================
    df_direct = integrate_airway_and_direct_decline(reliable, fvc_df, decline_df)
    df_direct.to_csv(OUTPUT_DIR / "04_dataset_direct_decline.csv", index=False)
    print(f"  Saved: 04_dataset_direct_decline.csv")
    
    # =========================================================================
    # 6. Define features
    # =========================================================================
    features = [
        'mean_peripheral_branch_volume_mm3',
        'peripheral_branch_density',
        'mean_peripheral_diameter_mm',
        'central_to_peripheral_diameter_ratio',
        'mean_lung_density_HU',
        'histogram_entropy'
    ]
    
    # =========================================================================
    # 7. Create unified plots (2x2 grid with all 4 targets)
    # =========================================================================
    print(f"\n{'='*80}")
    print("CREATING UNIFIED PREDICTION PLOTS (2x2 grid)")
    print(f"{'='*80}")
    
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    for feature in features:
        if feature not in df_traditional.columns or feature not in df_direct.columns:
            continue
        
        print(f"  Plotting: {feature}")
        plot_feature_predictions_unified(df_traditional, df_direct, feature, plots_dir)
    
    print(f"\n✓ All plots saved in: {plots_dir}")
    
    # =========================================================================
    # 8. Create unified summary CSV
    # =========================================================================
    summary_df = create_unified_summary(df_traditional, df_direct, features, OUTPUT_DIR)
    
    # =========================================================================
    # 9. Create comparison view (pivot table per facile confronto)
    # =========================================================================
    pivot_df = summary_df.pivot(index='Feature', columns='Target', values=['R2', 'MAE', 'n_samples'])
    pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    pivot_df.to_csv(OUTPUT_DIR / "05_performance_comparison_pivot.csv", index=False)
    print(f"  Saved: 05_performance_comparison_pivot.csv")
    
    # =========================================================================
    # 10. Final summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"\n📊 DATASETS:")
    print(f"  01_interpolated_fvc.csv          - FVC interpolati (tutti i pazienti)")
    print(f"  02_dataset_traditional.csv       - Metriche + FVC0/FVC52/Drop (n={len(df_traditional)})")
    print(f"  03_direct_decline.csv            - Declino diretto (tutti i pazienti)")
    print(f"  04_dataset_direct_decline.csv    - Metriche + Declino diretto (n={len(df_direct)})")
    print(f"  05_performance_comparison_pivot.csv - Tabella pivot per confronto")
    print(f"\n📈 PREDICTION PERFORMANCE:")
    print(f"  prediction_performance_summary.csv - Tutti i risultati in un unico file")
    print(f"\n🖼️  PLOTS:")
    print(f"  plots/ - Tutti i plot 2x2 per ogni feature")
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()