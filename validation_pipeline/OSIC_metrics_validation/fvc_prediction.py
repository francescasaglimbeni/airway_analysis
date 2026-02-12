"""
UNIFIED FVC PREDICTION ANALYSIS - VERSION 2
Combines traditional approach (week0/week52/drop) with direct decline calculation
Creates UNIFIED dataset with BOTH targets + multiple quality levels
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

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
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\unified_prediction")

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

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

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
            
            # Quality assessment
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
        
        # 3. Last resort: regression on all points if max week ≤ 40
        if len(patient_data) >= 2:
            weeks = patient_data['Weeks'].values
            percents = patient_data['Percent'].values
            
            if weeks.max() <= 40:
                slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
                estimated_value = slope * target_week + intercept
                
                return {
                    'value': estimated_value,
                    'actual_week': np.mean(weeks),
                    'method': 'regression_last_resort',
                    'quality': 'low' if abs(r_value) > 0.5 else 'very_low',
                    'n_points_used': len(patient_data),
                    'r_value': r_value,
                    'distance': np.mean(abs(weeks - target_week))
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
            elif min_distance <= 8:
                quality = 'medium'
            elif min_distance <= 13:
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
                
                # Determine method and quality
                if target_week > weeks.max():
                    method = 'extrapolation'
                elif target_week < weeks.min():
                    method = 'backward_extrapolation'
                else:
                    method = 'interpolation'
                
                avg_distance = np.mean(abs(weeks - target_week))
                
                if len(patient_data) >= 3 and abs(r_value) > 0.6:
                    quality = 'medium'
                elif len(patient_data) >= 2:
                    quality = 'low'
                else:
                    quality = 'very_low'
                
                return {
                    'value': estimated_value,
                    'actual_week': np.mean(weeks),
                    'method': method,
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
    
    else:
        raise ValueError(f"Invalid week_type: {week_type}. Use 'week0' or 'week52'")

def create_interpolated_fvc_dataset_balanced(clinical_data):    
    results = []
    
    patient_ids = clinical_data['Patient'].unique()
    print(f"\nProcessing {len(patient_ids)} patients with BALANCED LOGIC...")
    
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
            fvc_drop_percent = fvc_drop_absolute
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
    
    return df

# ============================================================================
# DIRECT DECLINE CALCULATION FUNCTIONS
# ============================================================================

def calculate_annual_fvc_decline(patient_data, min_measurements=3):
    """
    Calculate annual FVC decline rate directly from all available timepoints.
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
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
    
    # Convert slope to annual decline (52 weeks)
    annual_decline = -slope * 52  # Positive = decline
    
    # Calculate timespan
    timespan = weeks.max() - weeks.min()
    
    # Quality assessment
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
    Create a dataset with DIRECT annual decline calculation.
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

# ============================================================================
# UNIFIED INTEGRATION FUNCTION
# ============================================================================

def integrate_airway_fvc_and_decline_unified(reliable_cases, fvc_df, decline_df, quality_filter='balanced'):
    """
    UNIFIED INTEGRATION - Combine airway metrics with BOTH traditional FVC and direct decline.
    
    Args:
        quality_filter: 'strict', 'balanced', or 'all'
            - 'strict': high/medium for FVC, high/medium for decline
            - 'balanced': high/medium/low for FVC, high/medium for decline
            - 'all': no quality filter
    """
    print(f"\nIntegrating airway metrics with FVC + Decline (quality={quality_filter})...")
    
    rows = []
    
    for idx, case_row in reliable_cases.iterrows():
        case_name = case_row['case']
        patient_id = extract_patient_id(case_name)
        
        # Find FVC data
        patient_fvc = fvc_df[fvc_df['Patient'] == patient_id]
        if len(patient_fvc) == 0:
            continue
        
        patient_fvc = patient_fvc.iloc[0]
        
        # Find decline data
        patient_decline = decline_df[decline_df['Patient'] == patient_id]
        
        # Load metrics
        advanced = load_advanced_metrics(case_name)
        if advanced is None:
            continue
        
        parenchymal = load_parenchymal_metrics(case_name)
        
        # Base row with demographics and metrics
        row = {
            'patient': patient_id,
            'case': case_name,
            
            'Age': patient_fvc['Age'],
            'Sex': patient_fvc['Sex'],
            'SmokingStatus': patient_fvc['SmokingStatus'],
            
            # Airway metrics
            'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
            'peripheral_branch_density': advanced.get('peripheral_branch_density'),
            'mean_peripheral_diameter_mm': advanced.get('mean_peripheral_diameter_mm'),
            'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
            
            # Parenchymal metrics
            'mean_lung_density_HU': parenchymal.get('mean_lung_density_HU') if parenchymal else np.nan,
            'histogram_entropy': parenchymal.get('histogram_entropy') if parenchymal else np.nan,
        }
        
        # Add FVC data (traditional drop)
        has_fvc = False
        if not pd.isna(patient_fvc['FVC_percent_week0']) and not pd.isna(patient_fvc['FVC_percent_week52']):
            week0_quality = patient_fvc['week0_quality']
            week52_quality = patient_fvc['week52_quality']
            
            # Apply quality filter
            fvc_pass = False
            if quality_filter == 'strict':
                fvc_pass = (week0_quality in ['high', 'medium'] and week52_quality in ['high', 'medium'])
            elif quality_filter == 'balanced':
                fvc_pass = (week0_quality in ['high', 'medium', 'low'] and week52_quality in ['high', 'medium', 'low'])
            else:  # 'all'
                fvc_pass = True
            
            if fvc_pass:
                row['FVC_percent_week0'] = patient_fvc['FVC_percent_week0']
                row['FVC_percent_week52'] = patient_fvc['FVC_percent_week52']
                row['FVC_drop_percent'] = patient_fvc['FVC_drop_percent']
                row['week0_quality'] = week0_quality
                row['week52_quality'] = week52_quality
                has_fvc = True
            else:
                row['FVC_percent_week0'] = np.nan
                row['FVC_percent_week52'] = np.nan
                row['FVC_drop_percent'] = np.nan
                row['week0_quality'] = week0_quality
                row['week52_quality'] = week52_quality
        else:
            row['FVC_percent_week0'] = np.nan
            row['FVC_percent_week52'] = np.nan
            row['FVC_drop_percent'] = np.nan
            row['week0_quality'] = patient_fvc.get('week0_quality', 'failed')
            row['week52_quality'] = patient_fvc.get('week52_quality', 'failed')
        
        # Add decline data
        has_decline = False
        if len(patient_decline) > 0:
            patient_decline = patient_decline.iloc[0]
            decline_quality = patient_decline['decline_quality']
            
            # Apply quality filter
            decline_pass = False
            if quality_filter == 'strict':
                decline_pass = (decline_quality in ['high', 'medium'])
            elif quality_filter == 'balanced':
                decline_pass = (decline_quality in ['high', 'medium'])
            else:  # 'all'
                decline_pass = (decline_quality != 'insufficient_data')
            
            if decline_pass and not pd.isna(patient_decline['FVC_annual_decline_direct']):
                row['FVC_annual_decline_direct'] = patient_decline['FVC_annual_decline_direct']
                row['decline_quality'] = decline_quality
                row['decline_n_measurements'] = patient_decline['decline_n_measurements']
                row['decline_timespan_weeks'] = patient_decline['decline_timespan_weeks']
                row['decline_r_value'] = patient_decline['decline_r_value']
                has_decline = True
            else:
                row['FVC_annual_decline_direct'] = np.nan
                row['decline_quality'] = decline_quality
                row['decline_n_measurements'] = patient_decline.get('decline_n_measurements', 0)
                row['decline_timespan_weeks'] = np.nan
                row['decline_r_value'] = np.nan
        else:
            row['FVC_annual_decline_direct'] = np.nan
            row['decline_quality'] = 'no_data'
            row['decline_n_measurements'] = 0
            row['decline_timespan_weeks'] = np.nan
            row['decline_r_value'] = np.nan
        
        # Aggiungi paziente solo se ha almeno uno dei due target
        if has_fvc or has_decline:
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Statistics
    n_total = len(df)
    n_with_fvc = df['FVC_drop_percent'].notna().sum()
    n_with_decline = df['FVC_annual_decline_direct'].notna().sum()
    n_with_both = ((df['FVC_drop_percent'].notna()) & (df['FVC_annual_decline_direct'].notna())).sum()
    n_only_fvc = n_with_fvc - n_with_both
    n_only_decline = n_with_decline - n_with_both
    
    print(f"  Total patients: {n_total}")
    print(f"    • With FVC drop only: {n_only_fvc}")
    print(f"    • With decline only: {n_only_decline}")
    print(f"    • With BOTH: {n_with_both}")
    print(f"  FVC drop available: {n_with_fvc}/{n_total} ({100*n_with_fvc/n_total:.1f}%)")
    print(f"  Decline available: {n_with_decline}/{n_total} ({100*n_with_decline/n_total:.1f}%)")
    
    return df

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_feature_predictions_unified(df_traditional, df_direct, feature, output_dir):
    """
    Crea un plot 2x4 con tutti e 4 i target:
    - Riga 1: Scatter plots (FVC week0, week52, drop, decline)
    - Riga 2: Bland-Altman plots per ciascun target
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
            X = df_traditional.loc[common_idx, feature].values.reshape(-1, 1)
            y_drop = df_traditional.loc[common_idx, 'FVC_drop_percent'].values
            
            predictions, actuals, errors = [], [], []
            for i in range(len(X)):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y_drop, i)
                X_test = X[i].reshape(1, -1)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)[0]
                
                predictions.append(y_pred)
                actuals.append(y_drop[i])
                errors.append(y_pred - y_drop[i])
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            errors = np.array(errors)
            
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actuals, predictions)
            pearson_r, pearson_p = pearsonr(actuals, predictions)
            
            result_drop = {
                'feature': feature,
                'target': 'FVC_drop_percent',
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
    
    # Crea figura 2x4 (2 righe, 4 colonne)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Prediction Performance: {feature}', fontsize=18, fontweight='bold', y=0.995)
    
    # ========== RIGA 1: SCATTER PLOTS ==========
    
    # 1. FVC Week 0 - Scatter
    if result_week0:
        ax = axes[0, 0]
        ax.scatter(result_week0['actuals'], result_week0['predictions'], 
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5, color='steelblue')
        
        min_val = min(result_week0['actuals'].min(), result_week0['predictions'].min())
        max_val = max(result_week0['actuals'].max(), result_week0['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect')
        
        # Regression line
        z = np.polyfit(result_week0['actuals'], result_week0['predictions'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        ax.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Regression')
        
        ax.set_xlabel('Actual FVC Week 0 (% predicted)', fontsize=11)
        ax.set_ylabel('Predicted FVC Week 0 (% predicted)', fontsize=11)
        ax.set_title(f'FVC at Week 0\nR² = {result_week0["r2"]:.3f}, MAE = {result_week0["mae"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_week0["n_samples"]}\nRMSE = {result_week0["rmse"]:.2f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 2. FVC Week 52 - Scatter
    if result_week52:
        ax = axes[0, 1]
        ax.scatter(result_week52['actuals'], result_week52['predictions'], 
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5, color='steelblue')
        
        min_val = min(result_week52['actuals'].min(), result_week52['predictions'].min())
        max_val = max(result_week52['actuals'].max(), result_week52['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect')
        
        z = np.polyfit(result_week52['actuals'], result_week52['predictions'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        ax.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Regression')
        
        ax.set_xlabel('Actual FVC Week 52 (% predicted)', fontsize=11)
        ax.set_ylabel('Predicted FVC Week 52 (% predicted)', fontsize=11)
        ax.set_title(f'FVC at Week 52\nR² = {result_week52["r2"]:.3f}, MAE = {result_week52["mae"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_week52["n_samples"]}\nRMSE = {result_week52["rmse"]:.2f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 3. Drop Tradizionale - Scatter
    if result_drop:
        ax = axes[0, 2]
        ax.scatter(result_drop['actuals'], result_drop['predictions'], 
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5, color='coral')
        
        min_val = min(result_drop['actuals'].min(), result_drop['predictions'].min())
        max_val = max(result_drop['actuals'].max(), result_drop['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect')
        
        z = np.polyfit(result_drop['actuals'], result_drop['predictions'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        ax.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Regression')
        
        ax.set_xlabel('Actual FVC Drop (%)', fontsize=11)
        ax.set_ylabel('Predicted FVC Drop (%)', fontsize=11)
        ax.set_title(f'Traditional Drop (Week0-Week52)\nR² = {result_drop["r2"]:.3f}, MAE = {result_drop["mae"]:.2f}%', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_drop["n_samples"]}\nRMSE = {result_drop["rmse"]:.2f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        axes[0, 2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        axes[0, 2].set_title('Traditional Drop', fontsize=12, fontweight='bold')
    
    # 4. Declino Diretto - Scatter
    if result_direct:
        ax = axes[0, 3]
        ax.scatter(result_direct['actuals'], result_direct['predictions'], 
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5, color='darkgreen')
        
        min_val = min(result_direct['actuals'].min(), result_direct['predictions'].min())
        max_val = max(result_direct['actuals'].max(), result_direct['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect')
        
        z = np.polyfit(result_direct['actuals'], result_direct['predictions'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        ax.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Regression')
        
        ax.set_xlabel('Actual Annual Decline (%/year)', fontsize=11)
        ax.set_ylabel('Predicted Annual Decline (%/year)', fontsize=11)
        ax.set_title(f'Direct Annual Decline\nR² = {result_direct["r2"]:.3f}, MAE = {result_direct["mae"]:.2f}%/year', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'n = {result_direct["n_samples"]}\nRMSE = {result_direct["rmse"]:.2f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        axes[0, 3].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        axes[0, 3].set_title('Direct Annual Decline', fontsize=12, fontweight='bold')
    
    # ========== RIGA 2: BLAND-ALTMAN PLOTS ==========
    
    # 1. FVC Week 0 - Bland-Altman
    if result_week0:
        ax = axes[1, 0]
        means = (result_week0['actuals'] + result_week0['predictions']) / 2
        differences = result_week0['predictions'] - result_week0['actuals']
        
        ax.scatter(means, differences, alpha=0.7, s=60, 
                  edgecolors='black', linewidth=0.5, color='steelblue')
        
        mean_diff = np.mean(differences)
        sd_diff = np.std(differences)
        upper_limit = mean_diff + 1.96 * sd_diff
        lower_limit = mean_diff - 1.96 * sd_diff
        
        ax.axhline(y=mean_diff, color='blue', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_diff:.2f}')
        ax.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'+1.96 SD: {upper_limit:.2f}')
        ax.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'-1.96 SD: {lower_limit:.2f}')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Mean of Actual and Predicted', fontsize=11)
        ax.set_ylabel('Predicted - Actual', fontsize=11)
        ax.set_title('Bland-Altman: FVC Week 0', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 2. FVC Week 52 - Bland-Altman
    if result_week52:
        ax = axes[1, 1]
        means = (result_week52['actuals'] + result_week52['predictions']) / 2
        differences = result_week52['predictions'] - result_week52['actuals']
        
        ax.scatter(means, differences, alpha=0.7, s=60, 
                  edgecolors='black', linewidth=0.5, color='steelblue')
        
        mean_diff = np.mean(differences)
        sd_diff = np.std(differences)
        upper_limit = mean_diff + 1.96 * sd_diff
        lower_limit = mean_diff - 1.96 * sd_diff
        
        ax.axhline(y=mean_diff, color='blue', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_diff:.2f}')
        ax.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'+1.96 SD: {upper_limit:.2f}')
        ax.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'-1.96 SD: {lower_limit:.2f}')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Mean of Actual and Predicted', fontsize=11)
        ax.set_ylabel('Predicted - Actual', fontsize=11)
        ax.set_title('Bland-Altman: FVC Week 52', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 3. Drop Tradizionale - Bland-Altman
    if result_drop:
        ax = axes[1, 2]
        means = (result_drop['actuals'] + result_drop['predictions']) / 2
        differences = result_drop['predictions'] - result_drop['actuals']
        
        ax.scatter(means, differences, alpha=0.7, s=60, 
                  edgecolors='black', linewidth=0.5, color='coral')
        
        mean_diff = np.mean(differences)
        sd_diff = np.std(differences)
        upper_limit = mean_diff + 1.96 * sd_diff
        lower_limit = mean_diff - 1.96 * sd_diff
        
        ax.axhline(y=mean_diff, color='blue', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_diff:.2f}')
        ax.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'+1.96 SD: {upper_limit:.2f}')
        ax.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'-1.96 SD: {lower_limit:.2f}')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Mean of Actual and Predicted', fontsize=11)
        ax.set_ylabel('Predicted - Actual', fontsize=11)
        ax.set_title('Bland-Altman: Traditional Drop', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('Bland-Altman: Traditional Drop', fontsize=12, fontweight='bold')
    
    # 4. Declino Diretto - Bland-Altman
    if result_direct:
        ax = axes[1, 3]
        means = (result_direct['actuals'] + result_direct['predictions']) / 2
        differences = result_direct['predictions'] - result_direct['actuals']
        
        ax.scatter(means, differences, alpha=0.7, s=60, 
                  edgecolors='black', linewidth=0.5, color='darkgreen')
        
        mean_diff = np.mean(differences)
        sd_diff = np.std(differences)
        upper_limit = mean_diff + 1.96 * sd_diff
        lower_limit = mean_diff - 1.96 * sd_diff
        
        ax.axhline(y=mean_diff, color='blue', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_diff:.2f}')
        ax.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'+1.96 SD: {upper_limit:.2f}')
        ax.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1.5,
                  label=f'-1.96 SD: {lower_limit:.2f}')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Mean of Actual and Predicted', fontsize=11)
        ax.set_ylabel('Predicted - Actual', fontsize=11)
        ax.set_title('Bland-Altman: Direct Annual Decline', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 3].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        axes[1, 3].set_title('Bland-Altman: Direct Annual Decline', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{feature}_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_unified_summary(df_traditional, df_direct, features, output_dir):
    """
    Crea un unico file CSV con tutte le performance predittive.
    """
    summary_data = []
    
    for feature in features:
        # Week 0
        result_week0 = leave_one_out_predict(df_traditional, feature, 'FVC_percent_week0')
        if result_week0:
            summary_data.append({
                'Feature': feature,
                'Target': 'FVC_week0',
                'Target_Description': 'FVC at Week 0',
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
                'Target_Description': 'FVC at Week 52',
                'n_samples': result_week52['n_samples'],
                'R2': result_week52['r2'],
                'MAE': result_week52['mae'],
                'RMSE': result_week52['rmse'],
                'Pearson_r': result_week52['pearson_r'],
                'Pearson_p': result_week52['pearson_p']
            })
        
        # Drop tradizionale
        if result_week0 and result_week52:
            valid_data_week0 = df_traditional[[feature, 'FVC_percent_week0']].dropna()
            valid_data_week52 = df_traditional[[feature, 'FVC_percent_week52']].dropna()
            common_idx = valid_data_week0.index.intersection(valid_data_week52.index)
            
            if len(common_idx) >= 5:
                X = df_traditional.loc[common_idx, feature].values.reshape(-1, 1)
                y_drop = df_traditional.loc[common_idx, 'FVC_drop_percent'].values
                
                predictions, actuals = [], []
                for i in range(len(X)):
                    X_train = np.delete(X, i, axis=0)
                    y_train = np.delete(y_drop, i)
                    X_test = X[i].reshape(1, -1)
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)[0]
                    
                    predictions.append(y_pred)
                    actuals.append(y_drop[i])
                
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                mse = mean_squared_error(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(actuals, predictions)
                pearson_r, pearson_p = pearsonr(actuals, predictions)
                
                summary_data.append({
                    'Feature': feature,
                    'Target': 'Drop_traditional',
                    'Target_Description': 'FVC Drop (Week0 - Week52)',
                    'n_samples': len(X),
                    'R2': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Pearson_r': pearson_r,
                    'Pearson_p': pearson_p
                })
        
        # Declino diretto
        result_direct = leave_one_out_predict(df_direct, feature, 'FVC_annual_decline_direct')
        if result_direct:
            summary_data.append({
                'Feature': feature,
                'Target': 'Decline_direct',
                'Target_Description': 'Annual FVC Decline (direct)',
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
        print(f"\n🏆 TOP FEATURES for Direct Annual Decline (highest R²):")
        for idx, row in direct_summary.head(3).iterrows():
            print(f"  {row['Feature']}: R² = {row['R2']:.3f}, MAE = {row['MAE']:.2f}%/year (n={row['n_samples']})")
    
    return df_summary

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("UNIFIED FVC PREDICTION ANALYSIS - V2")
    print("="*80)
    print("✓ Traditional approach: Week0, Week52, Drop")
    print("✓ Direct decline: Annual FVC decline from regression")
    print("✓ UNIFIED dataset with BOTH targets")
    print("✓ Multiple quality levels: strict, balanced, all")
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
    print(f"\n✓ Saved: 01_interpolated_fvc.csv")
    
    # =========================================================================
    # 3. Create direct decline dataset
    # =========================================================================
    decline_df = create_dataset_with_direct_decline(clinical, min_measurements=3)
    decline_df.to_csv(OUTPUT_DIR / "02_direct_decline.csv", index=False)
    print(f"✓ Saved: 02_direct_decline.csv")
    
    # =========================================================================
    # 4. Create UNIFIED datasets with different quality filters
    # =========================================================================
    print(f"\n{'='*80}")
    print("CREATING UNIFIED DATASETS WITH MULTIPLE QUALITY LEVELS")
    print(f"{'='*80}")
    
    # STRICT: high/medium for FVC, high/medium for decline
    print(f"\n[1/3] Creating STRICT dataset (highest quality)...")
    df_strict = integrate_airway_fvc_and_decline_unified(reliable, fvc_df, decline_df, quality_filter='strict')
    df_strict.to_csv(OUTPUT_DIR / "dataset_strict.csv", index=False)
    print(f"  ✓ Saved: dataset_strict.csv")
    
    # BALANCED: high/medium/low for FVC, high/medium for decline (RECOMMENDED)
    print(f"\n[2/3] Creating BALANCED dataset (recommended)...")
    df_balanced = integrate_airway_fvc_and_decline_unified(reliable, fvc_df, decline_df, quality_filter='balanced')
    df_balanced.to_csv(OUTPUT_DIR / "dataset_balanced.csv", index=False)
    print(f"  ✓ Saved: dataset_balanced.csv")
    
    # ALL: no quality filter (maximum patients)
    print(f"\n[3/3] Creating ALL dataset (no quality filter)...")
    df_all = integrate_airway_fvc_and_decline_unified(reliable, fvc_df, decline_df, quality_filter='all')
    df_all.to_csv(OUTPUT_DIR / "dataset_all.csv", index=False)
    print(f"  ✓ Saved: dataset_all.csv")
    
    # =========================================================================
    # 5. Create separate datasets for specific analyses
    # =========================================================================
    print(f"\n{'='*80}")
    print("CREATING SPECIALIZED DATASETS")
    print(f"{'='*80}")
    
    # Traditional only (only patients with both week0 and week52)
    df_traditional_only = df_balanced[df_balanced['FVC_drop_percent'].notna()].copy()
    df_traditional_only.to_csv(OUTPUT_DIR / "dataset_traditional_only.csv", index=False)
    print(f"  ✓ dataset_traditional_only.csv: {len(df_traditional_only)} patients")
    
    # Decline only (only patients with valid decline)
    df_decline_only = df_balanced[df_balanced['FVC_annual_decline_direct'].notna()].copy()
    df_decline_only.to_csv(OUTPUT_DIR / "dataset_decline_only.csv", index=False)
    print(f"  ✓ dataset_decline_only.csv: {len(df_decline_only)} patients")
    
    # Both targets (patients with BOTH drop and decline)
    df_both = df_balanced[
        (df_balanced['FVC_drop_percent'].notna()) & 
        (df_balanced['FVC_annual_decline_direct'].notna())
    ].copy()
    df_both.to_csv(OUTPUT_DIR / "dataset_both_targets.csv", index=False)
    print(f"  ✓ dataset_both_targets.csv: {len(df_both)} patients (BOTH targets)")
    
    # =========================================================================
    # 6. Create quality distribution summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("QUALITY DISTRIBUTION SUMMARY")
    print(f"{'='*80}")
    
    quality_summary = []
    for name, df_version in [('strict', df_strict), ('balanced', df_balanced), ('all', df_all)]:
        n_total = len(df_version)
        n_fvc = df_version['FVC_drop_percent'].notna().sum()
        n_decline = df_version['FVC_annual_decline_direct'].notna().sum()
        n_both = ((df_version['FVC_drop_percent'].notna()) & (df_version['FVC_annual_decline_direct'].notna())).sum()
        
        quality_summary.append({
            'Dataset': name,
            'Total_Patients': n_total,
            'With_FVC_Drop': n_fvc,
            'With_Decline': n_decline,
            'With_Both': n_both,
            'FVC_Only': n_fvc - n_both,
            'Decline_Only': n_decline - n_both
        })
    
    quality_df = pd.DataFrame(quality_summary)
    quality_df.to_csv(OUTPUT_DIR / "quality_summary.csv", index=False)
    
    print("\n" + quality_df.to_string(index=False))
    print(f"\n✓ Saved: quality_summary.csv")
    
    # =========================================================================
    # 7. Define features and create plots
    # =========================================================================
    features = [
        'mean_peripheral_branch_volume_mm3',
        'peripheral_branch_density',
        'mean_peripheral_diameter_mm',
        'central_to_peripheral_diameter_ratio',
        'mean_lung_density_HU',
        'histogram_entropy'
    ]
    
    # Create unified plots (2x2 grid with all 4 targets)
    print(f"\n{'='*80}")
    print("CREATING UNIFIED PREDICTION PLOTS (2x2 grid)")
    print(f"Using BALANCED dataset (recommended)")
    print(f"{'='*80}")
    
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    for feature in features:
        print(f"  Creating plot for: {feature}")
        plot_feature_predictions_unified(df_traditional_only, df_decline_only, feature, plots_dir)
    
    print(f"\n✓ All plots saved in: {plots_dir}")
    
    # =========================================================================
    # 8. Create unified summary CSV
    # =========================================================================
    print(f"\n{'='*80}")
    print("CREATING PREDICTION PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    summary_df = create_unified_summary(df_traditional_only, df_decline_only, features, OUTPUT_DIR)
    
    # Create comparison view (pivot table for easy comparison)
    if len(summary_df) > 0:
        pivot_df = summary_df.pivot(index='Feature', columns='Target', values=['R2', 'MAE', 'n_samples'])
        pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
        pivot_df.to_csv(OUTPUT_DIR / "performance_comparison_pivot.csv", index=False)
        print(f"✓ Saved: performance_comparison_pivot.csv")
    
    # =========================================================================
    # 9. Final summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"\n📊 RAW DATA:")
    print(f"  01_interpolated_fvc.csv       - FVC interpolated (all patients)")
    print(f"  02_direct_decline.csv         - Direct decline (all patients)")
    print(f"\n📊 UNIFIED DATASETS (with BOTH targets):")
    print(f"  dataset_strict.csv            - Highest quality (n={len(df_strict)})")
    print(f"  dataset_balanced.csv          - Balanced quality - RECOMMENDED (n={len(df_balanced)})")
    print(f"  dataset_all.csv               - All data (n={len(df_all)})")
    print(f"\n📊 SPECIALIZED DATASETS:")
    print(f"  dataset_traditional_only.csv  - Only FVC drop (n={len(df_traditional_only)})")
    print(f"  dataset_decline_only.csv      - Only annual decline (n={len(df_decline_only)})")
    print(f"  dataset_both_targets.csv      - BOTH drop & decline (n={len(df_both)})")
    print(f"\n📄 SUMMARY:")
    print(f"  quality_summary.csv                   - Quality distribution across datasets")
    print(f"  prediction_performance_summary.csv    - Prediction performance for all features")
    print(f"  performance_comparison_pivot.csv      - Pivot table for easy comparison")
    print(f"\n🖼️  PLOTS:")
    print(f"  plots/ - 2x2 scatter plots for each feature (4 targets: Week0, Week52, Drop, Decline)")
    print(f"\n💡 RECOMMENDED FOR TESTING:")
    print(f"  • For Week52 prediction: dataset_balanced.csv or dataset_traditional_only.csv")
    print(f"  • For Decline prediction: dataset_balanced.csv or dataset_decline_only.csv")
    print(f"  • For comparing both: dataset_both_targets.csv")
    print(f"\n💡 WHY DIFFERENT PATIENT NUMBERS?")
    print(f"  • FVC drop requires Week0 AND Week52 with good quality")
    print(f"  • Decline requires ≥3 measurements spanning reasonable time")
    print(f"  • Some patients have one but not the other!")
    print(f"  • dataset_both_targets.csv has only patients with BOTH ✓")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
