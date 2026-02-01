"""
Funzioni per split dati e LOOCV
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from models.train_utils import train_mlp, predict_mlp
from evaluation.baselines import run_linear_baselines
from evaluation.metrics import compute_permutation_importance


def run_loocv(df, features, target, device, config):
    """
    Leave-One-Out CV con split interno train/val.
    
    Args:
        df: DataFrame con i dati
        features: Lista delle feature
        target: Nome della colonna target
        device: torch.device
        config: Dizionario con configurazioni
    
    Returns:
        results_df: DataFrame con predizioni e errori
        all_importances: Lista di dizionari con feature importance
        fold_curves: Lista di dizionari con curve di training
    """
    n = len(df)
    X_all = df[features].values.astype(np.float64)
    y_all = df[target].values.astype(np.float64)
    patients = df['patient'].values
    
    results = []
    all_importances = []
    fold_curves = []
    
    print(f"\n  LOOCV: {n} fold | Target: {target}")
    print(f"  Features: {len(features)}")
    print(f"  Inner splits: {config['n_inner_splits']} per fold")
    print()
    
    for i in range(n):
        patient_id = patients[i]
        print(f"  Fold {i+1:2d}/{n} | Test: {patient_id}", end="", flush=True)
        
        # --- SPLIT ESTERNA ---
        X_test = X_all[i:i+1]       # shape (1, n_features)
        y_test = y_all[i:i+1]       # shape (1,)
        X_pool = np.delete(X_all, i, axis=0)   # 25 pazienti
        y_pool = np.delete(y_all, i, axis=0)
        
        # --- LINEAR BASELINES sul pool intero ---
        scaler_base = StandardScaler()
        X_pool_scaled_base = scaler_base.fit_transform(X_pool)
        X_test_scaled_base = scaler_base.transform(X_test)
        
        # Linear Regression (no regularization)
        lr_full = LinearRegression()
        lr_full.fit(X_pool_scaled_base, y_pool)
        lr_multi_pred = lr_full.predict(X_test_scaled_base)[0]
        
        # Ridge Regression (L2 regularization)
        ridge = Ridge(alpha=1.0, random_state=config['seed'])
        ridge.fit(X_pool_scaled_base, y_pool)
        ridge_pred = ridge.predict(X_test_scaled_base)[0]
        
        # Lasso Regression (L1 regularization)
        lasso = Lasso(alpha=0.1, random_state=config['seed'], max_iter=5000)
        lasso.fit(X_pool_scaled_base, y_pool)
        lasso_pred = lasso.predict(X_test_scaled_base)[0]
        
        # Random Forest Regressor
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=config['seed'],
            n_jobs=-1
        )
        rf.fit(X_pool_scaled_base, y_pool)
        rf_pred = rf.predict(X_test_scaled_base)[0]
        
        # Single-feature baselines (non normalizzate)
        single_baselines = run_linear_baselines(
            X_pool, y_pool, X_test, y_test, features
        )
        
        # Trova la miglior feature singola (ignorando NaN)
        valid_baselines = {k: v for k, v in single_baselines.items() if not np.isnan(v)}
        if valid_baselines:
            best_single_feature = max(valid_baselines, key=valid_baselines.get)
            best_single_pred = valid_baselines[best_single_feature]
        else:
            best_single_feature = None
            best_single_pred = np.nan
        
        # --- MLP: ripeti N_INNER_SPLITS volte, tieni il best val MAE ---
        best_mlp_pred = None
        best_mlp_val_mae = float('inf')
        best_curves = None
        best_importances = None
        
        for split_idx in range(config['n_inner_splits']):
            seed = config['seed'] + i * config['n_inner_splits'] + split_idx
            
            # Split train/val sul pool
            np.random.seed(seed)
            indices = np.random.permutation(len(X_pool))
            n_val = max(1, int(len(X_pool) * config['val_fraction']))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            
            X_train_raw = X_pool[train_idx]
            y_train = y_pool[train_idx]
            X_val_raw = X_pool[val_idx]
            y_val = y_pool[val_idx]
            
            # StandardScaler fittato SOLO su train
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_test_scaled = scaler.transform(X_test)
            
            # Train MLP
            model, train_losses, val_losses = train_mlp(
                X_train, y_train, X_val, y_val, device,
                {**config, 'seed': seed}
            )
            
            # Val MAE finale (per model selection tra i split)
            val_pred = predict_mlp(model, X_val, device)
            current_val_mae = np.mean(np.abs(val_pred - y_val))
            
            if current_val_mae < best_mlp_val_mae:
                best_mlp_val_mae = current_val_mae
                best_mlp_pred = predict_mlp(model, X_test_scaled, device)[0]
                best_curves = (train_losses, val_losses)
                
                # Permutation importance sul pool (più stabile di 1 paziente)
                X_pool_scaled = scaler.transform(X_pool)
                best_importances = compute_permutation_importance(
                    model, X_pool_scaled, y_pool, features, device,
                    n_repeats=50
                )
        
        # Salva importances di questo fold
        if best_importances:
            all_importances.append(best_importances)
        
        # Salva curves del best split
        if best_curves:
            fold_curves.append({
                'fold': i,
                'patient': patient_id,
                'train_losses': best_curves[0],
                'val_losses': best_curves[1]
            })
        
        # --- REGISTRA RISULTATI ---
        actual = y_test[0]
        
        results.append({
            'fold': i,
            'patient': patient_id,
            'actual': actual,
            'mlp_pred': best_mlp_pred,
            'lr_multi_pred': lr_multi_pred,
            'ridge_pred': ridge_pred,
            'lasso_pred': lasso_pred,
            'rf_pred': rf_pred,
            'best_single_feature': best_single_feature,
            'best_single_pred': best_single_pred,
            'mlp_error': best_mlp_pred - actual if best_mlp_pred is not None else np.nan,
            'lr_multi_error': lr_multi_pred - actual,
            'ridge_error': ridge_pred - actual,
            'lasso_error': lasso_pred - actual,
            'rf_error': rf_pred - actual,
            'best_single_error': best_single_pred - actual if not np.isnan(best_single_pred) else np.nan,
            'mlp_val_mae': best_mlp_val_mae,
        })
        
        mlp_err = results[-1]['mlp_error']
        print(f"  | actual={actual:6.1f} | MLP={best_mlp_pred:6.1f} | "
              f"Ridge={ridge_pred:6.1f} | Lasso={lasso_pred:6.1f} | RF={rf_pred:6.1f}")
    
    results_df = pd.DataFrame(results)
    return results_df, all_importances, fold_curves