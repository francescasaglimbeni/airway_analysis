import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from models.train_utils import predict_mlp


def compute_aggregate_metrics(results_df):
    """
    Calcola R², MAE, RMSE per ogni modello.
    
    Args:
        results_df: DataFrame con risultati LOOCV
    
    Returns:
        summary_df: DataFrame con metriche aggregate
    """
    actual = results_df['actual'].values
    models = {
        'MLP (multi-feature)': results_df['mlp_pred'].values,
        'LR (multi-feature)': results_df['lr_multi_pred'].values,
        'LR (best single)': results_df['best_single_pred'].values,
    }
    
    summary = []
    for name, preds in models.items():
        # Rimuovi NaN
        mask = ~np.isnan(preds)
        a = actual[mask]
        p = preds[mask]
        
        if len(a) == 0:
            continue
            
        r2 = r2_score(a, p)
        mae = mean_absolute_error(a, p)
        rmse = np.sqrt(mean_squared_error(a, p))
        r, p_val = pearsonr(a, p)
        
        summary.append({
            'Model': name,
            'n_samples': int(mask.sum()),
            'R²': round(r2, 4),
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'Pearson_r': round(r, 4),
            'Pearson_p': round(p_val, 4),
        })
    
    return pd.DataFrame(summary)


def compute_permutation_importance(model, X_test, y_test, feature_names, device, n_repeats=100):
    """
    Permutation importance: per ogni feature, permuta i valori nel test set
    e misura quanto peggiora la predizione (MAE).
    
    Args:
        model: Modello MLP addestrato
        X_test: Dati di test
        y_test: Target di test
        feature_names: Lista dei nomi delle feature
        device: torch.device
        n_repeats: Numero di ripetizioni per permutazione
    
    Returns:
        importances: Dizionario {feature_name: importance_score}
    """
    base_pred = predict_mlp(model, X_test, device)
    base_mae = mean_absolute_error(y_test, base_pred)
    
    importances = {}
    for i, fname in enumerate(feature_names):
        maes = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            perm_pred = predict_mlp(model, X_perm, device)
            maes.append(mean_absolute_error(y_test, perm_pred))
        importances[fname] = np.mean(maes) - base_mae
    
    return importances