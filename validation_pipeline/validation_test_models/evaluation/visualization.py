"""
Funzioni per visualizzazione risultati
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from matplotlib.patches import Patch


def plot_all_results(results_df, importances, fold_curves, output_dir, features):
    """
    Genera tutte le visualizzazioni.
    
    Args:
        results_df: DataFrame con risultati LOOCV
        importances: Lista di dizionari con feature importance
        fold_curves: Lista di dizionari con curve di training
        output_dir: Directory di output
        features: Lista delle feature
    """
    print("  Generazione visualizzazioni...")
    
    # Plot 1: Actual vs Predicted
    plot_actual_vs_predicted(results_df, output_dir)
    
    # Plot 2: Bland-Altman
    plot_bland_altman(results_df, output_dir)
    
    # Plot 3: Per-patient errors
    plot_per_patient_errors(results_df, output_dir)
    
    # Plot 4: Feature importance (se disponibile)
    if importances:
        plot_feature_importance(importances, features, output_dir)
    else:
        print("  ⚠ Nessun dato di feature importance disponibile")
    
    # Plot 5: Training curves (se disponibili)
    if fold_curves:
        plot_training_curves(fold_curves, output_dir)
    else:
        print("  ⚠ Nessuna curva di training disponibile")
    
    print("  ✓ Tutte le visualizzazioni generate")


def plot_actual_vs_predicted(results_df, output_dir):
    """Scatter: actual vs predicted per tutti i modelli."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('FVC% Week 52 — Actual vs Predicted (LOOCV)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    actual = results_df['actual'].values
    
    configs = [
        ('ensemble_pred', '⭐ Ensemble\n(Ridge 60% + RF 40%)', 'darkviolet', axes[0, 0]),
        ('ridge_pred', 'Ridge Regression\n[L2 reg, tuned α]', 'forestgreen', axes[0, 1]),
        ('rf_pred', 'Random Forest\n[n=100, depth=3]', 'darkorange', axes[0, 2]),
        ('lasso_pred', 'Lasso Regression\n[L1 reg, α=0.1]', 'purple', axes[0, 3]),
        ('lr_multi_pred', 'Linear Regression\n[No Regularization]', 'coral', axes[1, 0]),
        ('mlp_pred', 'MLP (multi-feature)\n[Deep Learning - FAILURE]', 'steelblue', axes[1, 1]),
        ('best_single_pred', 'Best Single Feature\n[Linear]', 'gray', axes[1, 2]),
    ]
    
    for col, title, color, ax in configs:
        preds = results_df[col].values
        mask = ~np.isnan(preds)
        a, p = actual[mask], preds[mask]
        
        if len(a) == 0:
            continue
            
        r2 = r2_score(a, p)
        mae = mean_absolute_error(a, p)
        r, _ = pearsonr(a, p)
        
        ax.scatter(a, p, color=color, s=70, edgecolors='black',
                   linewidth=0.8, alpha=0.85, zorder=3)
        
        # Identity line
        lo, hi = min(a.min(), p.min()) - 2, max(a.max(), p.max()) + 2
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, alpha=0.5, label='y = x')
        
        # Regression line
        z = np.polyfit(a, p, 1)
        xr = np.linspace(lo, hi, 100)
        ax.plot(xr, np.poly1d(z)(xr), color=color, linewidth=2, alpha=0.7, label='Regression')
        
        ax.set_xlabel('Actual FVC% week52', fontsize=12)
        ax.set_ylabel('Predicted FVC% week52', fontsize=12)
        ax.set_title(f'{title}\nR²={r2:.3f} | MAE={mae:.2f} | r={r:.3f}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Nascondi l'ultimo subplot (7 modelli in griglia 2x4)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_actual_vs_predicted.png', bbox_inches='tight')
    plt.close()
    print("  ✓ plot_actual_vs_predicted.png salvato")


def plot_bland_altman(results_df, output_dir):
    """Bland-Altman per tutti i modelli."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Bland-Altman: Predicted − Actual  (LOOCV)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    actual = results_df['actual'].values
    
    configs = [
        ('ensemble_pred', '⭐ Ensemble', 'darkviolet', axes[0, 0]),
        ('ridge_pred', 'Ridge (tuned)', 'forestgreen', axes[0, 1]),
        ('rf_pred', 'Random Forest', 'darkorange', axes[0, 2]),
        ('lasso_pred', 'Lasso (L1)', 'purple', axes[0, 3]),
        ('lr_multi_pred', 'Linear Regression', 'coral', axes[1, 0]),
        ('mlp_pred', 'MLP', 'steelblue', axes[1, 1]),
        ('best_single_pred', 'Best Single', 'gray', axes[1, 2]),
    ]
    
    for col, title, color, ax in configs:
        preds = results_df[col].values
        mask = ~np.isnan(preds)
        a, p = actual[mask], preds[mask]
        
        if len(a) == 0:
            continue
            
        means = (a + p) / 2
        diffs = p - a
        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs)
        
        ax.scatter(means, diffs, color=color, s=60, edgecolors='black',
                   linewidth=0.7, alpha=0.85)
        
        # Lines
        ax.axhline(mean_diff, color='blue', linewidth=1.8,
                   label=f'Mean diff = {mean_diff:.2f}')
        ax.axhline(mean_diff + 1.96*sd_diff, color='red', linestyle='--', linewidth=1.2,
                   label=f'+1.96 SD = {mean_diff + 1.96*sd_diff:.2f}')
        ax.axhline(mean_diff - 1.96*sd_diff, color='red', linestyle='--', linewidth=1.2,
                   label=f'−1.96 SD = {mean_diff - 1.96*sd_diff:.2f}')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        
        ax.set_xlabel('Mean of Actual & Predicted', fontsize=12)
        ax.set_ylabel('Predicted − Actual', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Nascondi l'ultimo subplot
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_bland_altman.png', bbox_inches='tight')
    plt.close()
    print("  ✓ plot_bland_altman.png salvato")


def plot_per_patient_errors(results_df, output_dir):
    """Horizontal bar chart degli errori per paziente, ordinati per |errore MLP|."""
    df = results_df.sort_values('mlp_error', key=lambda x: abs(x), ascending=True).copy()
    y_pos = np.arange(len(df))
    
    fig, ax = plt.subplots(figsize=(15, max(9, len(df)*0.45)))
    
    bar_h = 0.12
    ax.barh(y_pos + 3*bar_h, df['ensemble_error'], height=bar_h*1.5,
            color='darkviolet', alpha=0.85, label='⭐ Ensemble', edgecolor='black', linewidth=0.4)
    ax.barh(y_pos + 2*bar_h, df['ridge_error'], height=bar_h*1.5,
            color='forestgreen', alpha=0.75, label='Ridge', edgecolor='black', linewidth=0.4)
    ax.barh(y_pos + bar_h, df['rf_error'], height=bar_h*1.5,
            color='darkorange', alpha=0.75, label='Random Forest', edgecolor='black', linewidth=0.4)
    ax.barh(y_pos, df['lr_multi_error'], height=bar_h*1.5,
            color='coral', alpha=0.75, label='Linear Reg', edgecolor='black', linewidth=0.4)
    ax.barh(y_pos - bar_h, df['lasso_error'], height=bar_h*1.5,
            color='purple', alpha=0.75, label='Lasso', edgecolor='black', linewidth=0.4)
    ax.barh(y_pos - 2*bar_h, df['mlp_error'], height=bar_h*1.5,
            color='steelblue', alpha=0.75, label='MLP', edgecolor='black', linewidth=0.4)
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['patient'][:20]}..." if len(str(row['patient'])) > 20
                        else str(row['patient'])
                        for _, row in df.iterrows()], fontsize=8.5)
    ax.set_xlabel('Prediction Error (Predicted − Actual) FVC%', fontsize=12)
    ax.set_title('Per-Patient Prediction Errors — sorted by |MLP error|',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_per_patient_errors.png', bbox_inches='tight')
    plt.close()
    print("  ✓ plot_per_patient_errors.png salvato")


def plot_feature_importance(all_importances, features, output_dir):
    """Bar chart della permutation importance media sui fold."""
    if not all_importances:
        print("  ⚠ Nessun dato di feature importance disponibile")
        return
    
    # Aggrega: media e std sui fold
    imp_matrix = pd.DataFrame(all_importances)[features]
    means = imp_matrix.mean()
    stds = imp_matrix.std()
    
    # Ordina per importanza media
    order = means.sort_values(ascending=True).index
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = []
    for f in order:
        if 'lung_density' in f or 'entropy' in f:
            colors.append('coral')          # parenchimale
        elif 'peripheral' in f or 'central_to' in f:
            colors.append('mediumseagreen') # periferico
        else:
            colors.append('steelblue')      # core airway
    
    ax.barh(range(len(order)), means[order], xerr=stds[order],
            color=colors, alpha=0.75, edgecolor='black', linewidth=0.6,
            capsize=4, height=0.6)
    
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=11)
    ax.set_xlabel('Permutation Importance (ΔMAE)', fontsize=12)
    ax.set_title('Feature Importance — MLP (media sui fold LOOCV)',
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Legend manuale
    legend_elements = [
        Patch(facecolor='coral', alpha=0.75, label='Parenchymal'),
        Patch(facecolor='mediumseagreen', alpha=0.75, label='Peripheral Airway'),
        Patch(facecolor='steelblue', alpha=0.75, label='Core Airway'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("  ✓ plot_feature_importance.png salvato")


def plot_training_curves(fold_curves, output_dir):
    """Salva le curve train/val loss per ogni fold."""
    curves_dir = output_dir / 'training_curves'
    curves_dir.mkdir(exist_ok=True)
    
    for fold_data in fold_curves:
        fold_idx = fold_data['fold']
        patient = fold_data['patient']
        train_l = fold_data['train_losses']
        val_l = fold_data['val_losses']
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        epochs = range(len(train_l))
        
        ax.plot(epochs, train_l, color='steelblue', linewidth=1.5, label='Train MSE')
        ax.plot(epochs, val_l, color='coral', linewidth=1.5, label='Val MSE')
        
        # Marca l'epoch del best val
        best_epoch = int(np.argmin(val_l))
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=1,
                   alpha=0.7, label=f'Best val (epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('MSE Loss', fontsize=11)
        ax.set_title(f'Fold {fold_idx+1} — Patient {str(patient)[:25]}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # scala log per leggibilità
        
        plt.tight_layout()
        plt.savefig(curves_dir / f'fold_{fold_idx+1:02d}_loss_curve.png', bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ training_curves/ ({len(fold_curves)} curve salvate)")