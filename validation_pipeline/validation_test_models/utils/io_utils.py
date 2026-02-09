"""
Funzioni per I/O e salvataggio risultati
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def save_results(results_df, importances, fold_curves, output_dir, features, model_config=None, metrics=None):
    """
    Salva tutti i risultati in file, includendo architettura del modello.
    
    Args:
        results_df: DataFrame con risultati LOOCV
        importances: Lista di dizionari con feature importance
        fold_curves: Lista di dizionari con curve di training
        output_dir: Directory di output
        features: Lista delle feature
        model_config: Dizionario con configurazione del modello (architettura, hyperparameters)
        metrics: Dizionario con metriche di performance (R2, MAE, RMSE)
    """
    # Salva predizioni LOOCV
    results_df.to_csv(output_dir / 'loocv_predictions.csv', index=False)
    print(f"  ✓ loocv_predictions.csv salvato ({len(results_df)} pazienti)")
    
    # Salva feature importances
    if importances:
        importance_df = pd.DataFrame(importances)
        importance_df.to_csv(output_dir / 'feature_importances.csv', index=False)
        print(f"  ✓ feature_importances.csv salvato ({len(importances)} fold)")
    
    # Salva training curves (solo metadati)
    if fold_curves:
        curves_metadata = []
        for curve in fold_curves:
            curves_metadata.append({
                'fold': curve['fold'],
                'patient': curve['patient'],
                'train_losses_length': len(curve['train_losses']),
                'val_losses_length': len(curve['val_losses']),
                'final_train_loss': curve['train_losses'][-1] if curve['train_losses'] else None,
                'final_val_loss': curve['val_losses'][-1] if curve['val_losses'] else None,
            })
        
        curves_df = pd.DataFrame(curves_metadata)
        curves_df.to_csv(output_dir / 'training_curves_metadata.csv', index=False)
        print(f"  ✓ training_curves_metadata.csv salvato ({len(fold_curves)} curve)")
    
    # Salva configurazione completa con architettura modello
    config_summary = {
        'n_patients': len(results_df),
        'n_features': len(features),
        'features': features,
        'target': 'FVC_percent_week52',
    }
    
    # Aggiungi configurazione modello se fornita
    if model_config:
        config_summary['model_architecture'] = {
            'hidden_layer_1': model_config.get('hidden1'),
            'hidden_layer_2': model_config.get('hidden2'),
            'dropout': model_config.get('dropout'),
            'input_dim': len(features),
            'output_dim': 1
        }
        config_summary['hyperparameters'] = {
            'learning_rate': model_config.get('learning_rate'),
            'weight_decay': model_config.get('weight_decay'),
            'epochs_max': model_config.get('epochs_max'),
            'patience': model_config.get('patience'),
        }
        config_summary['training_config'] = {
            'val_fraction': model_config.get('val_fraction'),
            'n_inner_splits': model_config.get('n_inner_splits'),
            'seed': model_config.get('seed'),
        }
    
    with open(output_dir / 'config_summary.json', 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"  ✓ config_summary.json salvato")
    if model_config:
        print(f"  ✓ Architettura modello: Input({len(features)}) → {model_config.get('hidden1')} → {model_config.get('hidden2')} → 1")
    
    # Salva architettura in CSV storico
    if model_config and metrics:
        _save_architecture_to_csv(model_config, metrics, output_dir, features)


def _save_architecture_to_csv(model_config, metrics, output_dir, features):
    """
    Aggiunge l'architettura corrente al file CSV storico.
    
    Args:
        model_config: Dizionario con configurazione del modello
        metrics: Dizionario con metriche di performance
        output_dir: Directory di output
        features: Lista delle feature
    """
    csv_file = output_dir / 'model_architectures_history.csv'
    
    # Crea riga per questa configurazione
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_dim': len(features),
        'hidden_layer_1': model_config.get('hidden1'),
        'hidden_layer_2': model_config.get('hidden2'),
        'dropout': model_config.get('dropout'),
        'learning_rate': model_config.get('learning_rate'),
        'weight_decay': model_config.get('weight_decay'),
        'epochs_max': model_config.get('epochs_max'),
        'patience': model_config.get('patience'),
        'val_fraction': model_config.get('val_fraction'),
        'n_inner_splits': model_config.get('n_inner_splits'),
        'seed': model_config.get('seed'),
        'R2': metrics.get('R2'),
        'MAE': metrics.get('MAE'),
        'RMSE': metrics.get('RMSE')
    }
    
    # Carica storico esistente o crea nuovo DataFrame
    if csv_file.exists():
        history_df = pd.read_csv(csv_file)
        history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
    else:
        history_df = pd.DataFrame([row])
    
    # Salva CSV aggiornato
    history_df.to_csv(csv_file, index=False)
    print(f"  ✓ Architettura aggiunta a model_architectures_history.csv (totale: {len(history_df)} configurazioni)")