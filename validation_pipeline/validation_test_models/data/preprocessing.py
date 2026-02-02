"""
Funzioni per caricamento e preprocessing dati
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_preprocess_data(input_path, features, target):
    """
    Carica il dataset CSV e fa preprocessing.
    
    Args:
        input_path: Path al file CSV
        features: Lista delle feature da usare
        target: Nome della colonna target
    
    Returns:
        df_clean: DataFrame pulito
    
    Raises:
        FileNotFoundError: Se il file non esiste
        ValueError: Se feature o target mancanti
    """
    # Verifica che il file esista
    if not Path(input_path).exists():
        raise FileNotFoundError(f"File non trovato: {input_path}")
    
    # Carica dati
    df = pd.read_csv(input_path)
    print(f"  Caricati {len(df)} pazienti da {input_path.name}")
    
    # Verifica feature disponibili
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Feature mancanti nel dataset: {missing_features}")
    
    # Verifica target
    if target not in df.columns:
        raise ValueError(f"Target '{target}' non trovato nel dataset")
    
    # Seleziona colonne necessarie (incluse le colonne di qualità)
    required_columns = features + [target, 'patient']
    
    # Aggiungi colonne di qualità se esistono
    quality_columns = []
    if 'week0_quality' in df.columns:
        quality_columns.append('week0_quality')
    if 'week52_quality' in df.columns:
        quality_columns.append('week52_quality')
    
    required_columns += quality_columns
    df_clean = df[required_columns].copy()
    
    # Rimuovi righe con NaN
    initial_len = len(df_clean)
    df_clean = df_clean.dropna().reset_index(drop=True)
    n_dropped = initial_len - len(df_clean)
    
    if n_dropped > 0:
        print(f"  ⚠ Rimossi {n_dropped} pazienti con valori mancanti")
    
    return df_clean