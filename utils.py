# utils.py
import pandas as pd

def create_input_features(history_df, sequence_length=10):
    """
    Crea un DataFrame de características a partir del historial de números.

    Args:
        history_df (pd.DataFrame): DataFrame con la columna 'result'.
        sequence_length (int): Número de giros pasados a considerar.

    Returns:
        pd.DataFrame: Un DataFrame con las características para la predicción.
    """
    if len(history_df) < sequence_length:
        return None

    last_spin = history_df.iloc[-1]
    last_n_spins = history_df['result'].tail(sequence_length)

    features = {
        'result': last_spin['result'],
        'rolling_mean': last_n_spins.mean(),
        'rolling_std': last_n_spins.std(),
        'rolling_min': last_n_spins.min(),
        'rolling_max': last_n_spins.max(),
        'even_odd': 1 if last_spin['result'] > 0 and last_spin['result'] % 2 == 0 else (0 if last_spin['result'] > 0 else -1),
        'high_low': 1 if last_spin['result'] >= 19 else (0 if last_spin['result'] >= 1 else -1),
        'dozen': (last_spin['result'] - 1) // 12 + 1 if last_spin['result'] > 0 else 0
    }
    ordered_features = ['result', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max', 'even_odd', 'high_low', 'dozen']
    return pd.DataFrame([features], columns=ordered_features)