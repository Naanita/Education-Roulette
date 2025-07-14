import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Iniciando el proceso de entrenamiento desde cero...")

# --- 1. Carga de Datos (Ajustado para CSV sin cabecera) ---
try:
    # Lee el CSV asumiendo que no tiene cabecera y la columna se llama 'result'
    df = pd.read_csv('results.csv', header=None, names=['result'])
    print(f"Dataset 'results.csv' cargado con {len(df)} registros.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'results.csv'. Por favor, asegúrate de que exista en la misma carpeta.")
    exit()

# --- 2. Ingeniería de Características (Simplificada sin tiempo) ---
print("Realizando ingeniería de características...")

sequence_length = 10
df['rolling_mean'] = df['result'].rolling(window=sequence_length).mean()
df['rolling_std'] = df['result'].rolling(window=sequence_length).std()
df['rolling_min'] = df['result'].rolling(window=sequence_length).min()
df['rolling_max'] = df['result'].rolling(window=sequence_length).max()

df['even_odd'] = df['result'].apply(lambda x: 1 if x > 0 and x % 2 == 0 else (0 if x > 0 else -1))
df['high_low'] = df['result'].apply(lambda x: 1 if x >= 19 else (0 if x >= 1 else -1))
df['dozen'] = df['result'].apply(lambda x: (x - 1) // 12 + 1 if x > 0 else 0)

# Crear las columnas objetivo (el resultado que ocurrirá N giros en el futuro)
future_steps = 10
df['target_number'] = df['result'].shift(-future_steps)
df['target_dozen'] = df['dozen'].shift(-future_steps)
df['target_high_low'] = df['high_low'].shift(-future_steps)

df.dropna(inplace=True)

# --- 3. Preparación para el Entrenamiento ---
# Lista de características actualizada (sin 'Hour', 'Minute')
features_cols = [
    'result', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
    'even_odd', 'high_low', 'dozen'
]

X = df[features_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Entrenamiento de los 3 Modelos ---

# A) Modelo para predecir NÚMEROS PLENOS
print("\n--- Entrenando Modelo para Números Plenos ---")
y_numbers = df['target_number'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_numbers, test_size=0.2, random_state=42)
model_numbers = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_numbers.fit(X_train, y_train)
print(f"Precisión del modelo de números: {model_numbers.score(X_test, y_test) * 100:.2f}%")

# B) Modelo para predecir DOCENAS
print("\n--- Entrenando Modelo para Docenas ---")
y_dozens = df['target_dozen'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_dozens, test_size=0.2, random_state=42)
model_dozens = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model_dozens.fit(X_train, y_train)
print(f"Precisión del modelo de docenas: {model_dozens.score(X_test, y_test) * 100:.2f}%")

# C) Modelo para predecir ALTOS/BAJOS
print("\n--- Entrenando Modelo para Altos/Bajos ---")
y_high_low = df['target_high_low'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_high_low, test_size=0.2, random_state=42)
model_high_low = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model_high_low.fit(X_train, y_train)
print(f"Precisión del modelo de altos/bajos: {model_high_low.score(X_test, y_test) * 100:.2f}%")

# --- 5. Guardado de los Modelos y el Scaler ---
print("\n--- Guardando todos los archivos de modelo ---")
joblib.dump(model_numbers, 'model_numbers.pkl')
joblib.dump(model_dozens, 'model_dozens.pkl')
joblib.dump(model_high_low, 'model_high_low.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n¡Proceso completado! Los 3 modelos y el scaler han sido creados.")