# train_v2.py
import os
import yaml
import json
import time
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Importar funciones de utils
from utils import create_input_features

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funciones de Carga y Preparación de Datos ---

def load_config(config_path='config.yml'):
    """Carga la configuración desde un archivo YAML."""
    logging.info(f"Cargando configuración desde {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """Carga el dataset de resultados."""
    try:
        df = pd.read_csv(file_path, header=None, names=['result'])
        logging.info(f"Dataset cargado con {len(df)} registros desde {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo de datos en {file_path}")
        return None

def engineer_features(df, sequence_length, future_steps):
    """Aplica ingeniería de características al DataFrame."""
    logging.info("Aplicando ingeniería de características...")
    
    # Crear características rodantes
    df['rolling_mean'] = df['result'].rolling(window=sequence_length, min_periods=1).mean()
    df['rolling_std'] = df['result'].rolling(window=sequence_length, min_periods=1).std()
    df['rolling_min'] = df['result'].rolling(window=sequence_length, min_periods=1).min()
    df['rolling_max'] = df['result'].rolling(window=sequence_length, min_periods=1).max()

    # Características categóricas
    df['even_odd'] = df['result'].apply(lambda x: 1 if x > 0 and x % 2 == 0 else (0 if x > 0 else -1))
    df['high_low'] = df['result'].apply(lambda x: 1 if x >= 19 else (0 if x >= 1 else -1))
    df['dozen'] = df['result'].apply(lambda x: (x - 1) // 12 + 1 if x > 0 else 0)
    
    # Crear la columna objetivo (el número que saldrá en el futuro)
    df['target_number'] = df['result'].shift(-future_steps)
    
    # Eliminar filas con NaN generadas por las operaciones de rolling y shift
    df.dropna(inplace=True)
    df['target_number'] = df['target_number'].astype(int)
    
    logging.info(f"El dataset procesado tiene {len(df)} filas.")
    return df

# --- Funciones de Modelos ---

def build_keras_model(input_shape, num_classes, config):
    """Construye un modelo de red neuronal con Keras."""
    model = Sequential()
    # Añade la primera capa densa
    model.add(Dense(
        units=config['layers'][0]['units'],
        activation=config['layers'][0]['activation'],
        input_shape=(input_shape,)
    ))
    
    # Añade el resto de capas definidas en la configuración
    for layer_config in config['layers'][1:]:
        if layer_config['type'] == 'Dense':
            model.add(Dense(units=layer_config['units'], activation=layer_config['activation']))
        elif layer_config['type'] == 'Dropout':
            model.add(Dropout(rate=layer_config['rate']))
            
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    logging.info("Modelo Keras construido exitosamente.")
    model.summary(print_fn=logging.info)
    return model

def save_training_plot(history, output_path):
    """Guarda las curvas de pérdida y precisión del entrenamiento."""
    plt.figure(figsize=(12, 5))

    # Gráfico de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Gráfico de Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Gráfico de entrenamiento guardado en: {output_path}")


# --- Flujo Principal de Entrenamiento ---

def main():
    """Función principal que orquesta el proceso de entrenamiento."""
    start_time = time.time()
    
    config = load_config()
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(config['data_path'])
    if df is None:
        return
        
    df_featured = engineer_features(df, config['sequence_length'], config['future_steps'])
    
    # Definir características y objetivo
    feature_cols = ['result', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max', 'even_odd', 'high_low', 'dozen']
    X = df_featured[feature_cols]
    y = df_featured['target_number']
    num_classes = len(np.unique(y)) # De 0 a 36, son 37 clases

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    logging.info("Scaler guardado en 'scaler.joblib'")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    training_summary = {}

    # --- Entrenamiento de Modelos ---
    for model_name in config['models_to_train']:
        model_start_time = time.time()
        logging.info(f"--- Entrenando modelo: {model_name} ---")
        
        hyperparams = config['hyperparameters'].get(model_name, {})
        model_path = os.path.join(output_dir, f'{model_name}_model')
        
        # --- Modelo: Random Forest ---
        if model_name == 'random_forest':
            model = RandomForestClassifier(**hyperparams)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path + '.joblib')
        
        # --- Modelo: SVM ---
        elif model_name == 'svm':
            model = SVC(**hyperparams)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path + '.joblib')

        # --- Modelo: Keras Neural Network ---
        elif model_name == 'keras_nn':
            # One-hot encode para Keras
            y_train_cat = to_categorical(y_train, num_classes=num_classes)
            y_test_cat = to_categorical(y_test, num_classes=num_classes)
            
            model = build_keras_model(X_train.shape[1], num_classes, hyperparams)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train_cat,
                epochs=hyperparams['epochs'],
                batch_size=hyperparams['batch_size'],
                validation_split=hyperparams['validation_split'],
                callbacks=[early_stopping],
                verbose=1
            )
            model.save(model_path + '.h5')
            save_training_plot(history, os.path.join(output_dir, f'{model_name}_training_plot.png'))
        
        else:
            logging.warning(f"Tipo de modelo '{model_name}' no soportado. Saltando.")
            continue
            
        logging.info(f"Modelo '{model_name}' guardado.")
        
        # --- Evaluación ---
        if model_name == 'keras_nn':
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logging.info(f"Resultados de {model_name}: Accuracy = {accuracy:.4f}, F1-Score = {f1:.4f}")
        
        # --- Guardar Resumen ---
        training_summary[model_name] = {
            'model_path': model_path + ('.h5' if model_name == 'keras_nn' else '.joblib'),
            'training_time_seconds': round(time.time() - model_start_time, 2),
            'hyperparameters': hyperparams,
            'metrics': {
                'accuracy': accuracy,
                'f1_score_weighted': f1,
                'classification_report': report
            }
        }

    # Guardar el resumen completo
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=4)
        
    total_time = time.time() - start_time
    logging.info(f"Proceso de entrenamiento completado en {total_time:.2f} segundos.")
    logging.info(f"Resumen del entrenamiento guardado en {summary_path}")

if __name__ == '__main__':
    main()