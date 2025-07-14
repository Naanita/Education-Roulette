# app_v2.py (Corregido)
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import logging

# Importar funciones de utils
from utils import create_input_features

# --- Configuración del Logging ---
# Usamos logging en lugar de st.toast dentro de la función en caché
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes y Configuración ---
MODELS_DIR = 'trained_models'
HISTORY_FILE = 'results.csv'
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
INITIAL_BALANCE = 100000
BET_AMOUNT_PER_NUMBER = 1000
MIN_HISTORY_FOR_PREDICTION = 10

# --- Funciones de Carga de Modelos ---

# <-- CAMBIO: Se eliminaron los st.toast y st.error de esta función para evitar el CacheReplayClosureError.
@st.cache_resource
def load_all_models(models_dir):
    """
    Carga todos los modelos (.h5, .joblib) y el scaler.
    Esta función está optimizada para el caché y no debe contener elementos de UI.
    """
    models = {}
    scaler = None
    
    if not os.path.exists(models_dir):
        logging.error(f"El directorio de modelos '{models_dir}' no existe.")
        return models, scaler

    for filename in os.listdir(models_dir):
        file_path = os.path.join(models_dir, filename)
        try:
            if filename.endswith('.h5'):
                model_name = filename.replace('_model.h5', '')
                models[model_name] = load_model(file_path)
                logging.info(f"Modelo Keras '{model_name}' cargado.")
            elif filename.endswith('.joblib') or filename.endswith('.pkl'):
                model_name = filename.replace('_model.joblib', '').replace('.pkl', '')
                if 'scaler' in model_name:
                    scaler = joblib.load(file_path)
                    logging.info("Scaler cargado.")
                else:
                    models[model_name] = joblib.load(file_path)
                    logging.info(f"Modelo scikit-learn '{model_name}' cargado.")
        except Exception as e:
            logging.error(f"Error al cargar {filename}: {e}")
            
    return models, scaler

# --- Funciones de Interfaz de Usuario ---

def display_prediction_grid(numbers_to_highlight):
    """Muestra una parrilla de números de ruleta, destacando los predichos."""
    html_string = """<style>.pred-grid td{border:2px solid #555;text-align:center;font-weight:700;font-size:1.2em;color:#fff;padding:10px 5px;text-shadow:1px 1px 2px #000}.num-red{background-color:#c0392b}.num-black{background-color:#2c3e50}.num-green{background-color:#27ae60}.highlight{background-color:#f1c40f!important;color:#333!important;border:3px solid #fff;text-shadow:none}</style><table class="pred-grid" style="border-collapse:collapse;width:100%;font-family:sans-serif">"""
    zero_class = "num-green highlight" if 0 in numbers_to_highlight else "num-green"
    html_string += f'<tr><td colspan="12" class="{zero_class}">0</td></tr>'
    for row in range(3):
        html_string += "<tr>"
        for col in range(12):
            num = 3 * col + (3 - row)
            highlight_class = " highlight" if num in numbers_to_highlight else ""
            color_class = "num-red" if num in RED_NUMBERS else "num-black"
            html_string += f'<td class="{color_class}{highlight_class}">{num}</td>'
        html_string += "</tr>"
    html_string += "</table>"
    st.markdown(html_string, unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa el estado de la sesión para el historial, balance, etc."""
    if 'history' not in st.session_state:
        try:
            st.session_state.history = pd.read_csv(HISTORY_FILE, header=None, names=['result'])
            st.session_state.history.dropna(inplace=True)
            st.session_state.history['result'] = st.session_state.history['result'].astype(int)
        except FileNotFoundError:
            st.session_state.history = pd.DataFrame(columns=['result'])
    
    # Inicializar el resto de variables de sesión
    for key, value in [('balance', INITIAL_BALANCE), ('last_outcome', 0), ('predicted_numbers', [])]:
        if key not in st.session_state:
            st.session_state[key] = value

# --- Lógica Principal ---

def make_prediction(model, scaler, history):
    """Función aislada para generar una predicción."""
    input_df = create_input_features(history, sequence_length=MIN_HISTORY_FOR_PREDICTION)
    if input_df is None or scaler is None:
        return []
    
    input_scaled = scaler.transform(input_df)
    
    if hasattr(model, 'predict_proba'): # Para modelos scikit-learn
        probs = model.predict_proba(input_scaled)[0]
    else: # Para modelos Keras
        probs = model.predict(input_scaled)[0]
        
    # Recomendar los 12 números con mayor probabilidad
    return np.argsort(probs)[-12:][::-1].tolist()

def process_input(selected_number, active_model, scaler):
    """Procesa un nuevo número, actualiza el estado y genera nuevas predicciones."""
    # Simulación de apuesta
    if st.session_state.predicted_numbers:
        previous_balance = st.session_state.balance
        total_bet = len(st.session_state.predicted_numbers) * BET_AMOUNT_PER_NUMBER
        
        if selected_number in st.session_state.predicted_numbers:
            winnings = 36 * BET_AMOUNT_PER_NUMBER
            st.session_state.balance += (winnings - total_bet)
        else:
            st.session_state.balance -= total_bet
        st.session_state.last_outcome = st.session_state.balance - previous_balance

    # Actualizar historial
    new_row = pd.DataFrame({'result': [selected_number]})
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    st.session_state.history.to_csv(HISTORY_FILE, index=False, header=False)

    # Generar y guardar la siguiente predicción
    st.session_state.predicted_numbers = make_prediction(active_model, scaler, st.session_state.history)
    
    st.rerun()

def main():
    st.set_page_config(layout="wide", page_title="💰 Simulador de Ruleta Avanzado")
    st.title("💰 Simulador y Predictor de Ruleta v3.0")

    # <-- CAMBIO: Chequeo del directorio de modelos fuera de la función en caché
    if not os.path.exists(MODELS_DIR):
        st.error(f"El directorio de modelos '{MODELS_DIR}' no existe. Por favor, ejecuta el script de entrenamiento primero.")
        return

    models, scaler = load_all_models(MODELS_DIR)
    
    if not models or scaler is None:
        st.warning("No se han cargado modelos o el scaler. Revisa la carpeta 'trained_models' y los logs de la consola.")
        return

    initialize_session_state()
    
    model_options = list(models.keys())
    
    # <-- CAMBIO: Lógica para generar la predicción inicial al cargar la app.
    if not st.session_state.predicted_numbers and len(st.session_state.history) >= MIN_HISTORY_FOR_PREDICTION:
        st.toast("Generando recomendación inicial con el historial existente...", icon="⏳")
        initial_model = models[model_options[0]] # Usar el primer modelo de la lista para la predicción inicial
        st.session_state.predicted_numbers = make_prediction(initial_model, scaler, st.session_state.history)

    # --- Sidebar (Panel de control) ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        selected_model_name = st.selectbox(
            "Selecciona el modelo para predecir:",
            options=model_options,
            help="Elige el algoritmo para las recomendaciones."
        )
        active_model = models[selected_model_name]

        if st.button("🔄 Recargar Modelos", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.subheader("Estado de la Simulación")
        st.metric("Balance Actual", f"${st.session_state.balance:,.0f}", delta=f"${st.session_state.last_outcome:,.0f}")
        
        if st.button("🔥 Reiniciar Simulación", use_container_width=True, type="primary"):
            # Limpiar estado de la sesión
            st.session_state.balance = INITIAL_BALANCE
            st.session_state.last_outcome = 0
            st.session_state.history = pd.DataFrame(columns=['result'])
            st.session_state.predicted_numbers = []
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            st.rerun()
            
    # --- Columnas Principales ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("👇 Ingresa el Último Número")
        grid_cols = st.columns(12)
        for i in range(1, 37):
            col_index = (i - 1) % 12
            if grid_cols[col_index].button(f"{i}", key=f"btn_{i}", use_container_width=True):
                process_input(i, active_model, scaler)
        if st.button("0", key="btn_0", use_container_width=True):
            process_input(0, active_model, scaler)

    with col2:
        st.header(f"🔮 Predicciones con '{selected_model_name}'")
        
        if len(st.session_state.history) < MIN_HISTORY_FOR_PREDICTION:
            st.info(f"Se necesita un historial de al menos {MIN_HISTORY_FOR_PREDICTION} números. Faltan {MIN_HISTORY_FOR_PREDICTION - len(st.session_state.history)}.")
        elif not st.session_state.predicted_numbers:
            st.info("Ingresa un número para generar la primera recomendación.")
        else:
            st.subheader("Números Plenos Recomendados")
            display_prediction_grid(st.session_state.predicted_numbers)
            st.write(f"**Recomendación:** Apostar a los siguientes números: {', '.join(map(str, sorted(st.session_state.predicted_numbers)))}")
            
        with st.expander("Ver historial de números"):
            st.dataframe(st.session_state.history.tail(20).iloc[::-1], use_container_width=True)

if __name__ == "__main__":
    main()