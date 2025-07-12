import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Cargar Modelos Entrenados ---
try:
    model_numbers = joblib.load('model_numbers.pkl')
    model_dozens = joblib.load('model_dozens.pkl')
    model_high_low = joblib.load('model_high_low.pkl')
    scaler = joblib.load('scaler.pkl')
    MODELS_LOADED = True
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos de modelo. Por favor, ejecuta primero el script 'train_models.py'.")
    MODELS_LOADED = False

# --- Constantes y Funciones de Ayuda ---
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BET_AMOUNT_PER_NUMBER = 1000
INITIAL_BALANCE = 100000

def create_input_features(history_df):
    """Crea el DataFrame de una fila con las características para la predicción."""
    if len(history_df) < 10:
        return None
    
    last_spin = history_df.iloc[-1]
    last_10_spins = history_df['result'].tail(10)
    
    features = {
        'result': last_spin['result'],
        'rolling_mean': last_10_spins.mean(),
        'rolling_std': last_10_spins.std(),
        'rolling_min': last_10_spins.min(),
        'rolling_max': last_10_spins.max(),
        'even_odd': 1 if last_spin['result'] > 0 and last_spin['result'] % 2 == 0 else (0 if last_spin['result'] > 0 else -1),
        'high_low': 1 if last_spin['result'] >= 19 else (0 if last_spin['result'] >= 1 else -1),
        'dozen': (last_spin['result'] - 1) // 12 + 1 if last_spin['result'] > 0 else 0
    }
    
    ordered_features = [
        'result', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
        'even_odd', 'high_low', 'dozen'
    ]
    
    return pd.DataFrame([features])[ordered_features]

def display_roulette_grid(numbers_to_highlight):
    """Muestra el tablero de la ruleta con los números resaltados."""
    html_string = """
    <style>
        .roulette-grid { border-collapse: collapse; width: 100%; font-family: sans-serif; text-shadow: 1px 1px 2px black; }
        .roulette-grid td { border: 2px solid #555; text-align: center; font-weight: bold; font-size: 1.2em; color: white; padding: 10px 5px; }
        .num-red { background-color: #c0392b; }
        .num-black { background-color: #2c3e50; }
        .num-green { background-color: #27ae60; }
        .highlight { background-color: #f1c40f !important; color: #333 !important; border: 3px solid white; text-shadow: none;}
    </style>
    <table class="roulette-grid">
    """
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

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide")
st.title("💰 Simulador y Predictor de Ruleta")

if MODELS_LOADED:
    # Inicializar el estado de la sesión para el historial y el balance
    if 'history' not in st.session_state:
        if os.path.exists('results.csv'):
            st.session_state.history = pd.read_csv('results.csv', header=None, names=['result'])
            st.toast(f"Cargados {len(st.session_state.history)} resultados desde 'results.csv'")
        else:
            st.session_state.history = pd.DataFrame(columns=['result'])
    
    if 'balance' not in st.session_state:
        st.session_state.balance = INITIAL_BALANCE

    if 'last_outcome' not in st.session_state:
        st.session_state.last_outcome = 0


    # --- Layout de la Aplicación ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎮 Panel de Control")
        new_number = st.number_input("Introduce el último número:", min_value=0, max_value=36, step=1, key="new_number")

        if st.button("Añadir, Apostar y Predecir", use_container_width=True):
            # Guardar el estado anterior para la simulación
            previous_balance = st.session_state.balance
            
            # Predecir ANTES de añadir el nuevo número
            if len(st.session_state.history) >= 10:
                input_df = create_input_features(st.session_state.history)
                input_scaled = scaler.transform(input_df)
                number_probs = model_numbers.predict_proba(input_scaled)[0]
                top_12_numbers = np.argsort(number_probs)[-12:][::-1]

                # Simulación de la apuesta
                total_bet = len(top_12_numbers) * BET_AMOUNT_PER_NUMBER
                if new_number in top_12_numbers:
                    winnings = 36 * BET_AMOUNT_PER_NUMBER
                    st.session_state.balance += (winnings - total_bet)
                else:
                    st.session_state.balance -= total_bet
                
                st.session_state.last_outcome = st.session_state.balance - previous_balance

            # Añadir nuevo número al historial para la siguiente predicción
            new_row = pd.DataFrame({'result': [new_number]})
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
            with open('results.csv', 'a') as f:
                f.write(f"\n{new_number}")
        
        st.markdown("---")
        st.subheader("Estado de la Simulación")
        st.metric("Balance Actual", f"${st.session_state.balance:,.0f} COP", delta=f"${st.session_state.last_outcome:,.0f} COP")
        
        if st.button("Reiniciar Simulación", use_container_width=True, type="secondary"):
            st.session_state.balance = INITIAL_BALANCE
            st.session_state.last_outcome = 0
            st.rerun()

    with col2:
        st.header("🔮 Predicciones para el Próximo Giro")
        
        if len(st.session_state.history) >= 10:
            input_df = create_input_features(st.session_state.history)
            
            if input_df is not None:
                input_scaled = scaler.transform(input_df)

                number_probs = model_numbers.predict_proba(input_scaled)[0]
                top_12_numbers = np.argsort(number_probs)[-12:][::-1]
                
                dozen_pred = model_dozens.predict(input_scaled)[0]
                dozen_map = {1: "1-12", 2: "13-24", 3: "25-36", 0: "Cero"}
                
                high_low_pred = model_high_low.predict(input_scaled)[0]
                high_low_map = {0: "Bajo (1-18)", 1: "Alto (19-36)", -1: "Cero"}

                st.subheader("Números Plenos Recomendados")
                display_roulette_grid(top_12_numbers)
                
                st.markdown("---")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric("Docena más Probable", dozen_map.get(dozen_pred, "N/A"))
                with pred_col2:
                    st.metric("Rango más Probable", high_low_map.get(high_low_pred, "N/A"))
        
        else:
            st.info(f"Se necesita un historial de al menos 10 números para predecir. Faltan {10 - len(st.session_state.history)}.")

        with st.expander("Ver historial de números"):
            if 'result' in st.session_state.history.columns:
                st.dataframe(st.session_state.history.tail(20).iloc[::-1], use_container_width=True)