import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Cargar Modelos Entrenados ---
@st.cache_resource
def load_models():
    try:
        model_numbers = joblib.load('model_numbers.pkl')
        model_dozens = joblib.load('model_dozens.pkl')
        model_high_low = joblib.load('model_high_low.pkl')
        scaler = joblib.load('scaler.pkl')
        return model_numbers, model_dozens, model_high_low, scaler, True
    except FileNotFoundError:
        st.error("Error: No se encontraron los archivos de modelo. Por favor, ejecuta primero el script 'train_models.py'.")
        return None, None, None, None, False

model_numbers, model_dozens, model_high_low, scaler, MODELS_LOADED = load_models()

# --- Constantes y Funciones de Ayuda ---
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BET_AMOUNT_PER_NUMBER = 1000
INITIAL_BALANCE = 100000

def create_input_features(history_df):
    if len(history_df) < 10: return None
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
    ordered_features = ['result', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max', 'even_odd', 'high_low', 'dozen']
    return pd.DataFrame([features])[ordered_features]

def display_prediction_grid(numbers_to_highlight):
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
    # Carga el historial desde el archivo CSV si no está en la sesión
    if 'history' not in st.session_state:
        if os.path.exists('results.csv'):
            st.session_state.history = pd.read_csv('results.csv', header=None, names=['result'])
            # Elimina filas vacías que puedan leerse del CSV
            st.session_state.history.dropna(inplace=True)
            st.session_state.history['result'] = st.session_state.history['result'].astype(int)
            st.toast(f"Cargados {len(st.session_state.history)} resultados desde 'results.csv'")
        else:
            st.session_state.history = pd.DataFrame(columns=['result'])
    
    # Inicializa el balance si no existe
    if 'balance' not in st.session_state:
        st.session_state.balance = INITIAL_BALANCE

    # Inicializa el último resultado si no existe
    if 'last_outcome' not in st.session_state:
        st.session_state.last_outcome = 0
    
    # Inicializa la lista de números predichos si no existe
    if 'predicted_numbers' not in st.session_state:
        st.session_state.predicted_numbers = []

    # --- BLOQUE AÑADIDO PARA LA PREDICCIÓN INICIAL ---
    # Si la lista de predicciones está vacía Y tenemos suficiente historial...
    if not st.session_state.predicted_numbers and len(st.session_state.history) >= 10:
        st.toast("Generando recomendación inicial...")
        # ...entonces, genera la primera predicción.
        input_df = create_input_features(st.session_state.history)
        if input_df is not None:
            input_scaled = scaler.transform(input_df)
            number_probs = model_numbers.predict_proba(input_scaled)[0]
            # Guarda los números predichos en el estado de la sesión
            st.session_state.predicted_numbers = np.argsort(number_probs)[-12:][::-1].tolist()

# --- Lógica Principal de la App ---
def main():
    st.set_page_config(layout="wide")
    st.title("💰 Simulador y Predictor de Ruleta v2.0")

    if not MODELS_LOADED:
        return

    initialize_session_state()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("👇 Haz Clic en el Último Número")
        
        # --- Tablero de Entrada Clicable ---
        # Usamos CSS para el efecto hover
        st.markdown("""
            <style>
            .input-grid-btn {
                width: 100%;
                height: 50px;
                margin: 1px !important;
                padding: 0 !important;
                border: 1px solid #555;
                transition: transform 0.1s;
            }
            .input-grid-btn:hover {
                transform: scale(1.1);
                border: 2px solid #f1c40f;
            }
            </style>
        """, unsafe_allow_html=True)

        # Generar los botones en un grid
        grid_cols = st.columns(12)
        for i in range(1, 37):
            col_index = (i - 1) % 12
            if grid_cols[col_index].button(f"{i}", key=f"btn_{i}", use_container_width=True):
                process_bet(i)
        
        if st.button("0", key="btn_0", use_container_width=True):
            process_bet(0)
        
        st.markdown("---")
        st.subheader("Estado de la Simulación")
        st.metric("Balance Actual", f"${st.session_state.balance:,.0f} COP", delta=f"${st.session_state.last_outcome:,.0f} COP")
        
        if st.button("Reiniciar Simulación", use_container_width=True, type="secondary"):
            st.session_state.balance = INITIAL_BALANCE
            st.session_state.last_outcome = 0
            st.session_state.history = pd.DataFrame(columns=['result'])
            st.session_state.predicted_numbers = [] # Limpiar predicciones al reiniciar
            if os.path.exists('results.csv'): os.remove('results.csv')
            st.rerun()

    with col2:
        st.header("🔮 Predicciones para el Próximo Giro")
        
        if len(st.session_state.history) >= 10:
            st.subheader("Números Plenos Recomendados")
            display_prediction_grid(st.session_state.predicted_numbers)
            
            st.markdown("---")
            
            # Realizar y mostrar las otras predicciones
            input_df = create_input_features(st.session_state.history)
            if input_df is not None:
                input_scaled = scaler.transform(input_df)
                dozen_pred = model_dozens.predict(input_scaled)[0]
                high_low_pred = model_high_low.predict(input_scaled)[0]
                
                dozen_map = {1: "1-12", 2: "13-24", 3: "25-36", 0: "Cero"}
                high_low_map = {0: "Bajo (1-18)", 1: "Alto (19-36)", -1: "Cero"}
                
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

def process_bet(selected_number):
    """Función que se ejecuta al hacer clic en un número."""
    previous_balance = st.session_state.balance
    
    # 1. Realizar la predicción con el historial actual (antes de añadir el nuevo número)
    if len(st.session_state.history) >= 10:
        input_df = create_input_features(st.session_state.history)
        input_scaled = scaler.transform(input_df)
        number_probs = model_numbers.predict_proba(input_scaled)[0]
        # Estos son los números en los que SE DEBÍA haber apostado
        predicted_numbers_for_bet = np.argsort(number_probs)[-12:][::-1]
        
        # 2. Simular la apuesta
        total_bet = len(predicted_numbers_for_bet) * BET_AMOUNT_PER_NUMBER
        if selected_number in predicted_numbers_for_bet:
            winnings = 36 * BET_AMOUNT_PER_NUMBER
            st.session_state.balance += (winnings - total_bet)
        else:
            st.session_state.balance -= total_bet
        
        st.session_state.last_outcome = st.session_state.balance - previous_balance
    
    # 3. Actualizar el historial con el número que acaba de salir
    new_row = pd.DataFrame({'result': [selected_number]})
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    with open('results.csv', 'a') as f:
        # Asegurarse de no escribir una línea en blanco al principio del archivo
        if os.stat('results.csv').st_size == 0:
             f.write(f"{selected_number}")
        else:
             f.write(f"\n{selected_number}")

    # 4. Generar la predicción para el *siguiente* giro y guardarla para mostrarla
    if len(st.session_state.history) >= 10:
        next_input_df = create_input_features(st.session_state.history)
        next_input_scaled = scaler.transform(next_input_df)
        next_number_probs = model_numbers.predict_proba(next_input_scaled)[0]
        # Guardar como lista para evitar el ValueError
        st.session_state.predicted_numbers = np.argsort(next_number_probs)[-12:][::-1].tolist()
    
    st.rerun()

if __name__ == "__main__":
    main()