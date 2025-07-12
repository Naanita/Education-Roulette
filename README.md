# Simulador y Predictor de Ruleta

Este proyecto es una aplicación web desarrollada en Streamlit que permite simular apuestas y predecir resultados en la ruleta utilizando modelos de machine learning previamente entrenados.

## Requisitos

- Python 3.8 o superior
- Los archivos de modelos (`model_numbers.pkl`, `model_dozens.pkl`, `model_high_low.pkl`, `scaler.pkl`) generados previamente con `train_models.py`.

## Instalación

1. **Clona el repositorio o descarga los archivos.**

2. **Instala las dependencias:**

   ```sh
   pip install -r requirements.txt
   ```

3. **(Opcional) Entrena los modelos si no tienes los archivos `.pkl`:**

   ```sh
   python train_models.py
   ```

4. **Inicia la aplicación:**

   ```sh
   streamlit run app.py
   ```

5. **Uso:**
   - Ingresa los resultados de la ruleta en la interfaz.
   - Visualiza las predicciones y el balance simulado.

## Archivos importantes

- `app.py`: Aplicación principal de Streamlit.
- `train_models.py`: Script para entrenar y guardar los modelos.
- `results.csv`: Historial de resultados de la ruleta.
- `sc.py`: (Opcional) Utilidades adicionales.

---

## requirements.txt

````txt
streamlit
pandas
numpy
joblib
scikit-learn