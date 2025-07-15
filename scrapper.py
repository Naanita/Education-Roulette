import requests
import csv
import math
import time

# --- Configuración ---
API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette"
ITEMS_POR_PAGINA = 27
# ============================================================================
# NUEVO PARÁMETRO: Cambia este valor para filtrar por diferentes períodos (ej. 24, 72, 168 para una semana)
DURACION_HORAS = 72
# ============================================================================
NOMBRE_ARCHIVO_SALIDA = f"resultados_ruleta_{DURACION_HORAS}h.csv"

def exportar_todos_los_numeros_filtrado():
    """
    Obtiene todos los resultados de la API para un período de tiempo específico
    y los exporta a un archivo CSV en orden cronológico.
    """
    print(f"Iniciando la exportación de datos para las últimas {DURACION_HORAS} horas...")

    # 1. Obtener el total de resultados para el período de tiempo especificado
    try:
        print("Obteniendo el total de resultados...")
        # Añadimos el parámetro 'duration' a la petición inicial
        params_iniciales = {'page': 0, 'size': 1, 'duration': DURACION_HORAS}
        respuesta_inicial = requests.get(API_URL, params=params_iniciales, timeout=10)
        respuesta_inicial.raise_for_status()
        
        total_items = int(respuesta_inicial.headers.get('X-Total-Count', 0))
        
        if total_items == 0:
            print(f"No se encontraron resultados para las últimas {DURACION_HORAS} horas. Abortando.")
            return

        total_paginas = math.ceil(total_items / ITEMS_POR_PAGINA)
        print(f"Se encontraron {total_items} resultados en un total de {total_paginas} páginas.")

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la API: {e}")
        return

    # 2. Recorrer todas las páginas y extraer los datos
    todos_los_numeros = []
    for pagina in range(total_paginas):
        # Añadimos el parámetro 'duration' a la petición de cada página
        params_pagina = {
            'page': pagina,
            'size': ITEMS_POR_PAGINA,
            'sort': 'data.settledAt,asc',
            'duration': DURACION_HORAS
        }
        
        print(f"Extrayendo página {pagina + 1} de {total_paginas}...")
        
        try:
            respuesta_pagina = requests.get(API_URL, params=params_pagina, timeout=15)
            respuesta_pagina.raise_for_status()
            
            resultados_json = respuesta_pagina.json()
            
            # Si una página viene vacía, es una señal de que hemos llegado al final de los datos disponibles
            if not resultados_json:
                print(f"La página {pagina + 1} no devolvió resultados. Finalizando la extracción.")
                break

            for item in resultados_json:
                numero = item['data']['result']['outcome']['number']
                todos_los_numeros.append(numero)
            
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"Error al obtener la página {pagina + 1}: {e}. Continuando...")
        except (KeyError, TypeError) as e:
            print(f"Error al procesar los datos de la página {pagina + 1}: {e}.")

    print(f"\nSe extrajeron un total de {len(todos_los_numeros)} números.")

    # 3. Guardar los datos en un archivo CSV
    try:
        with open(NOMBRE_ARCHIVO_SALIDA, mode='w', newline='', encoding='utf-8') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(['Numero'])
            
            for numero in todos_los_numeros:
                escritor_csv.writerow([numero])
        
        print(f"✅ ¡Éxito! Los datos han sido guardados en el archivo '{NOMBRE_ARCHIVO_SALIDA}'")

    except IOError as e:
        print(f"❌ Error al escribir el archivo CSV: {e}")


# --- Ejecución del Script ---
if __name__ == "__main__":
    exportar_todos_los_numeros_filtrado()