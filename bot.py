# Importamos las librerías necesarias
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup

def extraer_numeros_haciendo_clic():
    """
    Función que extrae los números navegando a través de la paginación
    haciendo clic en el botón "Siguiente", en lugar de cambiar la URL.
    """
    # Configuración de Selenium
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    options.add_argument('--log-level=3')
    
    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException:
        print("Error: No se pudo iniciar el WebDriver. Asegúrate de que 'chromedriver' esté en la misma carpeta que el script.")
        return []

    # URL inicial - Solo la visitaremos una vez
    url_inicial = "https://casinoscores.com/es-419/xxxtreme-lightning-roulette/"
    
    print(f"Visitando la página inicial: {url_inicial}")
    driver.get(url_inicial)

    numeros_encontrados = []
    pagina_numero = 1

    # Bucle que se ejecuta hasta que ya no podamos pasar a la siguiente página
    while True:
        print(f"-> Procesando página: {pagina_numero}")
        try:
            # Esperamos a que la tabla de la página actual cargue sus datos
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'roulette_badge__9BwRC')))
            
            # Damos un respiro extra para que todo se asiente bien
            time.sleep(1)

            # Extraemos el HTML y lo procesamos con BeautifulSoup
            html_final = driver.page_source
            soup = BeautifulSoup(html_final, 'html.parser')
            
            elementos_numeros = soup.find_all('span', class_='roulette_badge__9BwRC')
            
            # Extraemos los números de la página actual
            for elemento in elementos_numeros:
                numero = elemento.get_text(strip=True)
                if numero.isdigit():
                    numeros_encontrados.append(numero)

        except TimeoutException:
            print("La tabla de resultados no cargó a tiempo. Finalizando.")
            break

        # --- Lógica para pasar a la siguiente página ---
        try:
            # Buscamos el elemento <li> que contiene el botón "Siguiente".
            # Esto nos permite verificar si está deshabilitado.
            contenedor_boton_siguiente = driver.find_element(By.XPATH, "//li[contains(@class, 'page-item') and .//a[@aria-label='Next']]")

            # Si el elemento <li> tiene la clase 'disabled', es la última página.
            if 'disabled' in contenedor_boton_siguiente.get_attribute('class'):
                print("El botón 'Siguiente' está deshabilitado. Hemos llegado a la última página.")
                break

            # Si no está deshabilitado, buscamos el botón <a> y le hacemos clic
            boton_siguiente = driver.find_element(By.XPATH, "//a[@aria-label='Next']")
            
            # Usamos un clic con JavaScript, que a veces es más confiable
            driver.execute_script("arguments[0].click();", boton_siguiente)
            
            pagina_numero += 1

        except NoSuchElementException:
            # Si no se encuentra el botón, significa que ya no hay más páginas.
            print("No se encontró el botón 'Siguiente'. Hemos llegado al final.")
            break
            
    # Al final de todo, cerramos el navegador
    driver.quit()
    
    return numeros_encontrados

# --- Ejecución del Script ---
if __name__ == "__main__":
    lista_completa_numeros = extraer_numeros_haciendo_clic()

    if lista_completa_numeros:
        print("\n--- Extracción Completada ---")
        # Usamos set para obtener solo los números únicos y luego lo volvemos lista
        numeros_unicos = sorted(list(set(lista_completa_numeros)), key=int)
        print(f"Se encontraron un total de {len(numeros_encontrados)} resultados.")
        print(f"Los números únicos encontrados son: {len(numeros_unicos)}")
        print(numeros_unicos)
    else:
        print("\nNo se pudo extraer ningún número.")