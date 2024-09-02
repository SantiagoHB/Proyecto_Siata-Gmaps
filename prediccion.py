import googlemaps, requests, pandas as pd, os, pickle

from sklearn.neural_network import MLPClassifier

from datetime import datetime

from dotenv import load_dotenv

with open('C:/Users/Santi/Desktop/New folder (2)/gmapskey.env') as f:
    for line in f:
        if line.startswith('gmapskey='):
            gmaps_api_key = line.strip().split('=')[1]

gmaps = googlemaps.Client(key=gmaps_api_key)
# URL base del SIATA
base_url = "https://siata.gov.co/data/siata_app/"

def limpiar_consola():
    # Limpia la consola dependiendo del sistema operativo
    if os.name == 'nt':  # Para Windows
        os.system('cls')
    else:  # Para macOS/Linux
        os.system('clear')

# Nombres de los archivos JSON
json_files = [
    "wrfsabaneta.json", "wrfpalmitas.json", "wrfmedOriente.json", "wrfmedOccidente.json",
    "wrfmedCentro.json", "wrflaestrella.json", "wrfitagui.json", "wrfgirardota.json",
    "wrfenvigado.json", "wrfcopacabana.json", "wrfcaldas.json", "wrfbello.json", "wrfbarbosa.json"
]

# Inicializar el DataFrame global para mantener los datos
df_siata = pd.DataFrame()


# Método para hacer la solicitud a cada JSON y agregar los datos a un DataFrame
def obtener_datos_siata(base_url, json_files):
    global df_siata, df_historico  # Usar la variable global para mantener el DataFrame
    # Limpiar el DataFrame antes de agregar nuevos datos
    df_siata = pd.DataFrame()
    
    # Crear una lista para almacenar los registros
    registros = []
    
    for json_file in json_files:
        # Construir la URL completa
        url = base_url + json_file
        
        # Realizar la solicitud al URL del SIATA
        response = requests.get(url)
        
        # Verificar si la solicitud fue exitosa
        if response.status_code == 200:
            data = response.json()
            
            # Extraer la fecha y pronóstico del JSON
            pronosticos = data.get("pronostico", [])
            
            # Obtener el nombre de la zona desde el nombre del archivo (sin la extensión .json)
            zona = json_file.replace(".json", "")
            
            # Iterar sobre los pronósticos y extraer la información relevante
            for pronostico in pronosticos:
                fecha = pronostico.get("fecha")
                temp_max = pronostico.get("temperatura_maxima")
                temp_min = pronostico.get("temperatura_minima")
                lluvia_madrugada = pronostico.get("lluvia_madrugada")
                lluvia_mannana = pronostico.get("lluvia_mannana")
                lluvia_tarde = pronostico.get("lluvia_tarde")
                lluvia_noche = pronostico.get("lluvia_noche")
                
                # Agregar los datos a la lista, incluyendo la columna 'zona'
                registros.append({
                    "fecha": fecha,
                    "temperatura_maxima": temp_max,
                    "temperatura_minima": temp_min,
                    "lluvia_madrugada": lluvia_madrugada,
                    "lluvia_mannana": lluvia_mannana,
                    "lluvia_tarde": lluvia_tarde,
                    "lluvia_noche": lluvia_noche,
                    "zona": zona
                })
        else:
            print(f"Error al realizar la solicitud para {json_file}: {response.status_code}")
    
    # Convertir la lista en un DataFrame de pandas y asignarlo a la variable global
    df_siata = pd.DataFrame(registros)
    
    """ Depuración: Verificar el contenido de df_siata
    if df_siata.empty:
        print("Advertencia: df_siata está vacío. Verifica la conexión y los datos del SIATA.")
    else:
        print(f"df_siata contiene {len(df_siata)} registros.")
        print(df_siata.head())
    """
# Modificar el método para obtener la predicción de lluvia desde el DataFrame
def get_rain_prediction(time_of_day, df, zona):
    # Filtrar el DataFrame para obtener la fila correspondiente a la zona y el tiempo del día
    filtro = df[df["zona"] == zona]

    # Seleccionar la columna de interés según el tiempo del día
    if time_of_day == "madrugada":
        lluvia = filtro["lluvia_madrugada"].values[0]
    elif time_of_day == "mañana":
        lluvia = filtro["lluvia_mannana"].values[0]
    elif time_of_day == "tarde":
        lluvia = filtro["lluvia_tarde"].values[0]
    elif time_of_day == "noche":
        lluvia = filtro["lluvia_noche"].values[0]
    else:
        return None

    # Convertir las categorías de lluvia a valores numéricos
    if lluvia == "BAJA":
        return 0
    elif lluvia == "MEDIA":
        return 1
    elif lluvia == "ALTA":
        return 2
    else:
        return None
    
def get_travel_time(origin, destination):
    """
    Devuelve el tiempo estimado de trayecto en minutos entre el origen y el destino,
    y evalúa si el tráfico está más pesado de lo normal.
    
    :param origin: Dirección de origen (str)
    :param destination: Dirección de destino (str)
    :return: Tiempo estimado de trayecto en minutos (float), Indicador de tráfico pesado (bool)
    """
    # Hacer una solicitud a la API de direcciones de Google Maps
    directions_result = gmaps.directions(origin, destination, mode="driving", departure_time="now")
    
    if directions_result:
        # Tiempo de trayecto en condiciones normales
        duration_normal = directions_result[0]['legs'][0]['duration']['value']
        # Tiempo de trayecto teniendo en cuenta el tráfico actual
        duration_in_traffic = directions_result[0]['legs'][0]['duration_in_traffic']['value']
        
        # Convertir a minutos
        duration_in_minutes = duration_in_traffic / 60
        
        # Evaluar si el tráfico está más pesado de lo normal
        traffic_heavy = duration_in_traffic > duration_normal
        
        return duration_in_minutes, traffic_heavy
    else:
        print(f"No se pudo obtener el tiempo de trayecto entre {origin} y {destination}.")
        return None, None


# Método para hacer la predicción
def predecir_medio_transporte(origin, destination, current_time, model, df_siata, zona, travel_time, traffic_heavy):
    # Verificar si df_siata contiene la columna 'zona'
    if 'zona' not in df_siata.columns:
        return "Error: df_siata no contiene la columna 'zona'."

    if travel_time is None:
        return "No se pudo obtener el tiempo de trayecto."

    # Obtener las predicciones de lluvia para todos los periodos del día
    lluvia_madrugada = get_rain_prediction("madrugada", df_siata, zona)
    lluvia_mannana = get_rain_prediction("mañana", df_siata, zona)
    lluvia_tarde = get_rain_prediction("tarde", df_siata, zona)
    lluvia_noche = get_rain_prediction("noche", df_siata, zona)
    
    if None in [lluvia_madrugada, lluvia_mannana, lluvia_tarde, lluvia_noche]:
        return "No se pudo obtener la predicción de lluvia para todos los periodos del día."

    # Crear un DataFrame para la predicción
    X_new = pd.DataFrame({
        "lluvia_madrugada_num": [lluvia_madrugada],
        "lluvia_mannana_num": [lluvia_mannana],
        "lluvia_tarde_num": [lluvia_tarde],
        "lluvia_noche_num": [lluvia_noche],
        "trafico_pesado": [traffic_heavy]
    })
    
    # Hacer la predicción
    predicted_transport_mode = model.predict(X_new)[0]
    
    # Convertir la predicción numérica a la etiqueta correspondiente
    return "CARRO" if predicted_transport_mode == 0 else "MOTO"

# Llamar a la función de obtener datos del SIATA y combinar con el DataFrame histórico
obtener_datos_siata(base_url, json_files)

def seleccionar_zona():
    print("Seleccione la zona:")
    print("1. Sabaneta")
    print("2. Palmitas")
    print("3. MedOriente")
    print("4. MedOccidente")
    print("5. MedCentro")
    print("6. La Estrella")
    print("7. Itagüí")
    print("8. Girardota")
    print("9. Envigado")
    print("10. Copacabana")
    print("11. Caldas")
    print("12. Bello")
    print("13. Barbosa")

    while True:
        try:
            opcion = int(input("Ingrese el número correspondiente a la zona aproximada de la direccion de destino: "))
            if opcion == 1:
                return "wrfsabaneta"
            elif opcion == 2:
                return "wrfpalmitas"
            elif opcion == 3:
                return "wrfmedOriente"
            elif opcion == 4:
                return "wrfmedOccidente"
            elif opcion == 5:
                return "wrfmedCentro"
            elif opcion == 6:
                return "wrflaestrella"
            elif opcion == 7:
                return "wrfitagui"
            elif opcion == 8:
                return "wrfgirardota"
            elif opcion == 9:
                return "wrfenvigado"
            elif opcion == 10:
                return "wrfcopacabana"
            elif opcion == 11:
                return "wrfcaldas"
            elif opcion == 12:
                return "wrfbello"
            elif opcion == 13:
                return "wrfbarbosa"
            else:
                print("Opción no válida. Intente nuevamente.")
        except ValueError:
            print("Por favor, ingrese un número válido.")

# Ejemplo de uso
if __name__ == "__main__":
    # Solicitar la dirección de origen y destino

    # Cargar el modelo entrenado
    model_path = os.path.join(os.getcwd(), 'modelo_transporte.pkl')

    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file:
                modelo_entrenado = pickle.load(file)
            print("Modelo cargado exitosamente usando pickle.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
    else:
        print(f"Error: El archivo del modelo no se encuentra en {model_path}")


    limpiar_consola()
    origin = input("Ingrese la dirección de origen: ([Calle 1#1-1 o Lugar (Aereopuerto Olaya Herrera)], Municipio[Valle de Aburrá: Medellin, Envigado, etc], Colombia): ")
    limpiar_consola()
    destination = input("Ingrese la dirección de destino: ([Calle 1#1-1 o Lugar (Aereopuerto Olaya Herrera)], Municipio[Valle de Aburrá: Medellin, Envigado, etc], Colombia): ")
    limpiar_consola()
    
    # Seleccionar la zona
    zona = seleccionar_zona()
    
    # Obtener la hora actual del sistema
    current_time = datetime.now()
    
    # Obtener el tiempo estimado de trayecto y si el tráfico está pesado
    travel_time, traffic_heavy = get_travel_time(origin, destination)

    # Realizar la predicción del medio de transporte
    resultado = predecir_medio_transporte(origin, destination, current_time, modelo_entrenado, df_siata, zona, travel_time, traffic_heavy)

    # Determinar el estado del tráfico para imprimirlo
    estado_trafico = "pesado" if traffic_heavy else "normal"

    # Mostrar los resultados al usuario
    print(f"Para el trayecto entre {origin} y {destination}: ")
    print(f"El tiempo estimado de viaje es: {travel_time:.2f} minutos")
    print(f"Con un tráfico: {estado_trafico}")
    print(f"El medio de transporte recomendado es: {resultado}")