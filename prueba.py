import pickle
import os

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
