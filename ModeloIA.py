import pickle, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Cargar el DataFrame histórico desde un archivo CSV o Pickle
df_historico = pd.read_csv('df_historico.csv')

def entrenar_modelo(df):
    # Convertir las categorías de lluvia a valores numéricos para el entrenamiento
    df["lluvia_madrugada_num"] = df["lluvia_madrugada"].map({"BAJA": 0, "MEDIA": 1, "ALTA": 2})
    df["lluvia_mannana_num"] = df["lluvia_mannana"].map({"BAJA": 0, "MEDIA": 1, "ALTA": 2})
    df["lluvia_tarde_num"] = df["lluvia_tarde"].map({"BAJA": 0, "MEDIA": 1, "ALTA": 2})
    df["lluvia_noche_num"] = df["lluvia_noche"].map({"BAJA": 0, "MEDIA": 1, "ALTA": 2})
    df["medio_transporte_num"] = df["medio_transporte"].map({"CARRO": 0, "MOTO": 1})

    # Definir las características (X) y el objetivo (y)
    X = df[["lluvia_madrugada_num", "lluvia_mannana_num", "lluvia_tarde_num", "lluvia_noche_num", "trafico_pesado"]]
    y = df["medio_transporte_num"]

    # Dividir los datos en entrenamiento y prueba
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de Red Neuronal
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42, max_iter=300)
    model.fit(X_train, y_train)
    return model
# Entrenar y guardar el modelo
modelo_entrenado = entrenar_modelo(df_historico)
# Supón que 'model' es tu modelo ya entrenado
with open('modelo_transporte.pkl', 'wb') as file:
    pickle.dump(modelo_entrenado, file)

print("Modelo guardado exitosamente usando pickle.")