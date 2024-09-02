import numpy as np, pandas as pd, random

from datetime import datetime

# Crear el DataFrame histórico global
df_historico = pd.DataFrame()

# Función para decidir el medio de transporte basado en las reglas especificadas
def decidir_transporte(prob_lluvia, trafico_pesado):
    if prob_lluvia == "BAJA":
        if not trafico_pesado:
            return np.random.choice(["CARRO", "MOTO"], p=[0.5, 0.5])
        else:
            return np.random.choice(["CARRO", "MOTO"], p=[0.15, 0.85])
    elif prob_lluvia == "MEDIA":
        if not trafico_pesado:
            return np.random.choice(["CARRO", "MOTO"], p=[0.7, 0.3])
        else:
            return np.random.choice(["CARRO", "MOTO"], p=[0.15, 0.85])
    elif prob_lluvia == "ALTA":
        if not trafico_pesado:
            return np.random.choice(["CARRO", "MOTO"], p=[0.95, 0.05])
        else:
            return np.random.choice(["CARRO", "MOTO"], p=[0.15, 0.85])

# Generar el DataFrame histórico desde enero 1 del 2024 hasta la fecha actual
start_date = "2024-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
date_range = pd.date_range(start=start_date, end=end_date)

zonas_siata = [
    "wrfsabaneta", "wrfpalmitas", "wrfmedOriente", "wrfmedOccidente",
    "wrfmedCentro", "wrflaestrella", "wrfitagui", "wrfgirardota",
    "wrfenvigado", "wrfcopacabana", "wrfcaldas", "wrfbello", "wrfbarbosa"
]

data = []
for date in date_range:
    for zona in zonas_siata:
        # Generar la probabilidad de lluvia para cada periodo del día
        lluvia_madrugada = np.random.choice(["BAJA", "MEDIA", "ALTA"], p=[0.5, 0.35, 0.15])
        lluvia_mannana = np.random.choice(["BAJA", "MEDIA", "ALTA"], p=[0.6, 0.3, 0.1])
        lluvia_tarde = np.random.choice(["BAJA", "MEDIA", "ALTA"], p=[0.6, 0.3, 0.1])
        lluvia_noche = np.random.choice(["BAJA", "MEDIA", "ALTA"], p=[0.5, 0.35, 0.15])
        
        # Simular si el tráfico está pesado
        trafico_pesado = random.choice([True, False])
        
        # Decidir el medio de transporte basado en la probabilidad de lluvia en la tarde y el tráfico
        prob_lluvia_representativa = lluvia_tarde
        medio_transporte = decidir_transporte(prob_lluvia_representativa, trafico_pesado)
        
        data.append([
            date, zona, lluvia_madrugada, lluvia_mannana, lluvia_tarde, lluvia_noche, trafico_pesado, medio_transporte
        ])

df_historico = pd.DataFrame(data, columns=[
    "fecha", "zona", "lluvia_madrugada", "lluvia_mannana", "lluvia_tarde", "lluvia_noche", "trafico_pesado", "medio_transporte"
])

df_historico.to_csv('df_historico.csv', index=False)
