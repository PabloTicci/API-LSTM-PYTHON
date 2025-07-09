# python_scripts/modelo.py

import json
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Parámetro: cuántos valores usar para predecir el siguiente
N = 3

# Cargar datos históricos
with open('storage/app/series_impresoras.json', 'r') as f:
    datos = json.load(f)

X = []
y = []

for impresora in datos:
    serie = np.array(impresora['series'], dtype=float)
    minimo = serie.min()
    maximo = serie.max()
    # Normaliza si hay más de un valor distinto
    if maximo != minimo:
        serie_norm = (serie - minimo) / (maximo - minimo)
    else:
        serie_norm = serie
    if len(serie_norm) >= N + 1:
        for i in range(len(serie_norm) - N):
            X.append(serie_norm[i:i+N])
            y.append(serie_norm[i+N])

if not X:
    raise Exception("No hay suficientes datos históricos para entrenar el modelo.")

X = np.array(X)
y = np.array(y)

modelo = LinearRegression()
modelo.fit(X, y)

joblib.dump(modelo, 'ml_model/modelo_entrenado.pkl')
print("Modelo entrenado y guardado correctamente.")