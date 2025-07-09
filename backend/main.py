# filename: app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


app = FastAPI()

# N ahora representará la cantidad de pasos de tiempo (timesteps) para la entrada LSTM.
# Es decir, cuántos valores anteriores usamos para predecir el siguiente.
N = 3

output_dir = 'ml_model'
os.makedirs(output_dir, exist_ok=True)


class ImpresoraData(BaseModel):
    id: int
    series: List[float]


@app.post("/predecir")
def predecir(data: ImpresoraData):
    try:
        # Cargar el modelo y los parámetros de normalización
        model_path = os.path.join(output_dir, 'modelo_entrenado_lstm.pkl')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Modelo LSTM no encontrado. Entrena primero el modelo LSTM.")

        model_data = joblib.load(model_path)
        modelo = model_data['model']
        global_min = model_data['global_min']
        global_max = model_data['global_max']
        N_loaded = model_data['N'] # Usamos N_loaded para mayor claridad

        serie = np.array(data.series, dtype=float)

        # Validar longitud de la serie para predicción
        if len(serie) < N_loaded:
            raise HTTPException(status_code=400, detail=f"Se requieren al menos {N_loaded} valores para predecir.")

        # Normalizar la serie de entrada
        # Asegurarse de que global_max - global_min no sea cero para evitar división por cero
        if global_max == global_min:
            # Si todos los valores son iguales, la normalización no es necesaria o implicaría una división por cero
            # En este caso, la serie_norm será igual a la serie_para_prediccion si el rango es 0
            serie_norm = serie
        else:
            serie_norm = (serie - global_min) / (global_max - global_min)

        # La entrada para el LSTM debe tener la forma (muestras, timesteps, características)
        # En nuestro caso, (1, N, 1) porque tenemos una sola muestra, N timesteps y una característica por timestep.
        entrada = serie_norm[-N_loaded:].reshape(1, N_loaded, 1)

        # Realizar la predicción
        pred_norm = modelo.predict(entrada)[0][0] # El output es (1,1), tomamos el primer elemento

        # Desnormalizar la predicción
        if global_max == global_min:
            # Si los valores originales eran todos iguales, la predicción será el mismo valor
            pred = pred_norm
        else:
            pred = pred_norm * (global_max - global_min) + global_min

        return {
            "id_impresora": data.id,
            "entrada_usada_normalizada": serie_norm[-N_loaded:].tolist(),
            "prediccion_normalizada": float(pred_norm), # Convertir a float para JSON
            "prediccion_real": float(pred) # Convertir a float para JSON
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/entrenar")
def entrenar_modelo_lstm(data: List[ImpresoraData]):
    try:
        all_series_values = []
        for impresora in data:
            if isinstance(impresora.series, list):
                all_series_values.extend(impresora.series)
            else:
                raise HTTPException(status_code=400, detail=f"La impresora con ID {impresora.id} no tiene una lista válida de series")

        if not all_series_values:
            raise HTTPException(status_code=400, detail="No se encontraron datos válidos para entrenar")

        all_series_values = np.array(all_series_values, dtype=float)
        global_min = all_series_values.min()
        global_max = all_series_values.max()

        # Preparar los datos para LSTM
        # Las entradas (X) serán secuencias de N valores, y la salida (y) será el valor siguiente
        X, y = [], []

        for impresora in data:
            serie = np.array(impresora.series, dtype=float)
            
            if global_max == global_min:
                serie_norm = serie # No se normaliza si el rango es cero
            else:
                serie_norm = (serie - global_min) / (global_max - global_min)

            if len(serie_norm) >= N + 1:
                for i in range(len(serie_norm) - N):
                    X.append(serie_norm[i:i+N])
                    y.append(serie_norm[i+N])

        if not X:
            raise HTTPException(status_code=400, detail=f"No hay suficientes datos después de procesar las secuencias. Se requieren al menos {N+1} puntos por serie.")

        X = np.array(X)
        y = np.array(y)

        # Redimensionar X para el formato LSTM: (muestras, timesteps, características)
        # En este caso, 1 característica por timestep
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Definir el modelo LSTM
        modelo = Sequential()
        modelo.add(LSTM(50, activation='relu', input_shape=(N, 1))) # 50 unidades LSTM, relu para activación, input_shape (timesteps, features)
        modelo.add(Dense(1)) # Una única salida para la predicción del siguiente valor
        modelo.compile(optimizer='adam', loss='mse') # Optimizador Adam, Mean Squared Error como función de pérdida

        # Entrenar el modelo
        modelo.fit(X, y, epochs=100, verbose=0) # Entrenar por 100 épocas, no mostrar el progreso

        model_data = {
            'model': modelo,
            'global_min': global_min,
            'global_max': global_max,
            'N': N
        }

        # Guardar el modelo Keras con joblib
        # TensorFlow/Keras models se guardan de forma diferente, pero joblib puede serializar el objeto completo.
        # Sin embargo, una práctica más robusta sería guardar el modelo Keras directamente con modelo.save()
        # Para mantener la simplicidad y consistencia con tu enfoque anterior de joblib:
        joblib.dump(model_data, os.path.join(output_dir, 'modelo_entrenado_lstm.pkl'))

        return {
            "mensaje": "Modelo LSTM entrenado y guardado exitosamente.",
            "datos_entrenamiento": {
                "series_total": len(all_series_values),
                "secuencias": len(X),
                "modelo_path": os.path.join(output_dir, 'modelo_entrenado_lstm.pkl')
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))