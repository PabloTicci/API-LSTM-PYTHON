# filename: app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression

app = FastAPI()

# Parámetro de ventana
N = 3

# Ruta para guardar el modelo
output_dir = 'ml_model'
os.makedirs(output_dir, exist_ok=True)

# Definición del modelo de datos que se espera
class ImpresoraData(BaseModel):
    id: int
    series: List[float]


@app.post("/predecir")
def predecir(data: ImpresoraData):
    try:
        # Verifica que exista el modelo entrenado
        model_path = os.path.join(output_dir, 'modelo_entrenado.pkl')
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Modelo no encontrado. Entrena primero el modelo.")

        # Cargar modelo entrenado y parámetros
        model_data = joblib.load(model_path)
        modelo = model_data['model']
        global_min = model_data['global_min']
        global_max = model_data['global_max']
        N = model_data['N']

        serie = np.array(data.series, dtype=float)

        # Validar longitud suficiente
        if len(serie) < N:
            raise HTTPException(status_code=400, detail=f"Se requieren al menos {N} valores para predecir.")

        # Normalizar los últimos N valores
        serie_norm = (serie - global_min) / (global_max - global_min) if global_max != global_min else serie
        entrada = serie_norm[-N:]  # Últimos N valores

        # Hacer predicción
        pred_norm = modelo.predict([entrada])[0]

        # Desnormalizar
        pred = pred_norm * (global_max - global_min) + global_min if global_max != global_min else pred_norm

        return {
            "id_impresora": data.id,
            "entrada_usada": entrada.tolist(),
            "prediccion_normalizada": pred_norm,
            "prediccion_real": pred
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/entrenar")
def entrenar_modelo(data: List[ImpresoraData]):
    try:
        # Recolectar todos los datos para normalizar
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

        # Preparar X e y
        X, y = [], []

        for impresora in data:
            serie = np.array(impresora.series, dtype=float)
            serie_norm = (serie - global_min) / (global_max - global_min) if global_max != global_min else serie

            if len(serie_norm) >= N + 1:
                for i in range(len(serie_norm) - N):
                    X.append(serie_norm[i:i+N])
                    y.append(serie_norm[i+N])

        if not X:
            raise HTTPException(status_code=400, detail="No hay suficientes datos después de procesar las secuencias")

        X = np.array(X)
        y = np.array(y)

        modelo = LinearRegression()
        modelo.fit(X, y)

        model_data = {
            'model': modelo,
            'global_min': global_min,
            'global_max': global_max,
            'N': N
        }

        joblib.dump(model_data, os.path.join(output_dir, 'modelo_entrenado.pkl'))

        return {
            "mensaje": "Modelo entrenado y guardado exitosamente.",
            "datos_entrenamiento": {
                "series_total": len(all_series_values),
                "secuencias": len(X),
                "modelo_path": os.path.join(output_dir, 'modelo_entrenado.pkl')
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
