import sys
import json
import joblib
import numpy as np

modelo = joblib.load('ml_model/modelo_entrenado.pkl')


ruta_json = sys.argv[1]


with open(ruta_json, 'r') as f:
    datos_impresoras = json.load(f)


resultados = []

N = 3  

for item in datos_impresoras:
    idimpresora = item['idimpresora'] # Llamado de id que se espera en el JSON
    serie = item['series'] # Llamado de series que se espera en el JSON

    valores = np.array(serie).reshape(-1, 1)
    minimo = valores.min()
    maximo = valores.max()
    if maximo != minimo:
        valores_norm = (valores - minimo) / (maximo - minimo)
    else:
        valores_norm = valores


    if len(valores_norm) >= N:
        entrada = valores_norm[-N:].reshape(1, -1)
        prediccion = modelo.predict(entrada)
        
        if maximo != minimo:
            prediccion_real = prediccion[0] * (maximo - minimo) + minimo
        else:
            prediccion_real = prediccion[0]
    else:
        prediccion_real = None  

    resultados.append({
        'idimpresora': idimpresora, 
        'prediccion': prediccion_real
    })


print(json.dumps(resultados))
