import pickle
import json
from flask import Flask, request
import pandas as pd


# Cargar las características, el modelo y las equivalencias de columnas desde archivos creados con la libreria pickle
FEATURES = pickle.load(open("churn/models/features.pk", "rb"))
model = pickle.load(open("churn/models/model.pk", "rb"))
column_equivalence = pickle.load(open("churn/models/column_equivalence.pk", "rb"))


app = Flask(__name__)

def convert_numerical(features):
    """
    Función para convertir características en formato numérico.
    
    Si una característica es categórica, se reemplaza por su código numérico
    utilizando el diccionario 'column_equivalence'. Si no es categórica, 
    se intenta convertir a valor numérico con pandas. Si falla, se coloca un 0.
    
    Parámetros:
        features (list): Lista de características a convertir.
    
    Retorna:
        list: Lista de características convertidas a formato numérico.
    """
    output = []
    for i, feat in enumerate(features):
        if i in column_equivalence:
            output.append(column_equivalence[i][feat])
        else:
            try:
                output.append(pd.to_numeric(feat))
            except:
                output.append(0)
    return output

@app.route('/query')
def query_example():
    """
    Endpoint que recibe características en la URL, las convierte a formato numérico
    y devuelve la predicción del modelo en formato JSON.
    
    Parámetros (GET):
        feats: Cadena de características separadas por comas.
    
    Retorna:
        JSON: Respuesta con la predicción del modelo.
    """
     
    features = convert_numerical(request.args.get('feats').split(','))
    response = {
        'response': [int(x) for x in model.predict([features])]
    }
    return json.dumps(response)

if __name__ == '__main__':
    # Ejecutamos la aplicación Flask en modo debug en el puerto 3001
    app.run(debug=True, port=3001)