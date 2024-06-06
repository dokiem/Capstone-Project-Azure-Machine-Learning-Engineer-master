import json
import numpy as np
import os
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('hd-model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array([[sample['Pregnancies'], sample['Glucose'], sample['BloodPressure'], sample['SkinThickness'], 
                          sample['Insulin'], sample['BMI'], sample['DiabetesPedigreeFunction'], sample['Age']] 
                         for sample in data])
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
