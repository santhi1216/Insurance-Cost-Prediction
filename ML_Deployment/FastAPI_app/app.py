from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
import pandas as pd
from Utils.model_loader import model
app = FastAPI(title="Insurance Premium Predictor API")

feature_names = ['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
       'AnyChronicDiseases', 'KnownAllergies',
       'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries', 'BMI']

#Input schema

class InsuranceInput(BaseModel):
    Age : float
    Diabetes : int
    BloodPressureProblems : int
    AnyTransplants: int
    AnyChronicDiseases: int
    KnownAllergies: int
    HistoryOfCancerInFamily:int
    NumberOfMajorSurgeries:int
    BMI:float

# Home Route
@app.get("/")
def home():
    return {"Message": "Insurance Premium prediction API is running"}

# Prediction route
@app.post('/Predict')
def Predict(data:InsuranceInput):
    try:
        values = [[
            data.Age,
            data.Diabetes,
            data.BloodPressureProblems,
            data.AnyTransplants,
            data.AnyChronicDiseases,
            data.KnownAllergies,
            data.HistoryOfCancerInFamily,
            data.NumberOfMajorSurgeries,
            data.BMI
        ]]

        df= pd.DataFrame(values, columns = feature_names)

        # make Prediction
        prediction = model.predict(df)[0]

        return{
            "PRedicted_insurance_cost" : round(float(prediction),2)
        }


    except Exception as e:
        return {"error": str(e)}

