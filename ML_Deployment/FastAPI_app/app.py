from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from joblib import load
import numpy as np
import pandas as pd
import os
from utils.model_loader import model
app = FastAPI(title="Insurance Premium Predictor API")

feature_names = ['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
       'AnyChronicDiseases', 'KnownAllergies',
       'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries', 'BMI']

#Input schema

class InsuranceInput(BaseModel):
    Age: float = Field(...,ge=1,le=100)
    Diabetes: Literal[0, 1]
    BloodPressureProblems: Literal[0, 1]
    AnyTransplants: Literal[0, 1]
    AnyChronicDiseases: Literal[0, 1]
    KnownAllergies: Literal[0, 1]
    HistoryOfCancerInFamily: Literal[0, 1]
    NumberOfMajorSurgeries: Literal[0,1,2]
    BMI: float = Field(...,ge=10,le=60)

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
            "Predicted_insurance_cost" : round(float(prediction),2)
        }


    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)