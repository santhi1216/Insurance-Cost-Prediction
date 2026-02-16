from flask import Flask, render_template, request
from joblib import load
import numpy as np
import os
from utils.model_loader import model

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        age = float(request.form['Age'])
        diabetes = float(request.form['Diabetes'])
        bp = float(request.form['BloodPressureProblems'])
        transplants = float(request.form['AnyTransplants'])
        chronic = float(request.form['AnyChronicDiseases'])
        allergies = float(request.form['KnownAllergies'])
        cancer = float(request.form['HistoryOfCancerInFamily'])
        surgeries = float(request.form['NumberOfMajorSurgeries'])
        bmi = float(request.form['BMI'])

        # Arrange in correct order
        features = np.array([[ 
            age, diabetes, bp, transplants,
            chronic, allergies, cancer,
            surgeries, bmi
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Insurance Cost: {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
