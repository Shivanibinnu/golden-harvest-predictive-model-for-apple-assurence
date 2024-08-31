from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('rfc.pkl', 'rb'))
import joblib
scaler = StandardScaler()
Scaler=joblib.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template("output.html")

@app.route('/submit', methods=["POST", "GET"])
def submit():
    if request.method == "POST":
        # Read the inputs given by the user
        animal_name = request.form['animalName']
        symptoms = [
            request.form['symptoms1'],
            request.form['symptoms2'],
            request.form['symptoms3'],
            request.form['symptoms4'],
            request.form['symptoms5']
        ]
        
        # Example: Convert inputs to appropriate format
        input_features = [animal_name] + symptoms
        input_features = np.array(input_features).reshape(1, -1)
        
        # Scale the features
        input_features = scaler.transform(input_features)
        
        # Predict using the loaded model
        prediction = model.predict(input_features)
        
        result = "Your health is in normal condition." if prediction == 1 else "According to our study, we feel sad."
        
        return render_template("output.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
    debug=True,post=='0.0.0.0'