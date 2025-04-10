from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load("model.pkl")  # Load your trained model
scaler = joblib.load("scaler.pkl")  # Load the StandardScaler used in training

@app.route('/', methods=['GET'])
def sample():
    return jsonify({'message': "Hello all, this is a Flask Server"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Convert input values to float
        for key in data:
            data[key] = float(data[key])

        # Convert to DataFrame
        columns = ["GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA", "Research"]
        input_df = pd.DataFrame([data], columns=columns)

        # Standardize and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0] * 100
        prediction = max(0, min(100, prediction))

        return jsonify({'admission_chance': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
