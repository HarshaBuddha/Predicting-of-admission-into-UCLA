from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

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
        
        # Convert input values to float to handle string inputs
        for key in data:
            data[key] = float(data[key])
        
        # Convert data to DataFrame
        columns = ["GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA", "Research"]
        input_df = pd.DataFrame([data], columns=columns)

        # Standardize input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0] * 100  # Convert to percentage
        
        prediction = max(0, min(100, prediction))

        return jsonify({'admission_chance': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # By default, port number will be 5000
