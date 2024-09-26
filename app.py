
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model, encoder, and scaler
model = joblib.load('model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        surface = data['surface']
        pieces = data['pieces']
        chambre = data['chambre']
        sdb = data['sdb']
        etat = data['etat']
        ville = data['ville']
        quartier = data['quartier']
    except KeyError:
        return jsonify({'error': 'Missing or incorrectly formatted data'}), 400

    test_house = pd.DataFrame({
        'surface': [surface],
        'pieces': [pieces],
        'chambre': [chambre],
        'sdb': [sdb],
        'etat': [etat],
        'ville': [ville],
        'quartier': [quartier]
    })

    # Encode categorical variables
    test_house_encoded = encoder.transform(test_house[['etat', 'ville', 'quartier']])

    # Scale numeric variables
    test_house_scaled = scaler.transform(test_house[['surface', 'pieces', 'chambre', 'sdb']])

    # Combine the encoded and scaled data
    test_house_final = np.hstack((test_house_scaled, test_house_encoded))

    # Make the prediction
    prediction = model.predict(test_house_final)

    return jsonify({'prediction': prediction[0] * 1000000})

if __name__ == '__main__':
    app.run(debug=True)


