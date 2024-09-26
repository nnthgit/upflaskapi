from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model, encoder, and scaler
model = pickle.load(open('forest_regsq.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction de prix de maison!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Pas de données fournies'}), 400

    try:
        surface = data['surface']
        pieces = data['pieces']
        chambre = data['chambre']
        sdb = data['sdb']
        etat = data['etat']
        ville = data['ville']
        quartier = data['quartier']
    except KeyError:
        return jsonify({'error': 'Données manquantes ou mal formatées'}), 400

    # Create a DataFrame from received data
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

    # Combine encoded and scaled data
    test_house_final = np.hstack((test_house_scaled, test_house_encoded))

    # Make prediction
    prediction = model.predict(test_house_final)

    # Prepare response
    response = {
        'message': "Merci d'utiliser notre API de prédiction de prix de maison!",
        'input_data': {
            'surface': surface,
            'pieces': pieces,
            'chambre': chambre,
            'sdb': sdb,
            'etat': etat,
            'ville': ville,
            'quartier': quartier
        },
        'prediction': round(prediction[0] * 1000000, 2)  # Assuming the prediction is in millions
    }

    # Print received info and prediction result
    print("Received information:")
    print(response['input_data'])
    print(f"Predicted price: {response['prediction']} euros")

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
