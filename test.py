from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Charger le modèle, l'encodeur et le scaler
model = pickle.load(open('forest_regsq.pkl', 'rb'))

print(type(model)) 
encoder = pickle.load(open('encoder.pkl', 'rb'))  # L'encodeur doit être pré-enregistré
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Le scaler doit être pré-enregistré

@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction de maison !"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Pas de données fournies'}), 400

    # Récupérer les données envoyées
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

    # Créer un DataFrame à partir des données reçues
    test_house = pd.DataFrame({
        'surface': [surface],
        'pieces': [pieces],
        'chambre': [chambre],
        'sdb': [sdb],
        'etat': [etat],
        'ville': [ville],
        'quartier': [quartier]
    })

    print(test_house)


    # Encodage des variables catégoriques 'etat', 'ville' et 'quartier' avec transform() (sans réajustement)
    test_house_encoded = encoder.transform(test_house[['etat', 'ville', 'quartier']])

    # Normalisation des variables numériques avec transform()
    test_house_scaled = scaler.transform(test_house[['surface', 'pieces', 'chambre', 'sdb']])

    # Combiner les données encodées et normalisées
    test_house_final = np.hstack((test_house_scaled, test_house_encoded))


    print(test_house_final)

    # Faire la prédiction

    prediction = model.predict(test_house_final)


    # Retourner la prédiction sous forme JSON
    print(prediction[0])
    return jsonify({'prediction': prediction[0] * 1000000})

if __name__ == '__main__':
    app.run(debug=True)
