# Step 1: Install required packages
# pip install flask flask-cors scikit-learn pandas joblib firebase-admin

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import numpy as np
import os
import datetime
import base64
import json
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"

# --- Firebase Admin SDK (Base64-encoded key) ---
firebase_cred_b64 = os.getenv("FIREBASE_CRED_BASE64")
if not firebase_cred_b64:
    raise Exception("FIREBASE_CRED_BASE64 environment variable is missing")

cred_json = base64.b64decode(firebase_cred_b64).decode("utf-8")
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- Train model if not exists ---
def train_model():
    data = pd.DataFrame({
        'Age': [76, 45, 60, 33, 70],
        'Gender': [0, 1, 0, 1, 0],
        'Condition': [0, 1, 2, 0, 2],
        'Medication': [0, 1, 2, 0, 2],
        'Adherence': [1, 0, 1, 1, 0]
    })
    X = data[['Age', 'Gender', 'Condition', 'Medication']]
    y = data['Adherence']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    print("✅ Model trained and saved.")

if not os.path.exists(MODEL_PATH):
    train_model()

model = joblib.load(MODEL_PATH)

# --- Fetch mappings from Firestore ---
def fetch_mappings():
    condition_set = set()
    medication_set = set()

    for doc in db.collection('patients').stream():
        data = doc.to_dict()
        if 'condition' in data:
            condition_set.add(data['condition'].strip().lower())

    for doc in db.collection('medication_schedules').stream():
        data = doc.to_dict()
        if 'medication_name' in data:
            medication_set.add(data['medication_name'].strip().lower())

    condition_map = {val: idx for idx, val in enumerate(sorted(condition_set))}
    medication_map = {val: idx for idx, val in enumerate(sorted(medication_set))}

    return condition_map, medication_map

# --- Predict adherence ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        condition_map, medication_map = fetch_mappings()
        results = []

        for item in data:
            age = item.get("age", 0)
            gender = 0 if item.get("gender", "female").lower() == "female" else 1
            condition = condition_map.get(item.get("condition", "").lower(), 0)
            medication = medication_map.get(item.get("medication", "").lower(), 0)

            features = np.array([[age, gender, condition, medication]])
            prediction = model.predict(features)[0]

            results.append({
                "Medication_Name": item.get("medication"),
                "Adherence": "High" if prediction == 1 else "Low"
            })

        return jsonify(results)

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500

# --- Early warning alert ---
@app.route("/risk_alert", methods=["POST"])
def risk_alert():
    try:
        data = request.get_json()
        condition_map, medication_map = fetch_mappings()
        risk_threshold = 0.3
        results = []

        for item in data:
            age = item.get("age", 0)
            gender = 0 if item.get("gender", "female").lower() == "female" else 1
            condition = condition_map.get(item.get("condition", "").lower(), 0)
            medication = medication_map.get(item.get("medication", "").lower(), 0)

            features = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Condition": condition,
                "Medication": medication
            }])
            proba = model.predict_proba(features)[0][0]
            alert = proba > risk_threshold

            results.append({
                "Medication_Name": item.get("medication"),
                "NonAdherence_Probability": round(float(proba), 2),
                "Early_Warning_Alert": bool(alert)
            })

        return jsonify(results)

    except Exception as e:
        print("❌ /risk_alert error:", e)
        return jsonify({"error": str(e)}), 500

# --- Start server ---
if __name__ == "__main__":
    print("🚀 Starting Flask server at", datetime.datetime.now())
    app.run(debug=True, host="0.0.0.0", port=5000)
