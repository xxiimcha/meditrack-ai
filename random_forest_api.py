# Step 1: Install required packages
# pip install flask flask-cors scikit-learn pandas joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import numpy as np
import os
import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"

def train_model():
    # Sample dataset
    data = pd.DataFrame({
        'Age': [76, 45, 60, 33, 70],
        'Gender': [0, 1, 0, 1, 0],  # 0: Female, 1: Male
        'Condition': [0, 1, 2, 0, 2],  # e.g., 0: Hypertension, 1: Diabetes, etc.
        'Medication': [0, 1, 2, 0, 2],  # e.g., 0: Ibuprofen, 1: Metformin, etc.
        'Adherence': [1, 0, 1, 1, 0]  # 1: High, 0: Low
    })

    X = data[['Age', 'Gender', 'Condition', 'Medication']]
    y = data['Adherence']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    print("✅ Model trained and saved.")

# Load or train model
if not os.path.exists(MODEL_PATH):
    train_model()

model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔍 Incoming data: {data}")
        results = []

        for item in data:
            age = item.get("age", 0)
            gender = 0 if item.get("gender", "female").lower() == "female" else 1
            condition_map = {"hypertension": 0, "diabetes": 1, "asthma": 2}
            medication_map = {"ibuprofen": 0, "metformin": 1, "amlodipine": 2}

            condition = condition_map.get(item.get("condition", "hypertension").lower(), 0)
            medication = medication_map.get(item.get("medication", "ibuprofen").lower(), 0)

            features = np.array([[age, gender, condition, medication]])
            print(f"📊 Features: {features.tolist()}")

            prediction = model.predict(features)[0]
            print(f"✅ Prediction: {'High' if prediction == 1 else 'Low'}")

            results.append({
                "Medication_Name": item.get("medication"),
                "Adherence": "High" if prediction == 1 else "Low"
            })

        print(f"📤 Response: {results}")
        return jsonify(results)

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route("/risk_alert", methods=["POST"])
def risk_alert():
    try:
        data = request.get_json()
        risk_threshold = 0.3  # You can tune this threshold

        results = []
        for item in data:
            age = item.get("age", 0)
            gender = 0 if item.get("gender", "female").lower() == "female" else 1
            condition_map = {"hypertension": 0, "diabetes": 1, "asthma": 2}
            medication_map = {"ibuprofen": 0, "metformin": 1, "amlodipine": 2}

            condition = condition_map.get(item.get("condition", "hypertension").lower(), 0)
            medication = medication_map.get(item.get("medication", "ibuprofen").lower(), 0)

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
                "NonAdherence_Probability": round(float(proba), 2),  # ensure float type
                "Early_Warning_Alert": bool(alert)  # convert to native Python bool
            })

        return jsonify(results)

    except Exception as e:
        print("❌ /risk_alert error:", e)
        traceback.print_exc()  # <-- Add this to log the exact error and line number
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server at", datetime.datetime.now())
    app.run(debug=True, host="0.0.0.0", port=5000)
