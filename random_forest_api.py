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
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
CSV_PATH = "personalized_medication_dataset.csv"

# --- Firebase Admin SDK ---
firebase_cred_b64 = os.getenv("FIREBASE_CRED_BASE64")
if not firebase_cred_b64:
    raise Exception("FIREBASE_CRED_BASE64 environment variable is missing")

cred_json = base64.b64decode(firebase_cred_b64).decode("utf-8")
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- Train model from CSV ---
def train_model():
    df = pd.read_csv(CSV_PATH)

    # Basic cleaning and encoding
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    df['Chronic_Conditions'] = df['Chronic_Conditions'].fillna("None").str.lower()
    df['Recommended_Medication'] = df['Recommended_Medication'].fillna("None").str.lower()
    df['Treatment_Effectiveness'] = df['Treatment_Effectiveness'].map({
        "Very Effective": 1,
        "Effective": 1,
        "Neutral": 0,
        "Ineffective": 0
    })

    # Create mappings
    condition_map = {val: idx for idx, val in enumerate(df['Chronic_Conditions'].unique())}
    medication_map = {val: idx for idx, val in enumerate(df['Recommended_Medication'].unique())}

    df['Condition'] = df['Chronic_Conditions'].map(condition_map)
    df['Medication'] = df['Recommended_Medication'].map(medication_map)

    df = df.dropna(subset=['Age', 'Gender', 'Condition', 'Medication', 'Treatment_Effectiveness'])

    X = df[['Age', 'Gender', 'Condition', 'Medication']]
    y = df['Treatment_Effectiveness']

    # Split into train/test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Model trained from CSV and saved.")
    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📊 F1 Score: {f1:.4f}")

    joblib.dump(clf, MODEL_PATH)

    # Save mappings
    with open("condition_map.json", "w") as f:
        json.dump(condition_map, f)
    with open("medication_map.json", "w") as f:
        json.dump(medication_map, f)

train_model()
model = joblib.load(MODEL_PATH)
print("📦 Model retrained on every start.")

with open("condition_map.json") as f:
    condition_map = json.load(f)
with open("medication_map.json") as f:
    medication_map = json.load(f)

# --- Predict adherence ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        results = []

        for item in data:
            age = item.get("age", 0)
            gender = {"female": 0, "male": 1, "other": 2}.get(item.get("gender", "").lower(), 0)
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
        risk_threshold = 0.3
        results = []

        for item in data:
            age = item.get("age", 0)
            gender = {"female": 0, "male": 1, "other": 2}.get(item.get("gender", "").lower(), 0)
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

# --- Intake status check (unchanged) ---
@app.route("/check_intake_status", methods=["POST"])
def check_intake_status():
    try:
        user_id = request.json.get("user_id")
        schedules = db.collection("medication_schedules").where("user_id", "==", user_id).stream()

        result = []

        for schedule in schedules:
            data = schedule.to_dict()
            start = data['start_date'].replace(tzinfo=None)
            end = data['until_date'].replace(tzinfo=None)
            freq = int(data['frequency'])
            schedule_id = schedule.id
            med_name = data['medication_name']

            expected_intakes = []
            delta = (end - start).days + 1
            for day in range(delta):
                base = start + datetime.timedelta(days=day)
                if freq == 1:
                    expected_intakes.append(base)
                elif freq == 2:
                    expected_intakes.extend([base + datetime.timedelta(hours=8),
                                             base + datetime.timedelta(hours=20)])
                elif freq == 3:
                    expected_intakes.extend([base + datetime.timedelta(hours=8),
                                             base + datetime.timedelta(hours=14),
                                             base + datetime.timedelta(hours=20)])

            intake_docs = db.collection("medication_schedules").document(schedule_id)\
                .collection("medication_intakes").stream()
            taken_times = [doc.to_dict().get("taken_at").replace(tzinfo=None) for doc in intake_docs]

            status = []
            for expected in expected_intakes:
                matched = any(abs((expected - actual).total_seconds()) <= 3600 for actual in taken_times)
                status.append({
                    "datetime": expected.isoformat(),
                    "status": "Taken" if matched else "Missed"
                })

            result.append({
                "medication": med_name,
                "intake_status": status
            })

        return jsonify(result)

    except Exception as e:
        print("❌ Intake status check error:", e)
        return jsonify({"error": str(e)}), 500

# --- Start server ---
if __name__ == "__main__":
    print("🚀 Starting Flask server at", datetime.datetime.now())
    app.run(debug=True, host="0.0.0.0", port=5000)
