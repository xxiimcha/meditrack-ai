from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import base64
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import random
from collections import defaultdict
from flask import render_template_string

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
CSV_PATH = "personalized_medication_dataset.csv"

firebase_cred_b64 = os.getenv("FIREBASE_CRED_BASE64")
if not firebase_cred_b64:
    raise Exception("FIREBASE_CRED_BASE64 environment variable is missing")

cred_json = base64.b64decode(firebase_cred_b64).decode("utf-8")
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

def train_model():
    df = pd.read_csv(CSV_PATH)
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    df['Chronic_Conditions'] = df['Chronic_Conditions'].fillna("None").str.lower()
    df['Recommended_Medication'] = df['Recommended_Medication'].fillna("None").str.lower()
    df['Treatment_Effectiveness'] = df['Treatment_Effectiveness'].map({
        "Very Effective": 1,
        "Effective": 1,
        "Neutral": 0,
        "Ineffective": 0
    })

    condition_map = {val: idx for idx, val in enumerate(df['Chronic_Conditions'].unique())}
    medication_map = {val: idx for idx, val in enumerate(df['Recommended_Medication'].unique())}

    df['Condition'] = df['Chronic_Conditions'].map(condition_map)
    df['Medication'] = df['Recommended_Medication'].map(medication_map)
    df = df.dropna(subset=['Age', 'Gender', 'Condition', 'Medication', 'Treatment_Effectiveness'])

    X = df[['Age', 'Gender', 'Condition', 'Medication']]
    y = df['Treatment_Effectiveness']
    random_seed = random.randint(1, 9999)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, MODEL_PATH)
    with open("condition_map.json", "w") as f:
        json.dump(condition_map, f)
    with open("medication_map.json", "w") as f:
        json.dump(medication_map, f)
    with open("last_training_report.json", "w") as f:
        json.dump({
            "best_params": grid_search.best_params_,
            "accuracy": round(accuracy_score(y_test, best_model.predict(X_test)), 4),
            "f1_score": round(f1_score(y_test, best_model.predict(X_test)), 4),
            "confusion_matrix": confusion_matrix(y_test, best_model.predict(X_test)).tolist(),
            "feature_importance": dict(zip(X.columns, best_model.feature_importances_)),
            "random_seed": random_seed
        }, f, indent=2)

if "train" in __import__("sys").argv:
    train_model()

model = joblib.load(MODEL_PATH)
with open("condition_map.json") as f:
    condition_map = json.load(f)
with open("medication_map.json") as f:
    medication_map = json.load(f)

@app.route("/view_intake")
def view_intake():
    users = db.collection("users").stream()
    all_html = ""

    for user in users:
        user_id = user.id
        try:
            data = get_intake_status_for_user(user_id)
            for med in data:
                rows = ""
                for s in med["intake_status"]:
                    status_class = "taken" if s["status"] == "Taken" else "missed" if s["status"] == "Missed" else "wrong"
                    rows += f"<tr class='{status_class}'><td>{s['datetime']}</td><td>{s['status']}</td><td>{s.get('minutes_from_scheduled', '')}</td></tr>"
                all_html += f"<div class='medication-block'><h2>User: {user_id} - {med['medication']}</h2><table><tr><th>Datetime</th><th>Status</th><th>Time Diff (mins)</th></tr>{rows}</table></div>"
        except Exception as e:
            all_html += f"<p style='color:red;'>❌ Error for {user_id}: {str(e)}</p>"

    with open("intake_status_template.html") as f:
        html = f.read()

    return render_template_string(html.replace("%%CONTENT%%", all_html))

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
        return jsonify({"error": str(e)}), 500

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
        return jsonify({"error": str(e)}), 500

def get_intake_status_for_user(user_id):
    schedules = db.collection("medication_schedules").where("user_id", "==", user_id).stream()
    result = []
    for schedule in schedules:
        data = schedule.to_dict()
        start = data['start_date'].replace(tzinfo=None)
        end = data['until_date'].replace(tzinfo=None)
        freq = int(data['frequency'])
        interval_hr = int(data.get("interval_per_hour", 8))
        med_name = data['medication_name']
        schedule_id = schedule.id
        expected_intakes = []
        
        first_time = datetime.datetime.strptime(data['first_intake_time'], "%H:%M").time()

        for day in range((end - start).days + 1):
            base_date = start.date() + datetime.timedelta(days=day)
            base_datetime = datetime.datetime.combine(base_date, first_time)

            for i in range(freq):
                intake_time = base_datetime + datetime.timedelta(hours=interval_hr * i)
                expected_intakes.append(intake_time)

        intake_docs = db.collection("medication_schedules").document(schedule_id).collection("medication_intakes").stream()
        taken_logs = sorted([doc.to_dict() for doc in intake_docs if "taken_at" in doc.to_dict()], key=lambda x: x["taken_at"])
        taken_times = [log["taken_at"].replace(tzinfo=None) for log in taken_logs]

        daily_expected = defaultdict(int)
        daily_actual = defaultdict(int)
        for expected in expected_intakes:
            daily_expected[expected.date()] += 1
        for actual in taken_times:
            daily_actual[actual.date()] += 1

        daily_summary = []
        for day in sorted(daily_expected.keys()):
            daily_summary.append({
                "date": day.isoformat(),
                "scheduled_doses": daily_expected[day],
                "actual_doses_taken": daily_actual.get(day, 0),
                "doses_remaining": daily_expected[day] - daily_actual.get(day, 0)
            })

        status = []
        for expected in expected_intakes:
            match_log = None
            for log in taken_logs:
                taken_time = log["taken_at"].replace(tzinfo=None)
                if abs((expected - taken_time).total_seconds()) <= 3600:
                    match_log = log
                    break
            if match_log:
                taken_time = match_log["taken_at"].replace(tzinfo=None)
                diff_min = int((taken_time - expected).total_seconds() / 60)
                med_match = match_log.get("medication_name", "") == med_name
                status.append({
                    "datetime": expected.isoformat(),
                    "status": "Taken" if med_match else "Wrong Medication",
                    "actual_time": taken_time.isoformat(),
                    "minutes_from_scheduled": diff_min,
                    "within_window": abs(diff_min) <= 30,
                    "medication_match": med_match
                })
            else:
                status.append({
                    "datetime": expected.isoformat(),
                    "status": "Missed",
                    "within_window": False,
                    "medication_match": False
                })

        interval_hours = []
        for i in range(1, len(taken_times)):
            hours_gap = round((taken_times[i] - taken_times[i-1]).total_seconds() / 3600, 2)
            interval_hours.append(hours_gap)

        result.append({
            "medication": med_name,
            "intake_status": status,
            "daily_summary": daily_summary,
            "intervals_between_doses_hours": interval_hours[-5:]
        })
    return result

@app.route("/check_intake_status", methods=["POST"])
def check_intake_status():
    try:
        user_id = request.json.get("user_id")
        return jsonify(get_intake_status_for_user(user_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server at", datetime.datetime.now())
    preview_user_id = "JvHMPasYewUsAxjwKPR6B7OJ0r2"
    preview = get_intake_status_for_user(preview_user_id)
    print(json.dumps(preview, indent=2, default=str))
    app.run(debug=True, host="0.0.0.0", port=5000)