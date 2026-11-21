from pathlib import Path
import joblib
import pandas as pd
import shap
from flask import Flask, render_template, request

app = Flask(__name__)

# -----------------------------------------
# 1. Load saved ML artifacts
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "chd_rf_best_model (1).pkl"
SCALER_PATH = BASE_DIR / "models" / "minmax_scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns (1).pkl"

model = joblib.load(MODEL_PATH)
model_scaler = joblib.load(SCALER_PATH)

if FEATURES_PATH.exists():
    FEATURES = list(joblib.load(FEATURES_PATH))
elif hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = [
        "SEX", "AGE", "TOTCHOL", "SYSBP", "DIABP", "PULSE_PRESSURE",
        "CIGPDAY", "BMI", "BPMEDS", "HEARTRTE", "GLUCOSE", "DIABETES_GRADE",
        "PREVCHD", "PREVAP", "PREVMI", "PREVHYP"
    ]

explainer = shap.TreeExplainer(model)

# -----------------------------------------
# 2. User-facing form fields (14 base inputs)
# -----------------------------------------
INPUT_FIELDS = [
    "SEX", "AGE", "TOTCHOL", "SYSBP", "DIABP",
    "CIGPDAY", "BMI", "BPMEDS", "HEARTRTE", "GLUCOSE",
    "PREVCHD", "PREVAP", "PREVMI", "PREVHYP"
]

DEFAULT_VALUES = {
    "SEX": 1.0,
    "AGE": 54.0,
    "TOTCHOL": 238.0,
    "SYSBP": 132.0,
    "DIABP": 82.0,
    "CIGPDAY": 0.0,
    "BMI": 25.48,
    "BPMEDS": 0.0,
    "HEARTRTE": 75.0,
    "GLUCOSE": 80.0,
    "PREVCHD": 0.0,
    "PREVAP": 0.0,
    "PREVMI": 0.0,
    "PREVHYP": 0.0
}

FIELD_INFO = {
    "SEX": {
        "label": "Sex",
        "kind": "select",
        "choices": [("1", "Male"), ("2", "Female")]
    },
    "AGE": {
        "label": "Age",
        "unit": "years",
        "min": 18,
        "max": 120
    },
    "TOTCHOL": {
        "label": "Total Cholesterol",
        "unit": "mg/dL",
        "min": 50,
        "max": 700
    },
    "SYSBP": {
        "label": "Systolic Blood Pressure",
        "unit": "mmHg",
        "min": 50,
        "max": 300
    },
    "DIABP": {
        "label": "Diastolic Blood Pressure",
        "unit": "mmHg",
        "min": 20,
        "max": 200
    },
    "CIGPDAY": {
        "label": "Cigarettes per day",
        "unit": "cigarettes",
        "min": 0,
        "max": 100
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "min": 10,
        "max": 80
    },
    "BPMEDS": {
        "label": "On Blood Pressure Medication?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    },
    "HEARTRTE": {
        "label": "Resting Heart Rate",
        "unit": "bpm",
        "min": 20,
        "max": 250
    },
    "GLUCOSE": {
        "label": "Fasting Blood Glucose",
        "unit": "mg/dL",
        "min": 20,
        "max": 700
    },
    "PREVCHD": {
        "label": "Previous Coronary Heart Disease (CHD)?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    },
    "PREVAP": {
        "label": "History of Angina Pectoris (Chest Pain)?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    },
    "PREVMI": {
        "label": "Previous Myocardial Infarction (Heart Attack)?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    },
    "PREVHYP": {
        "label": "History of Hypertension?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    }
}

FEATURE_LABELS = {
    "SEX": "Sex",
    "AGE": "Age",
    "TOTCHOL": "Total Cholesterol",
    "SYSBP": "Systolic BP",
    "DIABP": "Diastolic BP",
    "PULSE_PRESSURE": "Pulse Pressure",
    "CIGPDAY": "Cigarettes/Day",
    "BMI": "Body Mass Index",
    "BPMEDS": "BP Medication",
    "HEARTRTE": "Resting Heart Rate",
    "GLUCOSE": "Blood Glucose",
    "DIABETES_GRADE": "Diabetes Grade",
    "PREVCHD": "Previous CHD History",
    "PREVAP": "Previous Angina",
    "PREVMI": "Previous Heart Attack",
    "PREVHYP": "Hypertension History"
}


def get_diabetes_grade(glucose: float) -> float:
    if glucose < 100:
        return 1.0
    elif glucose < 126:
        return 2.0
    elif glucose < 200:
        return 3.0
    return 4.0


def build_risk_factors_analysis(row: dict, shap_series: pd.Series) -> list[dict]:
    """Builds the structured Risk Factors Analysis cards shown on the left column of result.html."""
    factors = []

    # 1. Age
    age = int(row["AGE"])
    if age >= 60:
        factors.append({"title": "Age", "subtitle": f"Age {age} - Higher age-related risk group", "status": "high", "shap": round(float(shap_series.get("AGE", 0)), 4)})
    elif age >= 45:
        factors.append({"title": "Age", "subtitle": f"Age {age} - Moderate risk age group", "status": "moderate", "shap": round(float(shap_series.get("AGE", 0)), 4)})
    else:
        factors.append({"title": "Age", "subtitle": f"Age {age} - Lower age-related risk group", "status": "low", "shap": round(float(shap_series.get("AGE", 0)), 4)})

    # 2. Gender
    if int(row["SEX"]) == 1:
        factors.append({"title": "Gender", "subtitle": "Male - Statistically higher baseline CHD risk", "status": "moderate", "shap": round(float(shap_series.get("SEX", 0)), 4)})
    else:
        factors.append({"title": "Gender", "subtitle": "Female - Gender-related risk is currently lower", "status": "low", "shap": round(float(shap_series.get("SEX", 0)), 4)})

    # 3. Blood Pressure
    sysbp, diabp = int(row["SYSBP"]), int(row["DIABP"])
    bp_shap = round(float(shap_series.get("SYSBP", 0) + shap_series.get("DIABP", 0)), 4)
    if sysbp >= 140 or diabp >= 90:
        factors.append({"title": "Blood Pressure", "subtitle": f"{sysbp}/{diabp} mmHg - Hypertension stage", "status": "high", "shap": bp_shap})
    elif sysbp >= 120 or diabp >= 80:
        factors.append({"title": "Blood Pressure", "subtitle": f"{sysbp}/{diabp} mmHg - Elevated / Pre-hypertension", "status": "moderate", "shap": bp_shap})
    else:
        factors.append({"title": "Blood Pressure", "subtitle": f"{sysbp}/{diabp} mmHg - Normal range", "status": "low", "shap": bp_shap})

    # 4. Cholesterol
    chol = int(row["TOTCHOL"])
    chol_shap = round(float(shap_series.get("TOTCHOL", 0)), 4)
    if chol >= 240:
        factors.append({"title": "Cholesterol", "subtitle": f"{chol} mg/dL - High cholesterol level", "status": "high", "shap": chol_shap})
    elif chol >= 200:
        factors.append({"title": "Cholesterol", "subtitle": f"{chol} mg/dL - Borderline high", "status": "moderate", "shap": chol_shap})
    else:
        factors.append({"title": "Cholesterol", "subtitle": f"{chol} mg/dL - Desirable range", "status": "low", "shap": chol_shap})

    # 5. Blood Sugar
    gluc = int(row["GLUCOSE"])
    gluc_shap = round(float(shap_series.get("GLUCOSE", 0)), 4)
    if gluc >= 126:
        factors.append({"title": "Blood Sugar", "subtitle": f"{gluc} mg/dL - Diabetic range", "status": "high", "shap": gluc_shap})
    elif gluc >= 100:
        factors.append({"title": "Blood Sugar", "subtitle": f"{gluc} mg/dL - Pre-diabetic range", "status": "moderate", "shap": gluc_shap})
    else:
        factors.append({"title": "Blood Sugar", "subtitle": f"{gluc} mg/dL - Normal fasting glucose", "status": "low", "shap": gluc_shap})

    # 6. Body Mass Index
    bmi = round(row["BMI"], 1)
    bmi_shap = round(float(shap_series.get("BMI", 0)), 4)
    if bmi >= 30:
        factors.append({"title": "Body Mass Index", "subtitle": f"BMI {bmi} - Obese category", "status": "high", "shap": bmi_shap})
    elif bmi >= 25:
        factors.append({"title": "Body Mass Index", "subtitle": f"BMI {bmi} - Overweight", "status": "moderate", "shap": bmi_shap})
    else:
        factors.append({"title": "Body Mass Index", "subtitle": f"BMI {bmi} - Healthy weight range", "status": "low", "shap": bmi_shap})

    # 7. Smoking
    cigs = int(row["CIGPDAY"])
    cig_shap = round(float(shap_series.get("CIGPDAY", 0)), 4)
    if cigs > 0:
        factors.append({"title": "Smoking", "subtitle": f"Current smoker ({cigs} cigs/day) - Major risk factor", "status": "high", "shap": cig_shap})
    else:
        factors.append({"title": "Smoking", "subtitle": "Non-smoker - Low smoking risk", "status": "low", "shap": cig_shap})

    # 8. Resting Heart Rate
    hr = int(row["HEARTRTE"])
    hr_shap = round(float(shap_series.get("HEARTRTE", 0)), 4)
    if hr > 100 or hr < 50:
        factors.append({"title": "Resting Heart Rate", "subtitle": f"{hr} bpm - Outside normal resting range", "status": "moderate", "shap": hr_shap})
    else:
        factors.append({"title": "Resting Heart Rate", "subtitle": f"{hr} bpm - Normal resting pulse", "status": "low", "shap": hr_shap})

    # 9. Clinical History
    has_history = any(int(row[k]) == 1 for k in ["PREVCHD", "PREVAP", "PREVMI", "PREVHYP"])
    hist_shap = round(float(sum(shap_series.get(k, 0) for k in ["PREVCHD", "PREVAP", "PREVMI", "PREVHYP"])), 4)
    if int(row["PREVCHD"]) == 1 or int(row["PREVMI"]) == 1 or int(row["PREVAP"]) == 1:
        factors.append({"title": "Clinical Cardiac History", "subtitle": "Prior CHD, Angina, or MI history present", "status": "high", "shap": hist_shap})
    elif int(row["PREVHYP"]) == 1:
        factors.append({"title": "Clinical Cardiac History", "subtitle": "History of hypertension present", "status": "moderate", "shap": hist_shap})
    else:
        factors.append({"title": "Clinical Cardiac History", "subtitle": "No prior cardiovascular events reported", "status": "low", "shap": hist_shap})

    return factors


def generate_recommendations(row: dict, risk_band: str) -> list[str]:
    """Generates personalized numbered preventive recommendations for the right column."""
    recs = [
        "Regular health check-ups are increasingly important as you age"
    ]

    if row["TOTCHOL"] >= 200:
        recs.append("Focus on a heart-healthy diet (low saturated/trans fats) to lower cholesterol")
    if row["GLUCOSE"] >= 100:
        recs.append("Monitor blood sugar regularly and reduce refined carbohydrates")
    if row["BMI"] >= 25.0:
        recs.append("Aim for gradual weight loss through balanced nutrition and 150 mins/week aerobic exercise")
    if row["CIGPDAY"] > 0:
        recs.append("Quitting smoking is the single most important step for heart health")
        recs.append("Seek structured smoking cessation support programs")
    if row["SYSBP"] >= 130 or row["DIABP"] >= 85:
        recs.append("Adopt a low-sodium DASH diet (<2g sodium/day) and track blood pressure daily")
    if any(int(row[k]) == 1 for k in ["PREVCHD", "PREVAP", "PREVMI", "PREVHYP"]) or risk_band == "High Risk":
        recs.append("Regular cardiac screenings and ECG/Lipid evaluations are important given your clinical profile")

    recs.append("Maintain a balanced, heart-healthy diet rich in fruits, vegetables, and whole grains")
    recs.append("Stay hydrated and manage daily stress through relaxation techniques")
    recs.append("Schedule regular follow-up check-ups with your healthcare provider")
    return recs[:10]


# -----------------------------------------
# 3. Homepage
# -----------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------
# 4. Health assessment and prediction -> renders result.html
# -----------------------------------------
@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    error = None
    values = {}
    fields = []

    for feature in INPUT_FIELDS:
        info = FIELD_INFO[feature].copy()
        info["name"] = feature
        info["unit"] = info.get("unit", "")
        info["kind"] = info.get("kind", "number")
        fields.append(info)

    if request.method == "POST":
        try:
            row = {}

            for feature in INPUT_FIELDS:
                raw_value = request.form.get(feature, "").strip()
                values[feature] = raw_value

                if raw_value == "":
                    row[feature] = DEFAULT_VALUES[feature]
                    continue

                value = float(raw_value)
                info = FIELD_INFO[feature]

                if "min" in info and value < info["min"]:
                    raise ValueError(f"{info['label']} must be at least {info['min']}.")
                if "max" in info and value > info["max"]:
                    raise ValueError(f"{info['label']} must not exceed {info['max']}.")

                if info.get("kind") == "select":
                    allowed_values = [float(code) for code, _ in info["choices"]]
                    if value not in allowed_values:
                        raise ValueError(f"Please select a valid option for {info['label']}.")

                row[feature] = value

            # Compute engineered features
            row["PULSE_PRESSURE"] = row["SYSBP"] - row["DIABP"]
            row["DIABETES_GRADE"] = get_diabetes_grade(row["GLUCOSE"])

            # Align exact feature order and scale
            patient_df = pd.DataFrame([row])[FEATURES]
            scaled_array = model_scaler.transform(patient_df)
            scaled_df = pd.DataFrame(scaled_array, columns=FEATURES)

            # Predict probability & class on SCALED features
            result = int(model.predict(scaled_df)[0])
            probability = 0
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled_df)[0]
                classes = list(model.classes_)
                if 1 in classes:
                    probability = int(round(float(probabilities[classes.index(1)]) * 100))

            if probability >= 60 or result == 1 and probability >= 50:
                risk_level = "High Risk"
                risk_theme = "high"
            elif probability >= 30:
                risk_level = "Moderate Risk"
                risk_theme = "moderate"
            else:
                risk_level = "Low Risk"
                risk_theme = "low"

            # Compute SHAP Feature Contributions
            shap_vals = explainer.shap_values(scaled_df)
            patient_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0, :, 1]
            shap_series = pd.Series(patient_shap, index=FEATURES)

            top_5_cols = shap_series.abs().sort_values(ascending=False).head(5).index
            shap_top5 = []
            for col in top_5_cols:
                imp = float(shap_series[col])
                shap_top5.append({
                    "feature": FEATURE_LABELS.get(col, col),
                    "value": round(row[col], 1),
                    "impact": round(imp, 4),
                    "direction": "Increases Risk" if imp > 0 else "Reduces Risk"
                })

            risk_factors = build_risk_factors_analysis(row, shap_series)
            recommendations = generate_recommendations(row, risk_level)

            # Render the dedicated result.html dashboard
            return render_template(
                "result.html",
                risk_level=risk_level,
                risk_theme=risk_theme,
                probability=probability,
                risk_factors=risk_factors,
                recommendations=recommendations,
                shap_top5=shap_top5
            )

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid values."

        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Please check the model artifacts and input values."

    return render_template(
        "model.html",
        fields=fields,
        values=values,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)