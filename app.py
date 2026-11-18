
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# -----------------------------------------
# 1. Load saved ML model
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------------------
# 2. Dataset feature names
# -----------------------------------------
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

# Target is NOT an input feature
TARGET = "target"


# -----------------------------------------
# 3. Field information
# -----------------------------------------
FIELD_INFO = {
    "age": {
        "label": "Age",
        "kind": "number",
        "unit": "years",
        "min": 18,
        "max": 120
    },

    "sex": {
        "label": "Gender",
        "kind": "select",
        "choices": [
            ("1", "Male"),
            ("0", "Female")
        ]
    },

    "cp": {
        "label": "Chest Pain Type",
        "kind": "select",
        "choices": [
            ("0", "Type 0 - Typical Angina"),
            ("1", "Type 1 - Atypical Angina"),
            ("2", "Type 2 - Non-anginal Pain"),
            ("3", "Type 3 - Asymptomatic")
        ]
    },

    "trestbps": {
        "label": "Resting Blood Pressure",
        "kind": "number",
        "unit": "mmHg",
        "min": 50,
        "max": 300
    },

    "chol": {
        "label": "Serum Cholesterol",
        "kind": "number",
        "unit": "mg/dL",
        "min": 50,
        "max": 700
    },

    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dL",
        "kind": "select",
        "choices": [
            ("0", "No"),
            ("1", "Yes")
        ]
    },

    "restecg": {
        "label": "Resting ECG",
        "kind": "select",
        "choices": [
            ("0", "Category 0"),
            ("1", "Category 1"),
            ("2", "Category 2")
        ]
    },

    "thalach": {
        "label": "Maximum Heart Rate",
        "kind": "number",
        "unit": "beats/minute",
        "min": 30,
        "max": 250
    },

    "exang": {
        "label": "Exercise Induced Angina",
        "kind": "select",
        "choices": [
            ("0", "No"),
            ("1", "Yes")
        ]
    },

    "oldpeak": {
        "label": "Oldpeak (ST Depression)",
        "kind": "number",
        "unit": "Value",
        "min": 0,
        "max": 10
    },

    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "kind": "select",
        "choices": [
            ("0", "Category 0"),
            ("1", "Category 1"),
            ("2", "Category 2")
        ]
    },

    "ca": {
        "label": "Number of Major Vessels",
        "kind": "select",
        "choices": [
            ("0", "0"),
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
            ("4", "4")
        ]
    },

    "thal": {
        "label": "Thalassemia",
        "kind": "select",
        "choices": [
            ("0", "Category 0"),
            ("1", "Category 1"),
            ("2", "Category 2"),
            ("3", "Category 3")
        ]
    }
}


# -----------------------------------------
# 4. Check all model features are supported
# -----------------------------------------
for feature in FEATURES:
    if feature not in FIELD_INFO:
        raise ValueError(
            f"Missing FIELD_INFO configuration for: {feature}"
        )


# -----------------------------------------
# 5. Homepage
# -----------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------
# 6. Heart Disease Assessment
# -----------------------------------------
@app.route("/assessment", methods=["GET", "POST"])
def assessment():

    prediction = None
    probability = None
    error = None
    values = {}
    fields = []

    # Generate form fields
    for feature in FEATURES:
        info = FIELD_INFO[feature].copy()
        info["name"] = feature
        info["unit"] = info.get("unit", "")
        fields.append(info)

    # Process form submission
    if request.method == "POST":

        try:
            row = {}

            for feature in FEATURES:

                raw_value = request.form.get(
                    feature, ""
                ).strip()

                values[feature] = raw_value

                # All inputs are required
                if raw_value == "":
                    raise ValueError(
                        f"Please enter {FIELD_INFO[feature]['label']}."
                    )

                value = float(raw_value)

                info = FIELD_INFO[feature]

                # Validate numerical ranges
                if "min" in info and value < info["min"]:
                    raise ValueError(
                        f"{info['label']} must be at least {info['min']}."
                    )

                if "max" in info and value > info["max"]:
                    raise ValueError(
                        f"{info['label']} must not exceed {info['max']}."
                    )

                # Validate dropdown options
                if info["kind"] == "select":

                    allowed_values = [
                        float(code)
                        for code, label in info["choices"]
                    ]

                    if value not in allowed_values:
                        raise ValueError(
                            f"Invalid selection for {info['label']}."
                        )

                row[feature] = value

            # Create DataFrame in training feature order
            patient = pd.DataFrame(
                [row],
                columns=FEATURES
            )

            # Predict
            result = int(model.predict(patient)[0])

            if result == 1:
                prediction = "Heart Disease Detected by Model"
            else:
                prediction = "Heart Disease Not Detected by Model"

            # Calculate probability
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(patient)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = round(
                        float(
                            probabilities[classes.index(1)]
                        ) * 100,
                        2
                    )

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid values."

        except Exception:
            app.logger.exception("Prediction failed")

            error = (
                "Prediction failed. Check your saved model "
                "and training feature names."
            )

    return render_template(
        "model.html",
        fields=fields,
        values=values,
        prediction=prediction,
        probability=probability,
        error=error
    )


# -----------------------------------------
# 7. Run Flask
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
