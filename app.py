
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"

model = joblib.load(MODEL_PATH)

if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = [
        "SEX", "AGE", "TOTCHOL", "SYSBP", "DIABP",
        "CURSMOKE", "CIGPDAY", "BMI", "DIABETES",
        "BPMEDS", "HEARTRTE", "GLUCOSE", "educ", "PREVHYP"
    ]

# User-friendly labels, units, and basic input limits
FIELD_INFO = {
    "SEX": {
        "label": "Gender",
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
    "CURSMOKE": {
        "label": "Do you currently smoke?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    },
    "CIGPDAY": {
        "label": "Cigarettes per day",
        "min": 0,
        "max": 100
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "min": 10,
        "max": 80
    },
    "DIABETES": {
        "label": "Have you been diagnosed with diabetes?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    },
    "BPMEDS": {
        "label": "Do you take blood-pressure medication?",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    },
    "HEARTRTE": {
        "label": "Resting Heart Rate",
        "unit": "beats/minute",
        "min": 20,
        "max": 250
    },
    "GLUCOSE": {
        "label": "Blood Glucose",
        "unit": "mg/dL",
        "min": 20,
        "max": 700
    },
    "educ": {
        "label": "Education Level (dataset category)",
        "kind": "select",
        "choices": [
            ("1", "Category 1"),
            ("2", "Category 2"),
            ("3", "Category 3"),
            ("4", "Category 4")
        ]
    },
    "PREVHYP": {
        "label": "History of Hypertension",
        "kind": "select",
        "choices": [("0", "No"), ("1", "Yes")]
    }
}

# Add a readable label for any other feature used by the model.
for feature in FEATURES:
    if feature not in FIELD_INFO:
        FIELD_INFO[feature] = {
            "label": feature.replace("_", " ").title()
        }


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    values = {}
    fields = []

    for feature in FEATURES:
        info = FIELD_INFO[feature].copy()
        info["name"] = feature
        info["unit"] = info.get("unit", "")
        info["kind"] = info.get("kind", "number")
        fields.append(info)

    if request.method == "POST":
        try:
            row = {}

            for feature in FEATURES:
                raw_value = request.form.get(feature, "").strip()
                values[feature] = raw_value

                if raw_value == "":
                    row[feature] = float("nan")
                else:
                    value = float(raw_value)

                    info = FIELD_INFO[feature]
                    if "min" in info and value < info["min"]:
                        raise ValueError(
                            f"{info['label']} must be at least {info['min']}."
                        )
                    if "max" in info and value > info["max"]:
                        raise ValueError(
                            f"{info['label']} must not exceed {info['max']}."
                        )

                    if info.get("kind") == "select":
                        allowed = [float(code) for code, _ in info["choices"]]
                        if value not in allowed:
                            raise ValueError(
                                f"Please select a valid option for {info['label']}."
                            )

                    row[feature] = value

            patient = pd.DataFrame([row], columns=FEATURES)

            result = int(model.predict(patient)[0])
            prediction = f"Predicted class: {result}"

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(patient)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = round(
                        float(probabilities[classes.index(1)]) * 100, 2
                    )

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid values."
        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check model features, preprocessing, "
                "and the input values."
            )

    return render_template(
        "index.html",
        fields=fields,
        values=values,
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
