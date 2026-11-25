
from flask import Flask, request, jsonify, render_template
from pathlib import Path

import pandas as pd
import joblib
import json
import math


# Initialize Flask application
app = Flask(__name__)


# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


# Load the trained machine learning model
model = joblib.load(MODEL_PATH)


# Load model performance metrics
with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)


FEATURES = metrics["features"]

CATEGORICAL_FEATURES = [
    "Neighborhood",
    "HouseStyle",
    "ExterQual"
]


# Home route
@app.route("/")
def home():
    return render_template("index.html")


# Model performance route
@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics)


# House price prediction route
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON input."
        }), 400

    missing_features = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing": missing_features
        }), 400

    try:
        house_data = {}

        for feature in FEATURES:

            if feature in CATEGORICAL_FEATURES:
                value = data[feature]

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Invalid value for {feature}"
                    )

                house_data[feature] = value.strip()

            else:
                value = float(data[feature])

                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Invalid value for {feature}"
                    )

                house_data[feature] = value

        if not 1 <= house_data["OverallQual"] <= 10:
            raise ValueError(
                "OverallQual must be between 1 and 10."
            )

        input_df = pd.DataFrame(
            [house_data],
            columns=FEATURES
        )

        prediction = float(model.predict(input_df)[0])

        return jsonify({
            "success": True,
            "predicted_price": round(prediction, 2),
            "currency": "USD",
            "model": metrics["model"]
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# Run the Flask development server
if __name__ == "__main__":
    app.run(debug=True)
