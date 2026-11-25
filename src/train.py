from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "house_prices.csv"

MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# Load dataset
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()


# Select features
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "BedroomAbvGr",
    "LotArea",
    "Neighborhood",
    "HouseStyle",
    "ExterQual"
]


X = df[FEATURES].copy()
y = df["SalePrice"].copy()


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Identify numerical and categorical features
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# Create Gradient Boosting model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]
)


# Train model
print("Training Gradient Boosting model...")

model.fit(X_train, y_train)


# Generate predictions
y_pred = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# Save trained model
model_path = MODELS_DIR / "house_price_model.joblib"

joblib.dump(model, model_path)


# Save performance metrics
metrics = {
    "model": "Gradient Boosting",
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "features": FEATURES
}


with open(
    MODELS_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(metrics, file, indent=4)


# Display results
print("\nModel trained successfully!")

print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

print(f"\nModel saved to: {model_path}")
