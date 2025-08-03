# src/train_model.py

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from data_loader import load_airbnb_data
from features import preprocess_data

def train_model():
    print("Loading data...")
    df = load_airbnb_data("data/raw/listings.csv")
    df = df[df["price"] < 1000]  # remove extreme outliers
    df["log_price"] = np.log1p(df["price"])

    print("Preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("Training model...")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 4)
    }


    print("Metrics:", metrics)

    os.makedirs("models", exist_ok=True)

    print("Saving model and metrics...")
    joblib.dump(model, "models/trained_model.pkl")
    with open("models/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Done!")

if __name__ == "__main__":
    train_model()
