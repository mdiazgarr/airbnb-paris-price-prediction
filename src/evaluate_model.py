import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from data_loader import load_airbnb_data
from features import preprocess_data

def evaluate_model():
    print("Loading model...")
    model = joblib.load("models/trained_model.pkl")

    print("Loading and preparing test data...")
    df = load_airbnb_data("data/raw/listings.csv")
    df = df[df["price"] < 1000]
    df["log_price"] = np.log1p(df["price"])

    _, X_test, _, y_test, _ = preprocess_data(df)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("Plotting results...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual log(price)")
    plt.ylabel("Predicted log(price)")
    plt.title(f"Actual vs Predicted Prices\nR² = {r2_score(y_test, y_pred):.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
