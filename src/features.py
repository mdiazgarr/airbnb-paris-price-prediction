# src/features.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the Airbnb dataset and return X, y, and pipeline.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df = df.copy()

    # Drop irrelevant or ID columns
    df.drop(columns=["id", "host_id", "name", "host_name", "license"], inplace=True, errors='ignore')

    # Fill missing reviews_per_month with 0
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # Select target
    y = df["log_price"]
    X = df.drop(columns=["price", "log_price"])

    # Define features
    categorical_features = ["room_type", "neighbourhood"]
    numerical_features = ["minimum_nights", "number_of_reviews", "reviews_per_month",
                          "latitude", "longitude", "calculated_host_listings_count", "availability_365"]

    # Preprocessing pipelines
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor
