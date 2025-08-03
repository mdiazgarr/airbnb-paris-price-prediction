import pandas as pd
import os

def load_airbnb_data(file_path: str) -> pd.DataFrame:
    """
    Load Airbnb data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded Airbnb dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    return df

def basic_info(df: pd.DataFrame):
    """
    Print basic info and null values of a DataFrame.
    """
    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False))
    print("\nColumn types:\n", df.dtypes)
