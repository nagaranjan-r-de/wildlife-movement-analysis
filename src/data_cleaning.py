import pandas as pd


def clean_data(df):
    df = df.copy()

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing GPS coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    return df