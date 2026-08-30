import pandas as pd


DATA_PATH = "data/animal movement.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df