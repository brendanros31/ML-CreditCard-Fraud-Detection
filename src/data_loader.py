import pandas as pd


# Loading data
def load_data(path):
    df = pd.read_csv(path)
    return df


# Drop missing
def clean_data(df):
    return df.dropna()
