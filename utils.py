import pandas as pd
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_df(df, path):
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)

def load_df(path):
    return pd.read_csv(path, parse_dates=True)
