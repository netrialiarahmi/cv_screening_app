import pandas as pd
import os

def save_results(df, path="output/results.csv"):
    os.makedirs("output", exist_ok=True)
    df.to_csv(path, index=False)
