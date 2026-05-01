import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
def main():
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"
    load_dotenv(env_path)

    df = pd.read_csv(os.getenv("DATASET_PATH"))

    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

if __name__ == "__main__":
    main()