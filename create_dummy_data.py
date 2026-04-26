import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="prepare_data/data/SMOKE_data.csv")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    t = np.arange(args.n)
    rng = np.random.default_rng(20)
    df = pd.DataFrame({
        "load": 50 + 5*np.sin(2*np.pi*t/24) + 0.05*t + rng.normal(0, 0.5, args.n),
        "temperature": 20 + 8*np.sin(2*np.pi*t/96) + rng.normal(0, 0.3, args.n),
        "windspeed": 3 + np.sin(2*np.pi*t/36) + rng.normal(0, 0.2, args.n),
        "humidity": 60 + 10*np.sin(2*np.pi*t/72) + rng.normal(0, 1.0, args.n),
        "water": rng.normal(0, 1, args.n),
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)
    print(f"Dummy data saved to {args.out}")


if __name__ == "__main__":
    main()
