import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_PATH = "data/cgpa_package.csv"
MODEL_PATH = "models/linear_model.pkl"
OUT_PATH = "plots/regression_line.png"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df["cgpa"].values.reshape(-1, 1)
    y = df["package_lpa"].values

    model = joblib.load(MODEL_PATH)

    # Line points
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    # Plot
    plt.scatter(X, y)
    plt.plot(x_line, y_line)
    plt.xlabel("CGPA")
    plt.ylabel("Package (LPA)")
    plt.title("CGPA vs Package (Simple Linear Regression)")

    Path("plots").mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    plt.show()

    print(f"✅ Plot saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
