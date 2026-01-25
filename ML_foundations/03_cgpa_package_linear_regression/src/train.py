import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/cgpa_package.csv"
MODEL_PATH = "models/linear_model.pkl"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # ✅ Clean column names (fix spaces + casing issues)
    df.columns = df.columns.str.lower().str.strip()

    print("✅ Columns found:", df.columns.tolist())

    # ✅ Check required columns exist
    required_cols = {"cgpa", "package_lpa"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"❌ CSV must contain columns {required_cols}. "
            f"Currently found: {df.columns.tolist()}\n"
            f"👉 Fix your CSV header OR change code to match column names."
        )

    X = df[["cgpa"]]          # must be 2D
    y = df["package_lpa"]      # target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("\n✅ Training Complete")
    print(f"Slope (m): {model.coef_[0]:.4f}")
    print(f"Intercept (b): {model.intercept_:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

