import joblib
import numpy as np

MODEL_PATH = "models/linear_model.pkl"

def main():
    model = joblib.load(MODEL_PATH)

    cgpa = float(input("Enter CGPA: ").strip())
    pred = model.predict(np.array([[cgpa]]))[0]

    print(f"🎯 Predicted Placement Package: {pred:.2f} LPA")

if __name__ == "__main__":
    main()
