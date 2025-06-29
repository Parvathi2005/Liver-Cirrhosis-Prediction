from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ------------------ Load Model & Scaler ------------------
try:
    model = joblib.load("models/rf_model.pkl")  # You can change to xgb_model.pkl or lgbm_model.pkl
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError as e:
    raise FileNotFoundError("❌ 'rf_model.pkl' or 'scaler.pkl' missing in 'models/' folder.") from e

# ------------------ Home Page ------------------
@app.route("/")
def index():
    return render_template("index.html")

# ------------------ Prediction Route ------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Input field order must match model training
        fields = [
            "Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
            "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins",
            "Albumin", "Albumin_and_Globulin_Ratio", "Gender"
        ]

        # Extract and convert values
        features = []
        for field in fields:
            val = request.form.get(field)
            if not val or val.strip() == "":
                return render_template("index.html", result=f"❌ Missing value: {field}")
            try:
                features.append(float(val))
            except ValueError:
                return render_template("index.html", result=f"❌ Invalid number in: {field}")

        # Scale input
        scaled_input = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled_input)[0]
        result = "⚠️ Liver Disease Detected" if prediction == 1 else "✅ No Liver Disease Detected"

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"❌ Error: {str(e)}")

# ------------------ Run the App ------------------
if __name__ == "__main__":
    app.run(debug=True)
