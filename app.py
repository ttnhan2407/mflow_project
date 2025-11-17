from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.pyfunc
import numpy as np

app = Flask(__name__)
CORS(app) 

print("🔄 Loading model...")
model = mlflow.pyfunc.load_model("best_model")
print("✅ Model loaded!")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        water = data["water"]
        light = data["light"]
        input_data = np.array([[water, light]])
        prediction = model.predict(input_data)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return {"message": "Water-Light prediction API running!"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
