from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import os

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "MODEL_JAMUR.keras")
model = load_model(MODEL_PATH)

# Label dan status
label_map = ["edible", "poisonous"]
status_map = {
    "edible": "Aman Dimakan",
    "poisonous": "Beracun"
}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Step 1: Buka gambar dan resize
        img = Image.open(file.stream).resize((224, 224)).convert("RGB")

        # Step 2: Preprocessing
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Step 3: Prediksi
        prediction = model.predict(img_array)[0]
        pred_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        pred_label = label_map[pred_index]
        status = status_map.get(pred_label, "Tidak Diketahui")

        return jsonify({
            "jenis_jamur": pred_label,
            "status": status,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
