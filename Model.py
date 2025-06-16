from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="rumah_adat_final2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = {
    0: "Bukan Rumah Adat",
    1: "Rumah adat Bajawa",
    2: "Rumah adat Ende",
    3: "Rumah adat Pulau Timur",
    4: "Rumah adat Sumba",
    5: "Rumah adat Waraebo"
}

def preprocess_image(img_bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Rumah Adat Classification API",
        "status": "running",
        "endpoints": {
            "predict": {
                "url": "/predict",
                "method": "POST",
                "content-type": "multipart/form-data"
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        input_data = preprocess_image(img_bytes)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return jsonify({
            "class": predicted_class,
            "class_name": class_names.get(predicted_class, "Unknown"),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)