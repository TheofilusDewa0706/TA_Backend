import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
CORS(app)

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="rumah_adat_final_97.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Class names dictionary
class_names = {
    0: "Rumah adat Bajawa",
    1: "Rumah adat Ende", 
    2: "Rumah adat Pulau Timur",
    3: "Rumah adat Sumba",
    4: "Rumah adat Waraebo"
}

# Thresholds
MIN_CONFIDENCE = 0.70
MAX_ENTROPY = 1.0

def calculate_entropy(predictions):
    predictions = np.clip(predictions, 1e-10, 1.0)
    return -np.sum(predictions * np.log(predictions))

def preprocess_image(img_bytes):
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
            
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    try:
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Empty image file"}), 400

        # Process and predict
        input_data = preprocess_image(img_bytes)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Normalize predictions
        predictions = predictions.astype(np.float64)
        predictions = predictions / np.sum(predictions)

        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        entropy = float(calculate_entropy(predictions))

        # Prepare response
        response = {
            "class": predicted_class,
            "class_name": class_names[predicted_class],
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4)
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)