from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import math

app = Flask(__name__)
CORS(app)  # Simplified CORS configuration

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="rumah_adat_final_97.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Class names
class_names = [
    "Rumah adat Bajawa",
    "Rumah adat Ende",
    "Rumah adat Pulau Timur",
    "Rumah adat Sumba",
    "Rumah adat Waraebo"
]

# Thresholds
MIN_CONFIDENCE = 0.70
MAX_ENTROPY = 1.0  # Adjust based on your testing

def calculate_entropy(predictions):
    """Calculate prediction uncertainty"""
    predictions = np.clip(predictions, 1e-10, 1.0)  # Avoid log(0)
    return -np.sum(predictions * np.log(predictions))

def preprocess_image(img_bytes):
    """Process uploaded image for model prediction"""
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
            
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.route('/')
def home():
    return jsonify({
        "message": "Rumah Adat Classification API",
        "status": "active",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Upload image for classification"
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    try:
        # Read and validate image
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Empty image file"}), 400

        # Process and predict
        input_data = preprocess_image(img_bytes)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Convert numpy types to Python native types
        predictions = predictions.astype(np.float64)
        predictions = predictions / np.sum(predictions)  # Normalize
        
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        entropy = float(calculate_entropy(predictions))

        # Determine result
        if confidence < MIN_CONFIDENCE or entropy > MAX_ENTROPY:
            response = {
                "status": "rejected",
                "message": "Not recognized as NTT traditional house",
                "confidence": round(confidence, 4),
                "entropy": round(entropy, 4)
            }
        else:
            response = {
                "status": "success",
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