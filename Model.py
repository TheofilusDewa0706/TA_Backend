from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import math
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load TFLite model with error handling
try:
    logger.info("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path="rumah_adat_final_97.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

# Class names
CLASS_NAMES = [
    "Rumah adat Bajawa",
    "Rumah adat Ende",
    "Rumah adat Pulau Timur", 
    "Rumah adat Sumba",
    "Rumah adat Waraebo"
]

# Prediction thresholds
MIN_CONFIDENCE = 0.70  # Minimum confidence score to accept prediction
MAX_ENTROPY = 1.0      # Maximum entropy to accept prediction

def calculate_entropy(predictions):
    """Calculate prediction uncertainty using Shannon entropy"""
    predictions = np.clip(predictions, 1e-10, 1.0)  # Avoid log(0)
    return -np.sum(predictions * np.log(predictions))

def preprocess_image(img_bytes):
    """
    Process uploaded image for model prediction
    Args:
        img_bytes: Binary image data
    Returns:
        Processed image tensor
    Raises:
        ValueError: If image processing fails
    """
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
            
        # Resize and normalize image
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.route('/')
def home():
    """API root endpoint"""
    return jsonify({
        "message": "Rumah Adat Classification API",
        "status": "active",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Upload image for classification",
                "content-type": "multipart/form-data"
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image classification requests
    Returns:
        JSON response with classification results or error message
    """
    # Validate file upload
    if 'file' not in request.files:
        logger.warning("No file uploaded")
        return jsonify({
            "status": "error",
            "message": "Harus mengunggah file gambar"
        }), 400
        
    file = request.files['file']
    if not file or file.filename == '':
        logger.warning("Empty file uploaded")
        return jsonify({
            "status": "error",
            "message": "Nama file tidak boleh kosong"
        }), 400

    try:
        # Read and validate image
        img_bytes = file.read()
        if len(img_bytes) == 0:
            logger.warning("Empty image file")
            return jsonify({
                "status": "error",
                "message": "File gambar kosong"
            }), 400

        logger.info(f"Processing image: {file.filename}")
        
        # Preprocess image
        input_data = preprocess_image(img_bytes)
        
        # Run model inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Normalize and convert predictions
        predictions = predictions.astype(np.float64)
        predictions = predictions / np.sum(predictions)  # Ensure probabilities sum to 1
        
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        entropy = float(calculate_entropy(predictions))

        logger.info(f"Prediction results - Class: {predicted_class}, Confidence: {confidence:.2f}, Entropy: {entropy:.2f}")

        # Prepare response
        if confidence < MIN_CONFIDENCE or entropy > MAX_ENTROPY:
            response = {
                "status": "rejected",
                "class_name": "Bukan rumah adat NTT",
                "message": "Gambar tidak dikenali sebagai rumah adat NTT",
                "confidence": round(confidence, 4),
                "entropy": round(entropy, 4)
            }
        else:
            response = {
                "status": "success",
                "class": predicted_class,
                "class_name": CLASS_NAMES[predicted_class],
                "confidence": round(confidence, 4),
                "entropy": round(entropy, 4)
            }

        return jsonify(response)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Terjadi kesalahan pada server",
            "error_details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)