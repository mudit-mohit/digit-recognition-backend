from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import os
from werkzeug.utils import secure_filename
import onnxruntime as ort

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "https://digit-recognition-front.vercel.app"]}})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'digit_model.onnx'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the ONNX model
def setup_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ ONNX model not found. Please run convert_existing_model.py first.")
        return None
    
    try:
        # Try different providers for compatibility
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        print("✅ ONNX model loaded successfully!")
        
        # Print detailed model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        print(f"📊 Model input: {input_info.name}, shape: {input_info.shape}")
        print(f"📊 Model output: {output_info.name}, shape: {output_info.shape}")
        
        return session
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        return None

model_session = setup_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_digit(image):
    """Convert image to MNIST-compatible format."""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        # Get the largest contour
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out very small contours
        if w < 10 or h < 10:
            return None, None
        
        # Extract digit with padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(thresh.shape[1] - x, w + 2 * padding)
        h = min(thresh.shape[0] - y, h + 2 * padding)
        
        digit = thresh[y:y+h, x:x+w]
        
        # Create square image
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        
        # Center the digit
        if h > w:
            pad = (h - w) // 2
            square[:, pad:pad+w] = digit
        else:
            pad = (w - h) // 2
            square[pad:pad+h, :] = digit
        
        # Add border padding
        border_size = int(size * 0.2)
        square_with_border = cv2.copyMakeBorder(
            square, border_size, border_size, 
            border_size, border_size, 
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Resize to 28x28
        digit_resized = cv2.resize(square_with_border, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize
        digit_normalized = digit_resized.astype('float32') / 255.0
        
        # Reshape based on model input requirements
        if model_session:
            input_shape = model_session.get_inputs()[0].shape
            if len(input_shape) == 4 and input_shape[1] == 1:
                # NCHW format
                digit_processed = digit_normalized.reshape(1, 1, 28, 28)
            else:
                # NHWC format
                digit_processed = digit_normalized.reshape(1, 28, 28, 1)
        else:
            # Default to NHWC
            digit_processed = digit_normalized.reshape(1, 28, 28, 1)
        
        return digit_processed, digit_resized
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, None

def predict_digit(processed_image):
    """Make prediction using the ONNX model."""
    if model_session is None:
        return None, None, None
    
    try:
        input_name = model_session.get_inputs()[0].name
        output_name = model_session.get_outputs()[0].name
        
        # Ensure input matches model expectations
        input_shape = model_session.get_inputs()[0].shape
        if len(input_shape) == 4 and input_shape[1] == 1 and processed_image.shape[-1] == 1:
            # Convert from NHWC to NCHW if needed
            if len(processed_image.shape) == 4 and processed_image.shape[1] == 28:
                processed_image = np.transpose(processed_image, (0, 3, 1, 2))
        
        predictions = model_session.run([output_name], {input_name: processed_image})
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get all probabilities
        all_probabilities = {
            str(i): float(predictions[0][0][i] * 100) 
            for i in range(10)
        }
        
        return predicted_digit, confidence, all_probabilities
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None, None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_session is not None
    })

@app.route('/api/predict/camera', methods=['POST'])
def predict_from_camera():
    """Predict digit from camera frame."""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Preprocess and predict
        processed, debug_image = preprocess_digit(image)
        
        if processed is None:
            return jsonify({
                'success': False,
                'message': 'No digit detected in the frame'
            })
        
        digit, confidence, all_probs = predict_digit(processed)
        
        if digit is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'success': True,
            'digit': digit,
            'confidence': confidence,
            'probabilities': all_probs
        })
    
    except Exception as e:
        print(f"Error in camera prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/upload', methods=['POST'])
def predict_from_upload():
    """Predict digit from uploaded image file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Preprocess and predict
        processed, debug_image = preprocess_digit(image)
        
        if processed is None:
            return jsonify({
                'success': False,
                'message': 'No digit detected in the image'
            })
        
        digit, confidence, all_probs = predict_digit(processed)
        
        if digit is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Convert debug image to base64 for frontend display
        _, buffer = cv2.imencode('.png', debug_image)
        debug_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'digit': digit,
            'confidence': confidence,
            'probabilities': all_probs,
            'processed_image': f'data:image/png;base64,{debug_base64}'
        })
    
    except Exception as e:
        print(f"Error in upload prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/base64', methods=['POST'])
def predict_from_base64():
    """Predict digit from base64 encoded image."""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Preprocess and predict
        processed, debug_image = preprocess_digit(image)
        
        if processed is None:
            return jsonify({
                'success': False,
                'message': 'No digit detected in the image'
            })
        
        digit, confidence, all_probs = predict_digit(processed)
        
        if digit is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'success': True,
            'digit': digit,
            'confidence': confidence,
            'probabilities': all_probs
        })
    
    except Exception as e:
        print(f"Error in base64 prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Digit Recognition API Server")
    print("="*50)
    print("📡 Server running on: http://localhost:5000")
    print("🏥 Health check: http://localhost:5000/api/health")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)