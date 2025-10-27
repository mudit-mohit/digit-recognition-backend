import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np
import os

def convert_existing_model():
    # Path to your existing trained model
    keras_model_path = "/Users/appdev/Downloads/Handwritten-Digit-Recognition/backend/digit_recognition_model.keras"
    onnx_model_path = "digit_model.onnx"
    
    # Check if the model exists
    if not os.path.exists(keras_model_path):
        print(f"❌ Keras model not found at: {keras_model_path}")
        return None
    
    print(f"🔄 Loading existing Keras model from: {keras_model_path}")
    
    try:
        # Load your existing trained model with custom objects if needed
        model = tf.keras.models.load_model(keras_model_path)
        print("✅ Keras model loaded successfully!")
        
        # Print model summary to verify
        model.summary()
        
        # Convert to ONNX using a different approach
        print("🔄 Converting to ONNX format...")
        
        # Get the model's input shape
        input_shape = model.input_shape
        print(f"📊 Model input shape: {input_shape}")
        
        # Create a concrete function for conversion
        @tf.function
        def model_func(x):
            return model(x)
        
        # Create input signature
        input_spec = tf.TensorSpec((None, 28, 28, 1), tf.float32, name='input')
        
        # Convert using tf2onnx
        model_proto, _ = tf2onnx.convert.from_function(
            model_func,
            input_signature=[input_spec],
            output_path=onnx_model_path,
            opset=13
        )
        
        print(f"✅ Model converted to ONNX and saved as: {onnx_model_path}")
        
        # Test the ONNX model to make sure it works
        print("🔄 Testing ONNX model...")
        session = ort.InferenceSession(onnx_model_path)
        
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"📊 ONNX input name: {input_name}, shape: {session.get_inputs()[0].shape}")
        print(f"📊 ONNX output name: {output_name}, shape: {session.get_outputs()[0].shape}")
        
        # Create a test input
        test_input = np.random.random((1, 28, 28, 1)).astype(np.float32)
        
        # Run prediction
        result = session.run([output_name], {input_name: test_input})
        print(f"✅ ONNX model test successful! Prediction shape: {result[0].shape}")
        print(f"📊 Sample prediction: {np.argmax(result[0])}")
        
        return onnx_model_path
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    convert_existing_model()