import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import tf2onnx
import onnxruntime as ort
import tensorflow as tf

def create_and_convert_model():
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Preprocess
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Build model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, verbose=1)
    
    # Convert to ONNX
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    output_path = "digit_model.onnx"
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    print(f"✅ Model converted to ONNX and saved as {output_path}")
    
    # Test the ONNX model
    session = ort.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test prediction
    test_input = X_test[0:1]
    result = session.run([output_name], {input_name: test_input})
    print(f"Test prediction: {np.argmax(result[0])}")
    
    return output_path

if __name__ == "__main__":
    create_and_convert_model()