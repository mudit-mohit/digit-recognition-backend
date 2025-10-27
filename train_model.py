import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# ------------------------------
# 1️⃣ LOAD AND PREPARE DATA
# ------------------------------
train_data = pd.read_csv('Train.csv')
print("Shape of train_data:", train_data.shape)

X = train_data.iloc[:, 1:].values / 255.0
y = to_categorical(train_data.iloc[:, 0], num_classes=10)

X = X.reshape(-1, 28, 28, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 2️⃣ BUILD CNN MODEL
# ------------------------------
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
model.summary()

# ------------------------------
# 3️⃣ DATA AUGMENTATION
# ------------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

# ------------------------------
# 4️⃣ TRAIN MODEL
# ------------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10
)

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# ------------------------------
# 5️⃣ SAVE MODEL
# ------------------------------
model.save("digit_recognition_model.keras")
print("✅ Model saved successfully!")

# ------------------------------
# 6️⃣ LOAD MODEL
# ------------------------------
model = load_model("digit_recognition_model.keras")

# ------------------------------
# 7️⃣ IMAGE PREPROCESS FUNCTION
# ------------------------------
def preprocess_digit(frame):
    """Convert image or ROI to MNIST-compatible."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for better digit extraction
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thresh
    
    # Get the largest contour
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filter out very small contours
    if w < 10 or h < 10:
        return None, thresh
    
    # Extract digit with some padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(thresh.shape[1] - x, w + 2 * padding)
    h = min(thresh.shape[0] - y, h + 2 * padding)
    
    digit = thresh[y:y+h, x:x+w]
    
    # Create square image with aspect ratio preservation
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    
    # Center the digit in the square
    if h > w:
        pad = (h - w) // 2
        square[:, pad:pad+w] = digit
    else:
        pad = (w - h) // 2
        square[pad:pad+h, :] = digit
    
    # Add border padding to match MNIST style
    border_size = int(size * 0.2)
    square_with_border = cv2.copyMakeBorder(square, border_size, border_size, 
                                             border_size, border_size, 
                                             cv2.BORDER_CONSTANT, value=0)
    
    # Resize to 28x28
    digit_resized = cv2.resize(square_with_border, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize
    digit_normalized = digit_resized.astype('float32') / 255.0
    digit_normalized = digit_normalized.reshape(1, 28, 28, 1)
    
    return digit_normalized, digit_resized

# ------------------------------
# 🎥 CAMERA DETECTION FUNCTION
# ------------------------------
def get_available_cameras(max_tested=10):
    """Detect available camera indices."""
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

# ------------------------------
# 📹 CAMERA SELECTION
# ------------------------------
def select_camera():
    """Let user select camera from available options."""
    print("\n🔍 Detecting available cameras...")
    cameras = get_available_cameras()
    
    if not cameras:
        print("❌ No cameras detected!")
        return None
    
    print(f"\n✅ Found {len(cameras)} camera(s): {cameras}")
    
    if len(cameras) == 1:
        print(f"📷 Using camera index {cameras[0]}")
        return cameras[0]
    
    print("\n📷 Available cameras:")
    for idx in cameras:
        print(f"  [{idx}] Camera {idx}")
    
    while True:
        try:
            choice = input(f"\nSelect camera index {cameras}: ")
            camera_idx = int(choice)
            if camera_idx in cameras:
                return camera_idx
            else:
                print(f"❌ Invalid choice. Please select from {cameras}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Camera selection cancelled")
            return None

# ------------------------------
# 8️⃣ STATIC IMAGE TEST
# ------------------------------
image_path = "Screenshot 2025-10-23 at 3.19.42 PM.png"
if os.path.exists(image_path):
    frame = cv2.imread(image_path)
    processed, debug_img = preprocess_digit(frame)
    if processed is not None:
        pred = model.predict(processed)
        digit = np.argmax(pred)
        print(f"Predicted digit from image '{image_path}': {digit}")

        cv2.imshow("Original", frame)
        cv2.imshow("Processed (MNIST Style)", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Could not detect a digit.")
else:
    print("⚠️ Image file not found.")

# ------------------------------
# 9️⃣ LIVE CAMERA FEED
# ------------------------------
camera_index = select_camera()

if camera_index is None:
    print("❌ No camera selected. Exiting...")
    exit()

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"❌ Could not open camera {camera_index}")
    exit()

print(f"\n🎥 Live Digit Recognition started with Camera {camera_index}")
print("🟢 Press 'q' or 'ESC' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read from camera. Exiting...")
        break

    # Define Region of Interest
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)

    result = preprocess_digit(roi)
    if result is not None:
        processed, debug_img = result
        pred = model.predict(processed)
        digit = np.argmax(pred)
        cv2.putText(frame, f"Prediction: {digit}", (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No digit detected", (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show instruction overlay
    cv2.putText(frame, "Press 'q' or 'ESC' to quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display live feed
    cv2.imshow("Live Digit Recognition", frame)

    # ---- Exit conditions ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("👋 Exiting live feed...")
        break

# ---- Cleanup ----
cap.release()
cv2.destroyAllWindows()