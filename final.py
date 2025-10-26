import cv2
import os
import time
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# === SETTINGS ===
image_size = 64
gestures = ['Thumbs Up', 'Hi', 'Peace', 'One Finger', 'Two Fingers']  # Modify as needed

# Create directories for each gesture if they don't exist
for gesture in gestures:
    if not os.path.exists(gesture):
        os.makedirs(gesture)

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)


# Step 1: Capture Hand Gesture Images
def capture_gestures():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Process the frame to detect hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Define region of interest (ROI) for hand
                x, y, w, h = 100, 100, 200, 200
                roi = frame[y:y + h, x:x + w]

                # Display instructions and the ROI
                cv2.putText(frame, "Press '1' for Thumbs Up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press '2' for Hi", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press '3' for Peace", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press '4' for One Finger", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press '5' for Two Fingers", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'Q' to quit", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the Region of Interest (ROI)
                cv2.imshow("Hand Gesture Capture", roi)

                # Press corresponding key to capture a gesture
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    cv2.imwrite(f'Thumbs Up/thumbs_up_{int(time.time())}.jpg', roi)
                    print("Thumbs Up Gesture Captured")
                elif key == ord('2'):
                    cv2.imwrite(f'Hi/hi_{int(time.time())}.jpg', roi)
                    print("Hi Gesture Captured")
                elif key == ord('3'):
                    cv2.imwrite(f'Peace/peace_{int(time.time())}.jpg', roi)
                    print("Peace Gesture Captured")
                elif key == ord('4'):
                    cv2.imwrite(f'One Finger/one_finger_{int(time.time())}.jpg', roi)
                    print("One Finger Gesture Captured")
                elif key == ord('5'):
                    cv2.imwrite(f'Two Fingers/two_fingers_{int(time.time())}.jpg', roi)
                    print("Two Fingers Gesture Captured")

                if key == ord('q'):  # Exit the loop on pressing 'Q'
                    break

        # Display the frame
        cv2.imshow("Capture Hand Gestures", frame)

    cap.release()
    cv2.destroyAllWindows()


# Step 2: Preprocess Captured Data and Train the Model
# noinspection PyShadowingNames
def load_and_preprocess_data():
    data = []
    labels = []
    image_size = 64
    dataset_path = r'D:\downloads\ASL PROJECT\asl_alphabet_test\asl_alphabet_test'

    # Extract class names from filenames like 'A_test.jpg'
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError("❌ No image files found in the dataset folder.")

    class_names = sorted(set(f.split('_')[0] for f in image_files))
    print(f"Detected classes: {class_names}")
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for filename in image_files:
        label_name = filename.split('_')[0]
        label_index = class_map[label_name]

        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (image_size, image_size))
        data.append(image)
        labels.append(label_index)

    if len(data) == 0:
        raise ValueError("❌ No valid images loaded. Check if images are corrupted or incorrectly named.")

    data = np.array(data, dtype='float32') / 255.0
    labels = np.array(labels)
    data = data.reshape(-1, image_size, image_size, 1)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, class_names




# Step 3: Define and Train the CNN Model
# noinspection PyShadowingNames
def train_model(x_train, x_test, y_train, y_test, class_names):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(class_names), activation='softmax')  # Output layer for gesture classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    model.save("hand_sign_model.keras")
    return model


# Step 4: Real-Time Gesture Prediction
# noinspection PyShadowingNames
def predict_gesture(model, class_names):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Define ROI for hand
                x, y, w, h = 100, 100, 200, 200
                roi = frame[y:y + h, x:x + w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (image_size, image_size)) / 255.0
                input_img = roi_resized.reshape(1, image_size, image_size, 1)

                # Predict the gesture
                prediction = model.predict(input_img)
                predicted_class = np.argmax(prediction)
                predicted_gesture = class_names[predicted_class]
                confidence = np.max(prediction)
                # Display prediction on the frame
                # Draw prediction
                cv2.putText(frame, f"{predicted_gesture} ({confidence:.2f})", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Hand Gesture Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main Flow
if __name__ == "__main__":
    # Step 1: Capture Gesture Images (Run this only once)
    # capture_gestures()

    # Step 2: Load and preprocess data
    x_train, x_test, y_train, y_test ,class_names = load_and_preprocess_data()

    # Step 3: Train the model
    model = train_model(x_train, x_test, y_train, y_test, class_names)

    # Step 4: Real-Time Prediction using the trained model
    predict_gesture(model , class_names)
