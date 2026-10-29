
import cv2
from keras.models import model_from_json
import numpy as np

# Load model architecture
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load trained weights
model.load_weights("emotiondetector.h5")
print("Model loaded successfully!")

# Load Haar cascade
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image, dtype=np.float32)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Start webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, im = webcam.read()

        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Detect faces using grayscale image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]

            # Draw rectangle around face
            cv2.rectangle(
                im, (p, q), (p+r, q+s), (255, 0, 0), 2
            )

            # Resize and preprocess face
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)

            # Predict emotion
            pred = model.predict(img, verbose=0)
            prediction_label = labels[int(np.argmax(pred[0]))]

            # Display predicted emotion
            cv2.putText(
                im,
                prediction_label,
                (p, max(q-10, 20)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.5,
                (0, 0, 255),
                2
            )

        cv2.imshow("Face Emotion Detection", im)

        # Press Q or ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()
