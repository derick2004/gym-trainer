import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

# Generate all columns of the data frame
HEADERS = ["label"]  # Label column
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

def extract_important_keypoints(results) -> list:
    """Extract important keypoints from Mediapipe pose detection"""
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

def rescale_frame(frame, percent=100):
    """Resize frame to full screen"""
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# Load model
with open(r"D:\FormFix\plank\LR_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

# Load input scaler
with open(r"D:\FormFix\plank\input_scaler.pkl", "rb") as f2:
    input_scaler = pickle.load(f2)

# Transform prediction into class
def get_class(prediction: float) -> str:
    return {0: "C", 1: "H", 2: "L"}.get(prediction)

cap = cv2.VideoCapture(0)
current_stage = ""
prediction_probability_threshold = 0.6

# Set OpenCV Window to Full Screen
cv2.namedWindow("Workout Analysis", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Workout Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Resize frame to full screen
        image = rescale_frame(image, 100)

        # Convert BGR to RGB for Mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            continue

        # Convert RGB back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

        # Make detection
        try:
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=HEADERS[1:])
            X = pd.DataFrame(input_scaler.transform(X))

            predicted_class = sklearn_model.predict(X)[0]
            predicted_class = get_class(predicted_class)
            prediction_probability = sklearn_model.predict_proba(X)[0]

            # Evaluate model prediction
            if predicted_class == "C" and prediction_probability.max() >= prediction_probability_threshold:
                current_stage = "Correct"
            elif predicted_class == "L" and prediction_probability.max() >= prediction_probability_threshold:
                current_stage = "Low back"
            elif predicted_class == "H" and prediction_probability.max() >= prediction_probability_threshold:
                current_stage = "High back"
            else:
                current_stage = "Unknown"

            # Status box
            cv2.rectangle(image, (0, 0), (300, 70), (245, 117, 16), -1)

            # Display class
            cv2.putText(image, "CLASS", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display probability
            cv2.putText(image, "PROB", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"{round(prediction_probability.max(), 2)}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        # Show full-screen window
        cv2.imshow("Workout Analysis", image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)
