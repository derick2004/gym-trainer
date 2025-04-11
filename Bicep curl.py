import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # Text-to-speech library
import time  # Time module to manage intervals
import threading

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed
engine.setProperty('volume', 1.0)  # Max volume

mp_pose = mp.solutions.pose
# Use static_image_mode=False for video stream processing
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
   
    ba = a - b
    bc = c - b
   
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
   
    return np.degrees(angle)

def speak_feedback(feedback):
    engine.say(feedback)
    engine.runAndWait()

# Open the webcam with a lower resolution for processing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a larger window for display
cv2.namedWindow('Bicep Curl Counter', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Bicep Curl Counter', 1920, 1080)

right_curl_count = 0
left_curl_count = 0
right_stage = None
left_stage = None
last_feedback = ""
last_feedback_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
   
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Process the lower resolution frame
    processed_frame = cv2.resize(frame, (640, 480))
    image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   
    feedback = ""
    feedback_color = (0, 255, 0)  # Default to green for perfect form
   
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
       
        # Right arm keypoints
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
       
        # Left arm keypoints
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
       
        # Hip keypoints for stability check
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
       
        # Calculate angles
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
       
        # Curl detection and counting
        if right_angle > 160:
            right_stage = "down"
        if right_angle < 30 and right_stage == "down":
            right_stage = "up"
            right_curl_count += 1
       
        if left_angle > 160:
            left_stage = "down"
        if left_angle < 30 and left_stage == "down":
            left_stage = "up"
            left_curl_count += 1
       
        # Form validation using differences between keypoints
        if (abs(right_elbow[0] - right_shoulder[0]) > 0.08 or 
            abs(left_elbow[0] - left_shoulder[0]) > 0.08 or
            abs(right_shoulder[0] - right_hip[0]) > 0.15 or 
            abs(left_shoulder[0] - left_hip[0]) > 0.15):
            feedback = "Incorrect Form - Keep Your Elbow Straight"
            feedback_color = (0, 0, 255)  # Red for errors
        else:
            feedback = "Perfect Form"
            feedback_color = (0, 255, 0)  # Green
       
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
   
    # Provide voice feedback every 5 seconds if the message has changed
    current_time = time.time()
    if (current_time - last_feedback_time >= 5) and (feedback != last_feedback):
        threading.Thread(target=speak_feedback, args=(feedback,)).start()
        last_feedback = feedback
        last_feedback_time = current_time

    # Scale the processed image up for display
    display_image = cv2.resize(image, (1920, 1080))
   
    # Display feedback bar and text on the output image
    cv2.rectangle(display_image, (0, 0), (1920, 50), feedback_color, -1)
    cv2.putText(display_image, feedback, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(display_image, f"Right Arm: {right_curl_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(display_image, f"Left Arm: {left_curl_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
   
    cv2.imshow('Bicep Curl Counter', display_image)
   
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
