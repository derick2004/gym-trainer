import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import time
import pyttsx3
from FormFix_Classifier import FormFixClassifier

# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
lm_dict = {
    0: 0, 1: 10, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15, 8: 24, 9: 26, 10: 28, 11: 23, 12: 25, 13: 27, 14: 5, 15: 2, 16: 8, 17: 7,
}

def set_pose_parameters():
    return False, 1, True, False, True, 0.5, 0.5, mp.solutions.pose

def get_pose(img, results):
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    return img

def get_position(img, results):
    landmark_list = []
    if results.pose_landmarks:
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            landmark_list.append([id, int(landmark.x * w), int(landmark.y * h)])
    return landmark_list

def get_angle(img, landmark_list, point1, point2, point3):
    x1, y1 = landmark_list[point1][1:]
    x2, y2 = landmark_list[point2][1:]
    x3, y3 = landmark_list[point3][1:]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return angle if angle >= 0 else angle + 360

def run_full_pushup_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, 
                           elbow_angle_right, shoulder_angle_right, hip_angle_right, 
                           pushup_success_percentage, feedback, rep_counted, last_feedback_spoken, engine, last_feedback_time):
    current_time = time.time()
    
    # Condition for correct form
    correct_form = (
        elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and
        elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160
    )

    # Check for incorrect posture (hip too low or too high)
    if hip_angle < 160 or hip_angle_right < 160 or hip_angle > 200 or hip_angle_right > 200:
        new_feedback = "Keep your body straight."
    else:
        # If form is correct, update feedback
        if correct_form:
            new_feedback = "Good form."
        else:
            new_feedback = feedback  # Keep the last feedback if no change is needed

    # Rep Counting Logic
    if form == 1:
        if direction == 0:  # Moving down
            if elbow_angle <= 90 and hip_angle > 160 and elbow_angle_right <= 90 and hip_angle_right > 160:
                direction = 1  # Start moving up
                rep_counted = False  # Reset rep_counted when starting a new rep
        elif direction == 1:  # Moving up
            if correct_form:  # If full extension is reached
                if not rep_counted:
                    count += 1  # Count rep when full up position is achieved
                    rep_counted = True
                direction = 0  # Reset for next rep

    # Speak feedback only if 5 seconds have passed since the last feedback
    if (current_time - last_feedback_time) >= 5:
        if new_feedback != last_feedback_spoken:  # Prevent repeated feedback
            engine.say(new_feedback)
            engine.runAndWait()
            last_feedback_spoken = new_feedback
            last_feedback_time = current_time  # Update feedback timestamp

    return new_feedback, count, rep_counted, last_feedback_spoken, direction, last_feedback_time

def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon)
    cap = cv2.VideoCapture(0)
    count, direction, form, feedback = 0, 0, 0, "Bad Form."
    frame_queue = deque(maxlen=250)
    clf = FormFixClassifier(r"D:\FormFix\formfix1.tflite")
    rep_counted = False
    last_feedback_spoken = None
    engine = pyttsx3.init()
    last_feedback_time = 0  # Initialize last feedback time

    # Set OpenCV window to full screen
    cv2.namedWindow('Workout Tracker', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Workout Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = get_pose(img, results)
        landmark_list = get_position(img, results)
        if len(landmark_list) != 0:
            # Debugging: Print landmark list to verify data
            print("Landmark List:", landmark_list)
            
            # Calculate angles
            elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
            shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
            hip_angle = get_angle(img, landmark_list, 11, 23, 25)
            elbow_angle_right = get_angle(img, landmark_list, 12, 14, 16)
            shoulder_angle_right = get_angle(img, landmark_list, 14, 12, 24)
            hip_angle_right = get_angle(img, landmark_list, 12, 24, 26)
            
            # Debugging: Print angles to verify calculations
            print(f"Elbow Angle: {elbow_angle}, Shoulder Angle: {shoulder_angle}, Hip Angle: {hip_angle}")
            print(f"Right Elbow Angle: {elbow_angle_right}, Right Shoulder Angle: {shoulder_angle_right}, Right Hip Angle: {hip_angle_right}")
            
            pushup_success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
            form = 1 if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 else form
            feedback, count, rep_counted, last_feedback_spoken, direction, last_feedback_time = run_full_pushup_motion(
                count, direction, form, elbow_angle, shoulder_angle, hip_angle,
                elbow_angle_right, shoulder_angle_right, hip_angle_right,
                pushup_success_percentage, feedback, rep_counted, last_feedback_spoken, engine, last_feedback_time
            )
        cv2.putText(img, f'Reps: {count}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, feedback, (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.imshow('Workout Tracker', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()