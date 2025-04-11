import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import tensorflow as tf  # Import TensorFlow Lite

# Define the FormFixClassifier class
class FormFixClassifier:
    def __init__(self, model_path):
        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Define the labels for the model output
        self.LABELS = ["pushups"]

    def predict(self, input_data):
        # Ensure input_data is a numpy array of type float32
        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        champ_idx = np.argmax(output_data)

        # Return the corresponding label
        if champ_idx < len(self.LABELS):
            return self.LABELS[champ_idx]
        else:
            return ""  # Return empty string if index is out of range

# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
lm_dict = {
    0: 0, 1: 10, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15, 8: 24, 9: 26, 10: 28, 11: 23, 12: 25, 13: 27, 14: 5, 15: 2, 16: 8, 17: 7,
}

def set_pose_parameters():
    mode = False
    complexity = 1
    smooth_landmarks = True
    enable_segmentation = False
    smooth_segmentation = True
    detectionCon = 0.5
    trackCon = 0.5
    mpPose = mp.solutions.pose
    return mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose

def get_pose(img, results, draw=True):
    if results.pose_landmarks:
        if draw:
            mpDraw = mp.solutions.drawing_utils
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img

def get_position(img, results, height, width, draw=True):
    landmark_list = []
    if results.pose_landmarks:
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            # Finding height, width of the image printed
            height, width, c = img.shape
            # Determining the pixels of the landmarks
            landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
            landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
            if draw:
                cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255, 0, 0), cv2.FILLED)
    return landmark_list

def get_angle(img, landmark_list, point1, point2, point3, draw=True):
    # Retrieve landmark coordinates from point identifiers
    x1, y1 = landmark_list[point1][1:]
    x2, y2 = landmark_list[point2][1:]
    x3, y3 = landmark_list[point3][1:]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Handling angle edge cases: Obtuse and negative angles
    if angle < 0:
        angle += 360
        if angle > 180:
            angle = 360 - angle
    elif angle > 180:
        angle = 360 - angle

    if draw:
        # Drawing lines between the three points
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

        # Drawing circles at intersection points of lines
        cv2.circle(img, (x1, y1), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (75, 0, 130), 2)
        cv2.circle(img, (x2, y2), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (75, 0, 130), 2)
        cv2.circle(img, (x3, y3), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, (75, 0, 130), 2)

        # Show angles between lines
        cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return angle

def convert_mediapipe_keypoints_for_model(lm_dict, landmark_list):
    inp_pushup = []
    for index in range(0, 36):
        if index < 18:
            inp_pushup.append(round(landmark_list[lm_dict[index]][1], 3))
        else:
            inp_pushup.append(round(landmark_list[lm_dict[index - 18]][2], 3))
    return inp_pushup

# Setting variables for video feed
def set_video_feed_variables():
    cap = cv2.VideoCapture(0)
    count = 0
    direction = 0
    form = 0
    feedback = "Bad Form."
    frame_queue = deque(maxlen=250)
    clf = FormFixClassifier(r"D:\FormFix\formfix1.tflite")
    return cap, count, direction, form, feedback, frame_queue, clf

def set_percentage_bar_and_text(elbow_angle):
    pushup_success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
    pushup_progress_bar = np.interp(elbow_angle, (90, 160), (380, 30))
    return pushup_success_percentage, pushup_progress_bar

def set_body_angles_from_keypoints(get_angle, img, landmark_list):
    elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
    shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
    hip_angle = get_angle(img, landmark_list, 11, 23, 25)
    elbow_angle_right = get_angle(img, landmark_list, 12, 14, 16)
    shoulder_angle_right = get_angle(img, landmark_list, 14, 12, 24)
    hip_angle_right = get_angle(img, landmark_list, 12, 24, 26)
    return elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right

def set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list):
    inp_pushup = convert_mediapipe_keypoints_for_model(lm_dict, landmark_list)
    workout_name = clf.predict(inp_pushup)

    # Only allow "pushups"
    if workout_name != "pushups":
        workout_name = "pushups"  # Default to pushups if the classifier returns something else

    frame_queue.append(workout_name)

    # Ensure fair weighting of both workouts in the queue
    workout_counts = {w: frame_queue.count(w) for w in ["pushups"]}
    workout_name_after_smoothening = max(workout_counts, key=workout_counts.get)

    return workout_name_after_smoothening

def run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, pushup_success_percentage, feedback):
    if form == 1:
        if pushup_success_percentage == 0:
            if elbow_angle <= 90 and hip_angle > 160 and elbow_angle_right <= 90 and hip_angle_right > 160:
                feedback = "Feedback: Go Up"
                if direction == 0:
                    count += 0.5
                    direction = 1
            else:
                # Incorrect posture feedback
                if hip_angle <= 160:
                    feedback = "Wrong posture, keep your body straight."
                else:
                    feedback = "Feedback: Bad Form."

        if pushup_success_percentage == 100:
            if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
                feedback = "Feedback: Go Down"
                if direction == 1:
                    count += 0.5
                    direction = 0
            else:
                # Incorrect posture feedback
                if hip_angle <= 160:
                    feedback = "Wrong posture, keep your body straight."
                else:
                    feedback = "Feedback: Bad Form."
    return [feedback, count]

def draw_percentage_progress_bar(form, img, pushup_success_percentage, pushup_progress_bar):
    xd, yd, wd, hd = 10, 175, 50, 200
    if form == 1:
        cv2.rectangle(img, (xd, 30), (xd + wd, yd + hd), (0, 255, 0), 3)
        cv2.rectangle(img, (xd, int(pushup_progress_bar)), (xd + wd, yd + hd), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(pushup_success_percentage)}%', (xd, yd + hd + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

def display_rep_count(count, img):
    xc, yc = 85, 100
    cv2.putText(img, "Reps: " + str(int(count)), (xc, yc), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

def show_workout_feedback(feedback, img):
    xf, yf = 85, 70
    cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

def show_workout_name_from_model(img, workout_name_after_smoothening):
    xw, yw = 85, 40
    cv2.putText(img, workout_name_after_smoothening, (xw, yw), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

def check_form(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, form):
    if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
        form = 1
    return form

def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar):
    # Draw the pushup progress bar
    draw_percentage_progress_bar(form, img, pushup_success_percentage, pushup_progress_bar)

    # Show the rep count
    display_rep_count(count, img)

    # Show the pushup feedback
    show_workout_feedback(feedback, img)

    # Show workout name
    show_workout_name_from_model(img, "pushups")  # Always show "pushups" as the workout name

def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon)

    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue, clf = set_video_feed_variables()

    # Start video feed and run workout
    while cap.isOpened():
        # Getting image from camera
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Getting video dimensions
        width = cap.get(3)
        height = cap.get(4)

        # Convert from BGR (used by cv2) to RGB (used by Mediapipe)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Get pose and draw landmarks
        img = get_pose(img, results, False)

        # Get landmark list from mediapipe
        landmark_list = get_position(img, results, height, width, False)

        # If landmarks exist, get the relevant workout body angles and run workout
        if len(landmark_list) != 0:
            elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right = set_body_angles_from_keypoints(get_angle, img, landmark_list)

            workout_name_after_smoothening = set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list)
            workout_name_after_smoothening = workout_name_after_smoothening.replace("Workout Name:", "").strip()

            pushup_success_percentage, pushup_progress_bar = set_percentage_bar_and_text(elbow_angle)

            # Is the form correct at the start?
            form = check_form(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, form)

            # Full workout motion
            feedback, count = run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, pushup_success_percentage, feedback)

            # Display workout stats
            display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar)

        # Transparent Overlay
        overlay = img.copy()
        x, y, w, h = 75, 10, 500, 150
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        alpha = 0.8