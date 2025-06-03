import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
import string
from tensorflow import keras
import streamlit as st

# Load your trained model
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

st.title("Indian Sign Language Detection with Webcam")

# Initialize session state for camera toggle
if 'start_camera' not in st.session_state:
    st.session_state.start_camera = False

# Button to start camera
if not st.session_state.start_camera:
    if st.button("Start Camera"):
        st.session_state.start_camera = True

# When camera started
if st.session_state.start_camera:
    frame_placeholder = st.empty()
    label_placeholder = st.empty()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            # Run the webcam feed until the app is stopped manually
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)  # mirror image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(frame, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)

                        df = pd.DataFrame(pre_processed_landmark_list).transpose()
                        predictions = model.predict(df, verbose=0)
                        predicted_classes = np.argmax(predictions, axis=1)
                        label = alphabet[predicted_classes[0]]

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        cv2.putText(frame, label, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 2)
                        label_placeholder.markdown(f"### Prediction: {label}")
                else:
                    label_placeholder.markdown("### Prediction: None")

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

