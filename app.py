from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
import string
from tensorflow import keras

app = Flask(__name__, template_folder='.')

model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# Utilities
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
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    return [x / max_value if max_value != 0 else 0 for x in temp_landmark_list]

latest_prediction = "None"  # Global to store latest label

def generate_frames():
    global latest_prediction
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            label = "None"
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

            latest_prediction = label

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    return redirect(url_for('camera'))

@app.route('/camera')
def camera():
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live ISL Detection</title>
        <style>
            body {{
                text-align: center;
                font-family: Arial, sans-serif;
                background-color: #f0f2f5;
                padding-top: 30px;
            }}
            h2 {{
                color: #333;
            }}
            img {{
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
            }}
            .prediction {{
                font-size: 24px;
                margin-top: 20px;
                font-weight: bold;
                color: #0066cc;
            }}
            a {{
                display: inline-block;
                margin-top: 30px;
                text-decoration: none;
                color: #fff;
                background-color: #007BFF;
                padding: 10px 20px;
                border-radius: 8px;
                transition: 0.3s;
            }}
            a:hover {{
                background-color: #0056b3;
            }}
        </style>
        <script>
            async function fetchPrediction() {{
                const response = await fetch('/prediction');
                const data = await response.text();
                document.getElementById('predictionText').innerText = "Prediction: " + data;
            }}
            setInterval(fetchPrediction, 1000);
        </script>
    </head>
    <body>
        <h2>Live ISL Detection</h2>
        <img src="/video_feed" width="700">
        <div class="prediction" id="predictionText">Prediction: None</div>
        <a href="/">⬅ Back</a>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return latest_prediction

if __name__ == "__main__":
    app.run(debug=True)
