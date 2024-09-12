from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('styrofoam_classifier.h5')

# Set up the camera
cap = cv2.VideoCapture(0)
IMG_HEIGHT = 150
IMG_WIDTH = 150

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocess the frame
        img = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Predict
        prediction = model.predict(img)
        prob = prediction[0][0]

        # Label the frame
        if prob > 0.5:
            label = f'Styrofoam: {prob:.2f}'
            color = (0, 255, 0)  # Green
        else:
            label = f'Not Styrofoam: {1 - prob:.2f}'
            color = (0, 0, 255)  # Red

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
