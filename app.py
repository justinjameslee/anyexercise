from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app=Flask(__name__)
app.static_folder = 'static'

@app.route('/pose.html')
def pose():
    return render_template('pose.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    # Get webcam video
    cap = cv2.VideoCapture(0)

    #Set dimensions
    cap.set(3, 1920)
    cap.set(4, 1080)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )     

            ret,jpg=cv2.imencode('.jpg',image)
            yield(b'--frame\r\n'b'Content-Type:  image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n')
        
@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)