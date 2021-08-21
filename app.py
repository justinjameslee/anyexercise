from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
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

@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')
  
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    #gets difference from end to mid, mid to first
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    return angle

def gen():
    # Get webcam video
    cap = cv2.VideoCapture(0)

    #Set dimensions
    cap.set(3, 1280)
    cap.set(4, 720)
    # Curl counter variables
    counter = 0 
    stage = None
    displayTimer = None

    ## Setup mediapipe instance
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
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                #lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                #rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


                # Calculate angles for
                lhipangle = calculate_angle(lknee, lhip, lshoulder)
                rshoulderangle = calculate_angle(rhip, rshoulder, relbow)
                
                # Side bend counter logic
                if rshoulderangle < 130 or rshoulderangle > 200: #to make sure right arm is raised
                    stage = "raise right arm"
                    displayTimer = None
                    held = "no"
                elif lhipangle > 175: #r arm raised and raedy to side bend
                    stage = "bend"
                    displayTimer = None
                    held = "no"
                if lhipangle < 165 and stage =='bend': #correct side bend angle acheived
                    start = time.time()  #start timer
                    stage = "hold"
                    displayTimer = "timer"
                    held = "no"
                    while (time.time()-start) < 2:
                        held = 'yes'
                        if lhipangle > 170:   #person straightens          
                            held = 'no'
                    if held == 'yes' and lhipangle < 165: #checks if held for more than 1 second
                        stage="straighten"
                        if lhipangle > 170:
                            counter +=1
                            print(counter)
                            print(lhipangle)
                        
            except:
                pass
            
            # Render counter
            # Setup status box
            cv2.rectangle(image, (0,0), (625,73), (255, 255, 255), -1)
            #timer box
            cv2.rectangle(image, (800,0), (1225,73), (255, 255, 255), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

            #timer
            cv2.putText(image, displayTimer, 
                        (800,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=2) 
                                    )               
            
            ret,jpg=cv2.imencode('.jpg',image)
            yield(b'--frame\r\n'b'Content-Type:  image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n')

app.run(debug='true')