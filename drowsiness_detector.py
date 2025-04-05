import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from scipy.spatial import distance as dist
from playsound import playsound
import threading
import time

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def sound_alarm():
    playsound("static/alarm.wav")

def detect_and_stream():
    global COUNTER, ALARM_ON
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n.pt')
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for result in results.boxes:
            cls = int(result.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                face_roi = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(face_rgb)

                if mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        left_eye_idx = [33, 160, 158, 133, 153, 144]
                        right_eye_idx = [362, 385, 387, 263, 373, 380]

                        left_eye = [(int(face_landmarks.landmark[i].x * (x2 - x1)),
                                     int(face_landmarks.landmark[i].y * (y2 - y1))) for i in left_eye_idx]
                        right_eye = [(int(face_landmarks.landmark[i].x * (x2 - x1)),
                                      int(face_landmarks.landmark[i].y * (y2 - y1))) for i in right_eye_idx]

                        for pt in left_eye + right_eye:
                            cv2.circle(face_roi, pt, 2, (0, 255, 0), -1)

                        leftEAR = eye_aspect_ratio(left_eye)
                        rightEAR = eye_aspect_ratio(right_eye)
                        ear = (leftEAR + rightEAR) / 2.0

                        if ear < EAR_THRESHOLD:
                            COUNTER += 1
                            if COUNTER >= CONSEC_FRAMES:
                                if not ALARM_ON:
                                    ALARM_ON = True
                                    threading.Thread(target=sound_alarm, daemon=True).start()
                                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            COUNTER = 0
                            ALARM_ON = False

                        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
