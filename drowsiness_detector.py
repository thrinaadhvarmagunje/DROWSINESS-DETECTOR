import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from scipy.spatial import distance as dist

EAR_THRESHOLD = 0.20
CONSEC_FRAMES = 5
COUNTER = 0

model = YOLO("yolov8n.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def process_frame(frame):
    global COUNTER

    results = model(frame)[0]

    status = "Awake"
    ear_value = 0

    for result in results.boxes:
        cls = int(result.cls[0])
        if cls == 0:  # CLASS 0 = FACE
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            face_roi = frame[y1:y2, x1:x2]

            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(face_rgb)

            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:

                    left_idx = [33, 160, 158, 133, 153, 144]
                    right_idx = [362, 385, 387, 263, 373, 380]

                    left_eye = [(int(face_landmarks.landmark[i].x * (x2 - x1)),
                                 int(face_landmarks.landmark[i].y * (y2 - y1))) for i in left_idx]

                    right_eye = [(int(face_landmarks.landmark[i].x * (x2 - x1)),
                                  int(face_landmarks.landmark[i].y * (y2 - y1))) for i in right_idx]

                    leftEAR = eye_aspect_ratio(left_eye)
                    rightEAR = eye_aspect_ratio(right_eye)
                    ear = (leftEAR + rightEAR) / 2.0
                    ear_value = ear

                    if ear < EAR_THRESHOLD:
                        COUNTER += 1
                        if COUNTER >= CONSEC_FRAMES:
                            status = "Drowsy"
                    else:
                        COUNTER = 0
                        status = "Awake"

    return status, ear_value
