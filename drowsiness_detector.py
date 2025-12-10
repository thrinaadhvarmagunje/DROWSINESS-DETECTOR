import cv2
import numpy as np
import mediapipe as mp
from math import dist

# EAR config
EAR_THRESHOLD = 0.20          # eye aspect ratio below this => eye considered closed
FRAME_INTERVAL_SEC = 0.3      # must match JS setInterval in index.html
ALARM_AFTER_SEC = 5           # seconds of continuous low EAR
CONSEC_FRAMES = int(ALARM_AFTER_SEC / FRAME_INTERVAL_SEC)

counter = 0  # number of consecutive "closed eye" frames

# Mediapipe FaceMesh init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# indices for eyes in FaceMesh
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(eye_points):
    """eye_points: list[(x,y)] of 6 landmarks"""
    A = dist(eye_points[1], eye_points[5])
    B = dist(eye_points[2], eye_points[4])
    C = dist(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear


def process_frame(frame):
    """
    Input:  BGR frame (numpy array)
    Output: (status:str, ear:float)
    """
    global counter

    # Handle RGBA frames just in case
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        # No face found → reset
        counter = 0
        return "No face", 0.0

    face_landmarks = results.multi_face_landmarks[0]

    def landmarks_to_points(indices):
        pts = []
        for i in indices:
            lm = face_landmarks.landmark[i]
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
        return pts

    left_eye_pts = landmarks_to_points(LEFT_EYE_IDX)
    right_eye_pts = landmarks_to_points(RIGHT_EYE_IDX)

    left_ear = eye_aspect_ratio(left_eye_pts)
    right_ear = eye_aspect_ratio(right_eye_pts)
    ear = (left_ear + right_ear) / 2.0

    # EAR-based drowsiness logic
    if ear < EAR_THRESHOLD:
        counter += 1
    else:
        counter = 0

    if counter >= CONSEC_FRAMES:
        status = "Drowsy"
    else:
        status = "Awake"

    return status, float(round(ear, 3))
