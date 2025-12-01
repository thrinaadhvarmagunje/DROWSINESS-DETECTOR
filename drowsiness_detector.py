from ultralytics import YOLO
import cv2

# Load the YOLO Drowsiness Model
model = YOLO("drowsiness_yolov8n.pt")

def process_frame(frame):
    """
    Processes a frame and returns:
    - status: "Awake" or "Drowsy"
    - confidence: float between 0–1
    """
    results = model(frame)[0]

    status = "Awake"
    conf_score = 0.0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Class 0 = Drowsy / Closed Eyes
        # Class 1 = Awake / Open Eyes

        if cls == 0:  # drowsy
            status = "Drowsy"
            conf_score = conf
        else:
            status = "Awake"
            conf_score = conf

    return status, round(conf_score, 2)
