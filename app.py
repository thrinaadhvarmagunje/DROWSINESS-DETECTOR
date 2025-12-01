from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from drowsiness_detector import process_frame

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)

    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    status, conf = process_frame(frame)

    return jsonify({
        "status": status,
        "confidence": conf
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
