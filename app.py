import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from drowsiness_detector import process_frame

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"status": "error", "ear": 0.0}), 400

    # Decode base64 image
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)

    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    status, ear = process_frame(frame)

    return jsonify({"status": status, "ear": ear})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
