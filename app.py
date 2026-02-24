import cv2
import time
import os
import torch
import subprocess
import numpy as np
from flask import Flask, Response, request, render_template_string, jsonify, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- ROI DEFINITION ---
# The trapezoid for your rail track
ROI_POLYGON = np.array([[198, 290], [458, 282], [473, 353], [172, 353]], np.int32)

# Load Model (Priority: TensorRT > PyTorch)
MODEL_PATH = 'model/best.engine' if os.path.exists('model/best.engine') else 'model/best.pt'
print(f"Loading Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

class AppState:
    def __init__(self):
        self.streams = []
        self.conf_threshold = 0.5
        self.enable_detection = True
        self.show_roi = True # Highlights the inspection area

state = AppState()

# Helper: Masking Function
def apply_roi_mask(frame, polygon):
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)

@app.route('/gpu_status')
def gpu_status():
    try:
        cmd = "nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
        res = subprocess.check_output(cmd.split()).decode('utf-8').strip().split(', ')
        return jsonify({"name": res[0], "temp": res[1], "used": res[2], "total": res[3], "util": res[4]})
    except:
        return jsonify({"status": "offline"})

def generate_frames(stream_id):
    if stream_id >= len(state.streams): return
    source = state.streams[stream_id]
    cap = cv2.VideoCapture(source)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Apply ROI Masking
        processed_frame = apply_roi_mask(frame, ROI_POLYGON) if state.show_roi else frame.copy()

        if state.enable_detection:
            # Sliced-like inference can be simulated by focus on ROI
            results = model.predict(processed_frame, conf=state.conf_threshold, verbose=False)
            display_frame = results[0].plot()
        else:
            display_frame = processed_frame

        # Draw ROI Boundary for the user
        cv2.polylines(display_frame, [ROI_POLYGON], True, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# ... (Include standard routes from previous code: index, upload, clear_streams, update_settings) ...
# Ensure the HTML_TEMPLATE includes the GPU bar provided in the previous turn.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
