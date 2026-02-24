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

# --- 1. CONFIGURATION ---
# Your Specific Rail ROI (Polygon)
ROI_POLYGON = np.array([[198, 290], [458, 282], [473, 353], [172, 353]], np.int32)

# Model Priority: TensorRT (.engine) > PyTorch (.pt)
MODEL_PATH = 'model/best.engine' if os.path.exists('model/best.engine') else 'model/best.pt'
print(f"🚀 Initializing model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

class AppState:
    def __init__(self):
        self.streams = []
        self.conf_threshold = 0.5
        self.enable_detection = True
        self.show_roi_overlay = True

state = AppState()

# --- 2. GPU MONITORING ---
@app.route('/gpu_status')
def gpu_status():
    try:
        cmd = "nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
        res = subprocess.check_output(cmd.split()).decode('utf-8').strip().split(', ')
        return jsonify({"name": res[0], "temp": res[1], "used": res[2], "total": res[3], "util": res[4]})
    except:
        return jsonify({"status": "offline"})

# --- 3. AI INFERENCE ENGINE ---
def apply_roi_mask(frame, polygon):
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)

def generate_frames(stream_id):
    if stream_id >= len(state.streams): return
    cap = cv2.VideoCapture(state.streams[stream_id])
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # AI Focus on ROI
        input_frame = apply_roi_mask(frame, ROI_POLYGON)
        
        if state.enable_detection:
            results = model.predict(input_frame, conf=state.conf_threshold, verbose=False, half=True)
            output_frame = results[0].plot()
        else:
            output_frame = frame.copy()

        # Draw ROI Boundary on UI
        if state.show_roi_overlay:
            cv2.polylines(output_frame, [ROI_POLYGON], True, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', output_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, streams=state.streams)

@app.route('/video_feed/<int:sid>')
def video_feed(sid):
    return Response(generate_frames(sid), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file:
        path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(path)
        state.streams.append(path)
    return redirect('/')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    state.conf_threshold = float(data['conf'])
    state.enable_detection = bool(data['enable'])
    return jsonify({"status": "ok"})

@app.route('/clear')
def clear():
    state.streams = []
    return redirect('/')

# --- 5. HTML UI (Simplified with GPU Bar) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <title>AI Rail Command</title>
</head>
<body class="p-4">
    <div class="row mb-4">
        <div class="col-3"><div class="card p-2 text-center">GPU: <span id="g-name" class="text-info">--</span></div></div>
        <div class="col-3"><div class="card p-2 text-center">Temp: <span id="g-temp" class="text-danger">--</span></div></div>
        <div class="col-3"><div class="card p-2 text-center">VRAM: <span id="g-mem" class="text-primary">--</span></div></div>
        <div class="col-3"><div class="card p-2 text-center">Load: <span id="g-util" class="text-warning">--</span></div></div>
    </div>

    <div class="row g-3">
        {% for s in streams %}
        <div class="col-md-6"><img src="/video_feed/{{ loop.index0 }}" class="img-fluid rounded border border-secondary"></div>
        {% endfor %}
    </div>

    <div class="card mt-4 p-3">
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" class="form-control mb-2">
            <button type="submit" class="btn btn-primary">Add Stream</button>
            <a href="/clear" class="btn btn-danger">Clear All</a>
        </form>
    </div>

    <script>
        setInterval(() => {
            fetch('/gpu_status').then(r => r.json()).then(d => {
                document.getElementById('g-name').innerText = d.name;
                document.getElementById('g-temp').innerText = d.temp + "°C";
                document.getElementById('g-mem').innerText = d.used + "/" + d.total + " MB";
                document.getElementById('g-util').innerText = d.util + "%";
            });
        }, 2000);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
