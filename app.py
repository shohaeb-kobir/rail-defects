import cv2
import os
import glob
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==========================================
# --- 1. DYNAMIC CONFIGURATION ---
# ==========================================
# Paths inside the container
VIDEO_PATH = os.getenv("VIDEO_PATH", "/app/data/input/video.mp4")
MODEL_PATH = "/app/model/best.pt"
OUTPUT_DIR = "/app/data/output"
TEMP_FRAMES = "/tmp/processed_frames"

# Settings
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "1"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.4"))

# Your Specific Region of Interest (ROI)
ROI_POLYGON = np.array([
    [198, 290], [458, 282], [473, 353], [172, 353]
], np.int32)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_FRAMES, exist_ok=True)

# ==========================================
# --- 2. LOGIC ---
# ==========================================

def apply_roi_mask(frame, polygon):
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)

def process_pipeline():
    print(f"🚀 Starting Pipeline | Input: {VIDEO_PATH}")
    
    # 1. Extract & Mask
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {VIDEO_PATH}")
        return

    saved_count = 0
    while True:
        success, frame = cap.read()
        if not success: break
        
        if saved_count % FRAME_STRIDE == 0:
            masked = apply_roi_mask(frame, ROI_POLYGON)
            cv2.imwrite(f"{TEMP_FRAMES}/frame_{saved_count:05d}.jpg", masked)
        saved_count += 1
    cap.release()
    print(f"✅ Extracted {saved_count} frames.")

    # 2. Sliced Inference
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detection_model = AutoDetectionModel.from_model_type(
        model_type='yolov8', model_path=MODEL_PATH, 
        confidence_threshold=CONF_THRESHOLD, device=device
    )

    image_files = sorted(glob.glob(f"{TEMP_FRAMES}/*.jpg"))
    for img_path in image_files:
        result = get_sliced_prediction(
            img_path, detection_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        result.export_visuals(export_dir=OUTPUT_DIR, file_name=os.path.basename(img_path).replace(".jpg", ""))
    print(f"✅ Processing Complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_pipeline()