# Shared Configuration
import os

# --- Encoding Model ---
# Options: "dlib", "facenet"
ENCODING_MODEL = "facenet"

# --- Detection Model ---
# Options: "hog", "cnn", "yolo"
FACE_DETECTION_MODEL = "yolo"

# --- Recognition Threshold ---
# Distance threshold for matching faces.
# Lower = Stricter (Less False Positives, More False Negatives)
# Higher = Looser (More False Positives, Less False Negatives)
# Recommended: dlib ~ 0.6, facenet ~ 0.8 to 1.0 (L2 distance)
RECOGNITION_THRESHOLD = 0.6

# --- YOLO Specific Config ---
# Define the directory where models are stored
YOLO_DIR = "yolo"

# List of available YOLO models with their relative paths
YOLO_MODELS = {
    "YOLOv8 Nano": os.path.join(YOLO_DIR, "yolov8n-face.pt"),
    "YOLOv8 Medium": os.path.join(YOLO_DIR, "yolov8m-face.pt"),
    "YOLOv8 Large": os.path.join(YOLO_DIR, "yolov8l-face.pt"),
}

# The currently selected YOLO model file (Default)
YOLO_WEIGHTS = YOLO_MODELS["YOLOv8 Medium"]
YOLO_CONFIDENCE = 0.5

# --- Performance Tuning ---
# Scale factor for video processing (0.1 to 1.0)
PROCESSING_SCALE = 1.0

# Image size for training
TRAINING_IMAGE_SIZE = (800, 800)

# --- Database Config ---
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "postgres_image_ymg3"

# --- Encoding Model ---
MODEL_NAME = "dlib_face_recognition"
