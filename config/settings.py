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
# Recommended:
# - dlib (L2 Distance): ~0.6
# - facenet (Cosine Distance): ~0.4
RECOGNITION_THRESHOLD = 0.4

# --- YOLO Specific Config ---
# Define the directory where models are stored
YOLO_DIR = "assets/yolo"

# List of available YOLO models with their relative paths
# CORRECTED: Use os.path.join to build paths correctly
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
DB_CONFIG = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASS
}
# --- Encoding Model ---
MODEL_NAME = "dlib_face_recognition"
