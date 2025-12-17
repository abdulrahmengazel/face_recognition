# Shared Configuration
import os

# --- DYNAMIC PATH SETUP ---
# Get the absolute path of the project's root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Encoding Model ---
ENCODING_MODEL = "facenet"

# --- Detection Model ---
FACE_DETECTION_MODEL = "yolo"

# --- Recognition Threshold ---
RECOGNITION_THRESHOLD = 0.4

# --- YOLO Specific Config ---
# Use absolute path to avoid any ambiguity
YOLO_DIR = os.path.join(PROJECT_ROOT, "assets", "yolo")

# List of available YOLO models with their absolute paths
YOLO_MODELS = {
    "YOLOv8 Nano": os.path.join(YOLO_DIR, "yolov8n-face.pt"),
    "YOLOv8 Medium": os.path.join(YOLO_DIR, "yolov8m-face.pt"),
    "YOLOv8 Large": os.path.join(YOLO_DIR, "yolov8l-face.pt"),
}

# The currently selected YOLO model file (Default)
YOLO_WEIGHTS = YOLO_MODELS["YOLOv8 Medium"]
YOLO_CONFIDENCE = 0.5

# --- Performance Tuning ---
PROCESSING_SCALE = 1.0
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
