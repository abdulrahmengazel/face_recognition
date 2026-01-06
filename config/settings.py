# Shared Configuration
import os

# --- DYNAMIC PATH SETUP ---
# Get the absolute path of the project's root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- UI THEME COLORS ---
UI_COLORS = {
    "bg": "#1d2d44",  # Dark Navy (Main Background)
    "frame": "#3e5c76",  # Slate Blue (Card Background)
    "button": "#748cab",  # Muted Blue (Primary Action)
    "hover": "#5b7fa6",  # Slightly Darker/Richer Blue for Hover (Better Contrast with Cream Text)
    "text": "#f0ebd8",  # Cream (Text Color)
    "accent": "#748cab",  # Accent Color
    "danger": "#8B0000",  # Red for danger/exit
    "danger_hover": "#A52A2A"
}

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
YOLO_WEIGHTS = YOLO_MODELS["YOLOv8 Large"]
YOLO_CONFIDENCE = 0.5

# --- Performance Tuning ---
PROCESSING_SCALE = 1.0
TRAINING_IMAGE_SIZE = (800, 800)

# --- Classifier Model Path ---
CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "assets", "classifier.pkl")

# --- Training Configuration (For Fine-tuning Models) ---
TRAINING_CONFIG = {
    "yolo": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.01
    },
    "facenet": {
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "dlib": {
        "epochs": 100,
        "jitter": 1  # Reset to 1 for speed (since we store all encodings)
    },
    "detection": {
        "hog_upsample": 1,  # Reset to 1 (Safe)
        "cnn_upsample": 0  # Reset to 0 (Safe)
    },
    "classifier": {
        "type": "mlp",  # Options: "mlp", "xgboost"
        # MLP Settings
        "hidden_layers": (1024, 512, 256),
        "max_iter": 1000,
        "solver": "adam",
        "learning_rate_init": 0.001,
        "alpha": 0.0001,
        "n_iter_no_change": 20,
        # XGBoost Settings
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8
    }
}

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
