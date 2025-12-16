import face_recognition
from ultralytics import YOLO
import cv2
import os

# --- MODEL LOADING ---
# Keep track of the loaded model and its weights file
yolo_model = None
loaded_weights = None

def load_yolo_model(weights_path):
    """Loads or reloads the YOLO model if the weights file has changed."""
    global yolo_model, loaded_weights
    
    # If the requested model is not the one currently loaded, reload it
    if yolo_model is None or loaded_weights != weights_path:
        try:
            if os.path.exists(weights_path):
                yolo_model = YOLO(weights_path)
                loaded_weights = weights_path
                print(f"YOLO model '{weights_path}' loaded successfully.")
            else:
                print(f"Error: YOLO weights file not found at '{weights_path}'.")
                yolo_model = None
                loaded_weights = None
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            yolo_model = None
            loaded_weights = None

# --- DETECTION FUNCTION ---

def detect_faces(frame, model_name, confidence=0.5, yolo_weights='yolov8n.pt'):
    """
    Detects faces in an image using the specified model.
    Returns a list of face locations in (top, right, bottom, left) format.
    """
    if model_name == "hog":
        return face_recognition.face_locations(frame, model="hog")
    
    elif model_name == "cnn":
        return face_recognition.face_locations(frame, model="cnn", number_of_times_to_upsample=0)
    
    elif model_name == "yolo":
        # Ensure the correct model is loaded
        load_yolo_model(yolo_weights)
        
        if yolo_model is None:
            return [] # Return empty if model failed to load

        # Perform detection
        results = yolo_model(frame, stream=True, verbose=False)
        
        face_locations = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Check confidence
                if box.conf[0] > confidence:
                    # YOLO returns xyxy format (left, top, right, bottom)
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Convert to face_recognition format (top, right, bottom, left)
                    face_locations.append((y1, x2, y2, x1))
        
        return face_locations
    
    else:
        print(f"Warning: Unknown model '{model_name}'. Defaulting to HOG.")
        return face_recognition.face_locations(frame, model="hog")
