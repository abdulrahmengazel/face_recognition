import face_recognition
from ultralytics import YOLO
import cv2
import os
import torch

# --- MODEL LOADING ---
# Keep track of the loaded model and its weights file
yolo_model = None
loaded_weights = None

def load_yolo_model(weights_path):
    """Loads or reloads the YOLO model if the weights file has changed."""
    global yolo_model, loaded_weights
    
    # Pre-check if the weights file exists
    if not os.path.exists(weights_path):
        print(f"Error: YOLO weights file not found at '{weights_path}'.")
        if yolo_model is not None:
            yolo_model = None
            loaded_weights = None
        return

    if yolo_model is None or loaded_weights != weights_path:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            yolo_model = YOLO(weights_path)
            yolo_model.to(device) 
            loaded_weights = weights_path
            print(f"YOLO model '{weights_path}' loaded successfully on device: {device.upper()}.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            yolo_model = None
            loaded_weights = None

# --- STANDARD DETECTION FUNCTION (Returns Locations Only) ---

def detect_faces(frame, model_name, confidence=0.5, yolo_weights='yolov8n.pt'):
    """
    Returns a list of face locations in (top, right, bottom, left) format.
    """
    results_with_score = detect_faces_with_score(frame, model_name, confidence, yolo_weights)
    # Extract just the locations
    return [loc for loc, score in results_with_score]

# --- NEW FUNCTION: DETECTION WITH SCORE ---

def detect_faces_with_score(frame, model_name, confidence=0.5, yolo_weights='yolov8n.pt'):
    """
    Detects faces and returns a list of tuples: ((top, right, bottom, left), score)
    Score is between 0.0 and 1.0.
    """
    if model_name == "hog":
        # HOG in face_recognition doesn't return a probability score easily. 
        # If it finds a face, we assume high confidence.
        locs = face_recognition.face_locations(frame, model="hog")
        return [(loc, 0) for loc in locs]
    
    elif model_name == "cnn":
        # CNN also returns locations.
        locs = face_recognition.face_locations(frame, model="cnn")
        return [(loc, 0) for loc in locs]
    
    elif model_name == "yolo":
        load_yolo_model(yolo_weights)
        
        if yolo_model is None:
            return []

        # Perform detection
        results = yolo_model(frame, stream=True, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes.cpu() 
            for box in boxes:
                conf = box.conf[0].item()
                
                if conf > confidence:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    # Convert to (top, right, bottom, left)
                    loc = (y1, x2, y2, x1)
                    detections.append((loc, conf))
        
        return detections
    
    else:
        print(f"Warning: Unknown model '{model_name}'. Defaulting to HOG.")
        locs = face_recognition.face_locations(frame, model="hog")
        return [(loc, 1.0) for loc in locs]
