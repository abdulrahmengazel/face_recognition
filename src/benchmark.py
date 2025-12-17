import os
import cv2
import numpy as np
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from src.database import Database
import src.config as config
from src.face_detector import detect_faces
from deepface import DeepFace
import face_recognition

# --- Configuration ---
TEST_DIR = "../TestImages"

def get_encoding(image_rgb, model_name):
    """Extracts encoding based on the model name."""
    # Detect face first (using the configured detector, usually YOLO)
    # Note: We measure detection time separately in the main loop if needed, 
    # but here we include it in the total "Inference Time".
    
    start_time = time.perf_counter()
    
    locations = detect_faces(
        image_rgb, 
        config.FACE_DETECTION_MODEL, 
        confidence=config.YOLO_CONFIDENCE, 
        yolo_weights=config.YOLO_WEIGHTS
    )
    
    if not locations:
        return None, time.perf_counter() - start_time

    top, right, bottom, left = locations[0]
    encoding = None

    if model_name == "dlib":
        encs = face_recognition.face_encodings(image_rgb, [locations[0]])
        if encs: encoding = encs[0]
    
    elif model_name == "facenet":
        face_img = image_rgb[top:bottom, left:right]
        if face_img.size > 0:
            try:
                embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                if embedding_objs:
                    encoding = np.array(embedding_objs[0]['embedding'])
            except:
                pass
    
    end_time = time.perf_counter()
    return encoding, end_time - start_time

def find_nearest_face(encoding, cursor, model_name):
    """Finds the nearest face in DB for the specific model."""
    if encoding is None: return "UNKNOWN"
    
    vec_str = str(encoding.tolist()) if hasattr(encoding, 'tolist') else str(encoding)
    
    if model_name == "dlib":
        column_name = "encoding"
        distance_operator = "<->"
        threshold = 0.6 # Standard for dlib
    else:
        column_name = "encoding_facenet"
        distance_operator = "<=>"
        threshold = 0.4 # Standard for FaceNet (Cosine)

    try:
        query = f"""
            SELECT p.name, f.{column_name} {distance_operator} %s AS distance
            FROM people p JOIN face_encodings f ON p.id = f.person_id
            WHERE f.{column_name} IS NOT NULL
            ORDER BY distance ASC
            LIMIT 1;
        """
        cursor.execute(query, (vec_str,))
        result = cursor.fetchone()
        
        if result:
            name, distance = result
            if distance < threshold:
                return name
        return "UNKNOWN"
    except:
        return "ERROR"

def run_benchmark_for_model(model_name, cursor):
    print(f"\n--- Benchmarking Model: {model_name.upper()} ---")
    
    y_true = []
    y_pred = []
    processing_times = []
    
    people_dirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    if not people_dirs:
        print("No test images found!")
        return None

    total_images = 0
    
    for true_name in people_dirs:
        person_path = os.path.join(TEST_DIR, true_name)
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in images:
            total_images += 1
            img_path = os.path.join(person_path, img_file)
            
            # Load Image
            image = cv2.imread(img_path)
            if image is None: continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract Encoding & Measure Time
            encoding, duration = get_encoding(rgb_image, model_name)
            processing_times.append(duration)
            
            # Recognize
            predicted_name = find_nearest_face(encoding, cursor, model_name)
            
            y_true.append(true_name)
            y_pred.append(predicted_name)
            
            # print(f"  > {img_file}: {predicted_name} ({duration:.4f}s)")

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    avg_time = np.mean(processing_times)
    
    print(f"Results for {model_name}:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Avg Time: {avg_time:.4f} sec/image")
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Avg Time (s)": avg_time
    }

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Error: '{TEST_DIR}' directory not found.")
        return

    Database.initialize_pool()
    
    results = []
    
    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            # 1. Test Dlib
            res_dlib = run_benchmark_for_model("dlib", cursor)
            if res_dlib: results.append(res_dlib)
            
            # 2. Test FaceNet
            res_facenet = run_benchmark_for_model("facenet", cursor)
            if res_facenet: results.append(res_facenet)

    Database.close_all()
    
    if not results:
        return

    # --- Display Comparison ---
    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("FINAL COMPARISON REPORT")
    print("="*40)
    print(df.to_string(index=False))
    
    # --- Plotting ---
    try:
        # 1. Accuracy & F1 Score Comparison
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='Accuracy', data=df, palette='viridis')
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1.1)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='Avg Time (s)', data=df, palette='rocket')
        plt.title('Inference Speed (Lower is Better)')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\nCould not generate plots: {e}")

if __name__ == "__main__":
    main()
