import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import cv2
import face_recognition
# Updated import paths
from core.database import Database
import config.settings as settings
from core.detector import detect_faces
from deepface import DeepFace
import numpy as np

# --- DATABASE & HELPER FUNCTIONS ---

def add_new_face():
    messagebox.showinfo("Info", "Please use the 'Run Bulk Training' button in the main menu for adding new faces.")

def find_nearest_face_in_db(encoding_to_check):
    """Searches the database for the closest matching face using pgvector."""
    vec_str = str(encoding_to_check.tolist()) if hasattr(encoding_to_check, 'tolist') else str(encoding_to_check)
    
    if settings.ENCODING_MODEL == "dlib":
        column_name = "encoding"
        distance_operator = "<->"  # L2 distance for dlib
    else:
        column_name = f"encoding_{settings.ENCODING_MODEL}"
        distance_operator = "<=>"  # Cosine distance for FaceNet

    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            try:
                query = f"""
                    SELECT p.name, f.{column_name} {distance_operator} %s AS distance
                    FROM people p
                    JOIN face_encodings f ON p.id = f.person_id
                    WHERE f.{column_name} IS NOT NULL
                    ORDER BY distance ASC
                    LIMIT 1;
                """
                cursor.execute(query, (vec_str,))
                
                result = cursor.fetchone()
                return result if result else (None, None)
            except Exception as e:
                print(f"DB Search Error: {e}")
                return None, None

def select_and_recognize_image():
    """Handles image selection and recognition using pgvector."""
    file_path = filedialog.askopenfilename()
    if not file_path: return

    try:
        # 1. Load Original Image (Keep full resolution for processing)
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            messagebox.showerror("Error", "Could not read image file.")
            return

        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2. Perform Detection & Recognition on ORIGINAL image
        locations = detect_faces(image_rgb, settings.FACE_DETECTION_MODEL, confidence=settings.YOLO_CONFIDENCE, yolo_weights=settings.YOLO_WEIGHTS)
        
        encodings = []
        if settings.ENCODING_MODEL == "dlib":
            encodings = face_recognition.face_encodings(image_rgb, locations)
        else: # facenet
            for (top, right, bottom, left) in locations:
                face_img = image_rgb[top:bottom, left:right]
                if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    encodings.append(None)
                    continue
                    
                try:
                    embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                    encodings.append(embedding_objs[0]['embedding'])
                except:
                    encodings.append(None)
        
        # 3. Prepare Image for Display (Resize to 800x600)
        original_height, original_width = image_bgr.shape[:2]
        target_width = 800
        target_height = 600
        
        # Calculate scale ratios
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Resize the image for display
        image_display = cv2.resize(image_bgr, (target_width, target_height))

        if not locations:
            messagebox.showinfo("Result", "No faces found in the image.")
            cv2.imshow("Recognition Result", image_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        # 4. Draw Results on the RESIZED image
        for i, (top, right, bottom, left) in enumerate(locations):
            encoding = encodings[i]
            
            name = "UNKNOWN"
            color = (0, 0, 255)
            
            if encoding is not None:
                db_name, distance = find_nearest_face_in_db(encoding)
                if db_name and distance < settings.RECOGNITION_THRESHOLD:
                    name = f"{db_name.upper()} ({distance:.2f})"
                    color = (0, 255, 0)

            # Scale coordinates to match the resized image
            new_top = int(top * scale_y)
            new_bottom = int(bottom * scale_y)
            new_left = int(left * scale_x)
            new_right = int(right * scale_x)

            cv2.rectangle(image_display, (new_left, new_top), (new_right, new_bottom), color, 2)
            cv2.putText(image_display, name, (new_left + 6, new_bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Recognition Result", image_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")

def run_image_app(parent_root):
    """Creates the Image Recognition window as a child of the main app."""
    window = tk.Toplevel(parent_root)
    window.title(f"Image Recognition (Encoding: {settings.ENCODING_MODEL})")
    window.geometry("400x250")
    
    window.transient(parent_root)
    window.grab_set()

    main_frame = ttk.Frame(window, padding="20")
    main_frame.pack(expand=True, fill="both")

    add_face_btn = ttk.Button(main_frame, text="Add New Face", command=add_new_face)
    add_face_btn.pack(pady=10, fill="x")

    recognize_btn = ttk.Button(main_frame, text="Recognize from Image", command=select_and_recognize_image)
    recognize_btn.pack(pady=10, fill="x")

    quit_btn = ttk.Button(main_frame, text="Close", command=window.destroy)
    quit_btn.pack(pady=10, fill="x")
    
    parent_root.wait_window(window)

if __name__ == "__main__":
    import numpy as np 
    root = tk.Tk()
    Database.initialize_pool()
    run_image_app(root)
    Database.close_all()
    root.destroy()
