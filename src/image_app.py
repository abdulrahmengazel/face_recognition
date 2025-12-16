import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import cv2
import face_recognition
from src.database import Database
import src.config as config
from src.face_detector import detect_faces
from deepface import DeepFace

# --- DATABASE & HELPER FUNCTIONS ---

def add_new_face():
    messagebox.showinfo("Info", "Please use the 'Run Bulk Training' button in the main menu for adding new faces.")

def find_nearest_face_in_db(encoding_to_check):
    """Searches the database for the closest matching face using pgvector."""
    # FIX: Use .tolist() for numpy arrays
    vec_str = str(encoding_to_check.tolist()) if hasattr(encoding_to_check, 'tolist') else str(encoding_to_check)
    column_name = "encoding" if config.ENCODING_MODEL == "dlib" else f"encoding_{config.ENCODING_MODEL}"
    
    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            try:
                query = f"""
                    SELECT p.name, f.{column_name} <-> %s AS distance
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
        image = face_recognition.load_image_file(file_path)
        locations = detect_faces(image, config.FACE_DETECTION_MODEL, confidence=config.YOLO_CONFIDENCE, yolo_weights=config.YOLO_WEIGHTS)
        
        encodings = []
        if config.ENCODING_MODEL == "dlib":
            encodings = face_recognition.face_encodings(image, locations)
        else: # facenet
            for (top, right, bottom, left) in locations:
                face_img = image[top:bottom, left:right]
                if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    encodings.append(None)
                    continue
                    
                try:
                    embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                    encodings.append(embedding_objs[0]['embedding'])
                except:
                    encodings.append(None)
        
        image_to_draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not locations:
            messagebox.showinfo("Result", "No faces found in the image.")
            return

        for i, (top, right, bottom, left) in enumerate(locations):
            encoding = encodings[i]
            
            name = "UNKNOWN"
            color = (0, 0, 255)
            
            if encoding is not None:
                db_name, distance = find_nearest_face_in_db(encoding)
                if db_name and distance < config.RECOGNITION_THRESHOLD:
                    name = db_name.upper()
                    color = (0, 255, 0)

            cv2.rectangle(image_to_draw, (left, top), (right, bottom), color, 2)
            cv2.putText(image_to_draw, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Recognition Result", image_to_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")

def run_image_app(parent_root):
    """Creates the Image Recognition window as a child of the main app."""
    window = tk.Toplevel(parent_root)
    window.title(f"Image Recognition (Encoding: {config.ENCODING_MODEL})")
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
