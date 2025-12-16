import cv2
import face_recognition
import os
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from src.database import Database
import src.config as config
from src.face_detector import detect_faces
from deepface import DeepFace

def resize_image(image, target_size):
    height, width = image.shape[:2]
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height))

def get_encodings_for_image(image_rgb):
    """
    Gets both dlib and facenet encodings for a single image.
    """
    dlib_enc = None
    facenet_enc = None

    # Use the configured detector
    locs = detect_faces(image_rgb, config.FACE_DETECTION_MODEL, confidence=config.YOLO_CONFIDENCE, yolo_weights=config.YOLO_WEIGHTS)
    
    if not locs:
        return None, None

    top, right, bottom, left = locs[0]

    # 1. Get Dlib encoding
    dlib_encs = face_recognition.face_encodings(image_rgb, [locs[0]])
    if dlib_encs:
        dlib_enc = dlib_encs[0]
        
    # 2. Get FaceNet encoding
    face_img = image_rgb[top:bottom, left:right]
    if face_img.shape[0] > 20 and face_img.shape[1] > 20:
        try:
            embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
            if embedding_objs:
                facenet_enc = np.array(embedding_objs[0]['embedding'])
        except:
            pass 
            
    return dlib_enc, facenet_enc

def train_model(training_dir="TrainingImages", progress_callback=None):
    """
    Scans images and saves encodings.
    progress_callback: function(current_index, total_count, message)
    """
    if not os.path.exists(training_dir):
        if progress_callback: progress_callback(0, 0, "Training directory not found!")
        return

    print(f"Starting Unified Training...")
    
    Database.init_tables()

    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            
            people = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]
            total_people = len(people)
            
            if total_people == 0:
                if progress_callback: progress_callback(0, 0, "No people found to train.")
                return

            saved_count = 0
            
            for i, person_name in enumerate(people):
                # Update Progress
                if progress_callback:
                    progress_callback(i, total_people, f"Processing: {person_name}...")

                person_path = os.path.join(training_dir, person_name)
                
                cursor.execute("SELECT id FROM people WHERE name = %s;", (person_name,))
                row = cursor.fetchone()
                person_id = row[0] if row else cursor.execute("INSERT INTO people (name) VALUES (%s) RETURNING id;", (person_name,)) or cursor.fetchone()[0]
                
                images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                dlib_encodings = []
                facenet_encodings = []

                for img_path in images:
                    try:
                        with open(img_path, 'rb') as f:
                            file_bytes = np.fromfile(f, dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        if img is None: continue
                        
                        img = resize_image(img, config.TRAINING_IMAGE_SIZE)
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        dlib_enc, facenet_enc = get_encodings_for_image(rgb)
                        
                        if dlib_enc is not None:
                            dlib_encodings.append(dlib_enc)
                        if facenet_enc is not None:
                            facenet_encodings.append(facenet_enc)

                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

                final_dlib_enc = np.mean(dlib_encodings, axis=0) if dlib_encodings else None
                final_facenet_enc = np.mean(facenet_encodings, axis=0) if facenet_encodings else None

                dlib_vec_str = str(final_dlib_enc.tolist()) if final_dlib_enc is not None else None
                facenet_vec_str = str(final_facenet_enc.tolist()) if final_facenet_enc is not None else None

                cursor.execute("SELECT id FROM face_encodings WHERE person_id = %s", (person_id,))
                existing_id = cursor.fetchone()

                if existing_id:
                    update_query = "UPDATE face_encodings SET encoding = %s::vector, encoding_facenet = %s::vector WHERE id = %s"
                    cursor.execute(update_query, (dlib_vec_str, facenet_vec_str, existing_id[0]))
                else:
                    insert_query = "INSERT INTO face_encodings (person_id, model_name, encoding, encoding_facenet) VALUES (%s, %s, %s::vector, %s::vector)"
                    cursor.execute(insert_query, (person_id, "multi-model", dlib_vec_str, facenet_vec_str))
                
                conn.commit()
                saved_count += 1
                
            # Final update
            if progress_callback:
                progress_callback(total_people, total_people, "Training Completed!")

    print(f"Training Finished. Processed {saved_count} people.")

# --- GUI WRAPPER ---

def run_training_gui(parent_root):
    """
    Opens a progress window and runs training in a separate thread.
    """
    window = tk.Toplevel(parent_root)
    window.title("Training Progress")
    window.geometry("400x150")
    window.transient(parent_root)
    window.grab_set()
    
    # Center the window
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

    # UI Elements
    lbl_status = ttk.Label(window, text="Initializing...", font=("Helvetica", 10))
    lbl_status.pack(pady=(20, 10))

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(window, variable=progress_var, maximum=100)
    progress_bar.pack(fill="x", padx=20, pady=10)

    lbl_percent = ttk.Label(window, text="0%")
    lbl_percent.pack(pady=5)

    # Thread-safe callback
    def update_ui(current, total, message):
        def _update():
            if total > 0:
                percent = (current / total) * 100
                progress_var.set(percent)
                lbl_percent.config(text=f"{int(percent)}%")
            lbl_status.config(text=message)
            
            if message == "Training Completed!":
                messagebox.showinfo("Success", "Training finished successfully!")
                window.destroy()
        
        window.after(0, _update)

    # Run training in background thread
    def start_thread():
        try:
            train_model(progress_callback=update_ui)
        except Exception as e:
            window.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
            window.after(0, window.destroy)

    threading.Thread(target=start_thread, daemon=True).start()
    
    # Don't wait_window here if you want the main app to remain responsive (though grab_set makes it modal)
    # parent_root.wait_window(window)

if __name__ == "__main__":
    Database.initialize_pool()
    train_model()
    Database.close_all()
