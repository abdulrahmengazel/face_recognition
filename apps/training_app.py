import cv2
import face_recognition
import os
import numpy as np
import threading
import customtkinter as ctk
from tkinter import messagebox
# Updated import paths
from core.database import Database
import config.settings as settings
from core.detector import detect_faces
from deepface import DeepFace

# --- COLOR PALETTE (from main.py) ---
COLORS = {
    "bg": "#344e41",
    "frame": "#3a5a40",
    "button": "#588157",
    "hover": "#a3b18a",
    "text": "#dad7cd"
}

def resize_image(image, target_size):
    height, width = image.shape[:2]
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height))

def get_encodings_for_image(image_rgb):
    dlib_enc = None
    facenet_enc = None
    locs = detect_faces(image_rgb, settings.FACE_DETECTION_MODEL, confidence=settings.YOLO_CONFIDENCE, yolo_weights=settings.YOLO_WEIGHTS)
    
    if not locs:
        return None, None

    top, right, bottom, left = locs[0]

    dlib_encs = face_recognition.face_encodings(image_rgb, [locs[0]])
    if dlib_encs:
        dlib_enc = dlib_encs[0]
        
    face_img = image_rgb[top:bottom, left:right]
    if face_img.shape[0] > 20 and face_img.shape[1] > 20:
        try:
            embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
            if embedding_objs:
                facenet_enc = np.array(embedding_objs[0]['embedding'])
        except:
            pass 
            
    return dlib_enc, facenet_enc

def train_model(training_dir="data/TrainingImages", progress_callback=None):
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

            for i, person_name in enumerate(people):
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
                        
                        img = resize_image(img, settings.TRAINING_IMAGE_SIZE)
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        dlib_enc, facenet_enc = get_encodings_for_image(rgb)
                        
                        if dlib_enc is not None: dlib_encodings.append(dlib_enc)
                        if facenet_enc is not None: facenet_encodings.append(facenet_enc)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

                final_dlib_enc = np.mean(dlib_encodings, axis=0) if dlib_encodings else None
                final_facenet_enc = np.mean(facenet_encodings, axis=0) if facenet_encodings else None

                dlib_vec_str = str(final_dlib_enc.tolist()) if final_dlib_enc is not None else None
                facenet_vec_str = str(final_facenet_enc.tolist()) if final_facenet_enc is not None else None

                cursor.execute("SELECT id FROM face_encodings WHERE person_id = %s", (person_id,))
                existing_id = cursor.fetchone()

                if existing_id:
                    cursor.execute("UPDATE face_encodings SET encoding = %s::vector, encoding_facenet = %s::vector WHERE id = %s", (dlib_vec_str, facenet_vec_str, existing_id[0]))
                else:
                    cursor.execute("INSERT INTO face_encodings (person_id, model_name, encoding, encoding_facenet) VALUES (%s, %s, %s::vector, %s::vector)", (person_id, "multi-model", dlib_vec_str, facenet_vec_str))
                
                conn.commit()
                
            if progress_callback:
                progress_callback(total_people, total_people, "Training Completed!")

    print(f"Training Finished.")

# --- GUI WRAPPER ---

def run_training_gui(parent_root):
    window = ctk.CTkToplevel(parent_root)
    window.title("Training Progress")
    window.geometry("500x200")
    window.transient(parent_root)
    window.grab_set()
    window.configure(fg_color=COLORS["bg"])
    
    window.grid_columnconfigure(0, weight=1)

    # UI Elements
    ctk.CTkLabel(window, text="Training in Progress...", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text"]).grid(row=0, column=0, padx=20, pady=(20, 10))
    
    lbl_status = ctk.CTkLabel(window, text="Initializing...", font=ctk.CTkFont(size=12), text_color=COLORS["hover"])
    lbl_status.grid(row=1, column=0, padx=20, pady=5)

    progress_bar = ctk.CTkProgressBar(window, width=400, progress_color=COLORS["button"], fg_color=COLORS["frame"])
    progress_bar.set(0)
    progress_bar.grid(row=2, column=0, padx=20, pady=10)

    lbl_percent = ctk.CTkLabel(window, text="0%", font=ctk.CTkFont(size=12), text_color=COLORS["text"])
    lbl_percent.grid(row=3, column=0, padx=20, pady=(0, 20))

    def update_ui(current, total, message):
        def _update():
            if total > 0:
                percent = current / total
                progress_bar.set(percent)
                lbl_percent.configure(text=f"{int(percent*100)}%")
            lbl_status.configure(text=message)
            
            if "Completed" in message:
                messagebox.showinfo("Success", "Training finished successfully!")
                window.destroy()
        
        window.after(0, _update)

    def start_thread():
        try:
            train_model(progress_callback=update_ui)
        except Exception as e:
            window.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
            window.after(0, window.destroy)

    threading.Thread(target=start_thread, daemon=True).start()
