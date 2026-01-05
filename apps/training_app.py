import cv2
import face_recognition
import os
import numpy as np
import threading
import customtkinter as ctk
from tkinter import messagebox
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Updated import paths
from core.database import Database
import config.settings as settings
from core.detector import detect_faces
from deepface import DeepFace

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

    # --- DLIB ENCODING ---
    if settings.ENCODING_MODEL == "dlib":
        jitter = settings.TRAINING_CONFIG.get("dlib", {}).get("jitter", 1)
        dlib_encs = face_recognition.face_encodings(image_rgb, [locs[0]], num_jitters=jitter)
        if dlib_encs:
            dlib_enc = dlib_encs[0]

    # --- FACENET ENCODING ---
    elif settings.ENCODING_MODEL == "facenet":
        face_img = image_rgb[top:bottom, left:right]
        if face_img.shape[0] > 20 and face_img.shape[1] > 20:
            try:
                embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                if embedding_objs:
                    facenet_enc = np.array(embedding_objs[0]['embedding'])
            except:
                pass 
            
    return dlib_enc, facenet_enc


def train_classifier(progress_callback=None):
    """Trains an MLP Classifier on the stored embeddings."""
    print("Training Classifier...")
    if progress_callback: progress_callback(0, 0, "Sınıflandırıcı Eğitiliyor...")

    X = []
    y = []
    names = {}

    # Determine which column to fetch based on the selected model
    if settings.ENCODING_MODEL == "dlib":
        column_name = "encoding"
    else:
        column_name = "encoding_facenet"

    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            # Fetch encodings ONLY for the selected model
            query = f"""
                SELECT p.name, f.{column_name} 
                FROM face_encodings f 
                JOIN people p ON f.person_id = p.id 
                WHERE f.{column_name} IS NOT NULL
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            for name, encoding_str in rows:
                if encoding_str:
                    try:
                        clean_str = encoding_str.replace('[', '').replace(']', '')
                        encoding = np.fromstring(clean_str, sep=',')
                        if len(encoding) == 128:
                            X.append(encoding)
                            y.append(name)
                    except Exception as e:
                        print(f"Error parsing encoding for {name}: {e}")

    if len(X) < 2:
        print("Not enough data to train classifier (need at least 2 classes/samples).")
        return

    # Train MLP Classifier
    clf_config = settings.TRAINING_CONFIG.get("classifier", {})
    clf = MLPClassifier(
        hidden_layer_sizes=clf_config.get("hidden_layers", (128, 64)),
        max_iter=clf_config.get("max_iter", 500),
        solver=clf_config.get("solver", "adam"),
        random_state=42,
        verbose=True
    )

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf.fit(X, y_encoded)

    # Save Model and Label Encoder
    model_data = {"classifier": clf, "label_encoder": le}
    with open(settings.CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Classifier saved to {settings.CLASSIFIER_PATH}")

def train_model(training_dir="data/TrainingImages", progress_callback=None):
    if not os.path.exists(training_dir):
        if progress_callback: progress_callback(0, 0, "Eğitim klasörü bulunamadı!")
        return

    print(f"Birleşik Eğitim Başlatılıyor...")

    # --- CONFIG LOGGING ---
    yolo_cfg = settings.TRAINING_CONFIG.get("yolo", {})
    facenet_cfg = settings.TRAINING_CONFIG.get("facenet", {})
    dlib_cfg = settings.TRAINING_CONFIG.get("dlib", {})

    print(f"--- Aktif Eğitim Parametreleri ---")
    print(f"Model: {settings.ENCODING_MODEL.upper()}")
    if settings.ENCODING_MODEL == "dlib":
        print(f"[Dlib] Jitter (Tekrar Örnekleme): {dlib_cfg.get('jitter', 1)}")
    elif settings.ENCODING_MODEL == "facenet":
        print(
            f"[FaceNet] Fine-Tuning Hedefi: Epochs={facenet_cfg.get('epochs')}, Batch={facenet_cfg.get('batch_size')}, LR={facenet_cfg.get('learning_rate')}")
    print(f"----------------------------------")

    Database.init_tables()

    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            people = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]
            total_people = len(people)
            
            if total_people == 0:
                if progress_callback: progress_callback(0, 0, "Eğitilecek kişi bulunamadı.")
                return

            for i, person_name in enumerate(people):
                if progress_callback:
                    progress_callback(i, total_people, f"İşleniyor: {person_name}...")

                person_path = os.path.join(training_dir, person_name)

                # Get or Create Person ID
                cursor.execute("SELECT id FROM people WHERE name = %s;", (person_name,))
                row = cursor.fetchone()
                person_id = row[0] if row else cursor.execute("INSERT INTO people (name) VALUES (%s) RETURNING id;", (person_name,)) or cursor.fetchone()[0]

                # --- CLEANUP OLD ENCODINGS FOR THIS MODEL ---
                # We delete old encodings for this person/model to avoid duplicates or stale data
                if settings.ENCODING_MODEL == "dlib":
                    cursor.execute("DELETE FROM face_encodings WHERE person_id = %s AND encoding IS NOT NULL",
                                   (person_id,))
                else:
                    cursor.execute("DELETE FROM face_encodings WHERE person_id = %s AND encoding_facenet IS NOT NULL",
                                   (person_id,))
                
                images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_path in images:
                    try:
                        with open(img_path, 'rb') as f:
                            file_bytes = np.fromfile(f, dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        if img is None: continue
                        
                        img = resize_image(img, settings.TRAINING_IMAGE_SIZE)
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        dlib_enc, facenet_enc = get_encodings_for_image(rgb)

                        # --- INSERT INDIVIDUAL ENCODINGS ---
                        if settings.ENCODING_MODEL == "dlib" and dlib_enc is not None:
                            vec_str = str(dlib_enc.tolist())
                            cursor.execute(
                                "INSERT INTO face_encodings (person_id, model_name, encoding) VALUES (%s, %s, %s::vector)",
                                (person_id, "dlib", vec_str))

                        elif settings.ENCODING_MODEL == "facenet" and facenet_enc is not None:
                            vec_str = str(facenet_enc.tolist())
                            cursor.execute(
                                "INSERT INTO face_encodings (person_id, model_name, encoding_facenet) VALUES (%s, %s, %s::vector)",
                                (person_id, "facenet", vec_str))
                            
                    except Exception as e:
                        print(f"Hata: {img_path} işlenemedi: {e}")

                conn.commit()

            # --- Train Classifier After Enrollment ---
            train_classifier(progress_callback)

            if progress_callback:
                progress_callback(total_people, total_people, "Eğitim Tamamlandı!")

    print(f"Eğitim Bitti.")

# --- GUI WRAPPER ---

def run_training_gui(parent_root):
    window = ctk.CTkToplevel(parent_root)
    window.title("Eğitim İlerlemesi")
    window.geometry("500x200")
    window.transient(parent_root)
    window.grab_set()
    window.configure(fg_color=settings.UI_COLORS["bg"])
    
    window.grid_columnconfigure(0, weight=1)

    # UI Elements
    ctk.CTkLabel(window, text="Eğitim Devam Ediyor...", font=ctk.CTkFont(size=16, weight="bold"),
                 text_color=settings.UI_COLORS["text"]).grid(row=0, column=0, padx=20, pady=(20, 10))

    lbl_status = ctk.CTkLabel(window, text="Başlatılıyor...", font=ctk.CTkFont(size=12),
                              text_color=settings.UI_COLORS["hover"])
    lbl_status.grid(row=1, column=0, padx=20, pady=5)

    progress_bar = ctk.CTkProgressBar(window, width=400, progress_color=settings.UI_COLORS["button"],
                                      fg_color=settings.UI_COLORS["frame"])
    progress_bar.set(0)
    progress_bar.grid(row=2, column=0, padx=20, pady=10)

    lbl_percent = ctk.CTkLabel(window, text="0%", font=ctk.CTkFont(size=12), text_color=settings.UI_COLORS["text"])
    lbl_percent.grid(row=3, column=0, padx=20, pady=(0, 20))

    def update_ui(current, total, message):
        def _update():
            if total > 0:
                percent = current / total
                progress_bar.set(percent)
                lbl_percent.configure(text=f"{int(percent*100)}%")
            lbl_status.configure(text=message)
            
            if "Tamamlandı" in message:
                messagebox.showinfo("Başarılı", "Eğitim başarıyla tamamlandı!")
                window.destroy()
        
        window.after(0, _update)

    def start_thread():
        try:
            train_model(progress_callback=update_ui)
        except Exception as e:
            window.after(0, lambda: messagebox.showerror("Hata", f"Eğitim başarısız oldu: {e}"))
            window.after(0, window.destroy)

    threading.Thread(target=start_thread, daemon=True).start()
