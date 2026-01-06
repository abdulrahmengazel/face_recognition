import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import face_recognition
import pickle
import os
import numpy as np
# Updated import paths
from core.database import Database
import config.settings as settings
from core.detector import detect_faces
from deepface import DeepFace

# --- CLASSIFIER LOADING ---
classifier_model = None
label_encoder = None
scaler = None  # Add scaler

def load_classifier():
    global classifier_model, label_encoder, scaler
    if os.path.exists(settings.CLASSIFIER_PATH):
        try:
            with open(settings.CLASSIFIER_PATH, 'rb') as f:
                data = pickle.load(f)
                classifier_model = data.get("classifier")
                label_encoder = data.get("label_encoder")
                scaler = data.get("scaler")  # Load scaler
            print("Classifier loaded successfully.")
        except Exception as e:
            print(f"Failed to load classifier: {e}")
            classifier_model = None
            label_encoder = None
            scaler = None
    else:
        print("Classifier file not found. Falling back to database search.")


def predict_person(encoding):
    """Predicts the person using the loaded classifier or falls back to DB."""
    if classifier_model and label_encoder and scaler:
        try:
            # Ensure encoding is a numpy array
            if isinstance(encoding, list):
                encoding = np.array(encoding)
                
            # Reshape for single sample prediction
            encoding_reshaped = encoding.reshape(1, -1)

            # Scale the input
            encoding_scaled = scaler.transform(encoding_reshaped)

            # Get probabilities
            probs = classifier_model.predict_proba(encoding_scaled)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]

            # Invert confidence to match "distance" logic (lower is better for distance, higher is better for prob)
            # We'll just return the name and (1 - confidence) as a pseudo-distance for compatibility
            if confidence > 0.5:  # Threshold for classifier
                name = label_encoder.inverse_transform([best_idx])[0]
                return name, (1.0 - confidence)
            else:
                return None, 1.0
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 1.0
    else:
        # Fallback to Database Search
        return find_nearest_face_in_db(encoding)

def find_nearest_face_in_db(encoding_to_check):
    vec_str = str(encoding_to_check.tolist()) if hasattr(encoding_to_check, 'tolist') else str(encoding_to_check)
    
    if settings.ENCODING_MODEL == "dlib":
        column_name, op = "encoding", "<->"
    else:
        column_name, op = f"encoding_{settings.ENCODING_MODEL}", "<=>"

    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            try:
                query = f"SELECT p.name, f.{column_name} {op} %s AS distance FROM people p JOIN face_encodings f ON p.id = f.person_id WHERE f.{column_name} IS NOT NULL ORDER BY distance ASC LIMIT 1;"
                cursor.execute(query, (vec_str,))
                return cursor.fetchone() or (None, None)
            except Exception as e:
                print(f"Veritabanı Arama Hatası: {e}")
                return None, None

def select_and_recognize_image():
    file_path = filedialog.askopenfilename(
        title="Bir Resim Seçin",
        filetypes=(("Resim Dosyaları", "*.jpg *.jpeg *.png"), ("Tüm Dosyalar", "*.*"))
    )
    if not file_path: return

    try:
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            messagebox.showerror("Hata", "Resim dosyası okunamadı.")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        locations = detect_faces(image_rgb, settings.FACE_DETECTION_MODEL, confidence=settings.YOLO_CONFIDENCE, yolo_weights=settings.YOLO_WEIGHTS)
        
        encodings = []
        if settings.ENCODING_MODEL == "dlib":
            encodings = face_recognition.face_encodings(image_rgb, locations)
        else:
            for (top, right, bottom, left) in locations:
                face_img = image_rgb[top:bottom, left:right]
                if face_img.size > 0:
                    try:
                        embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                        encodings.append(embedding_objs[0]['embedding'])
                    except:
                        encodings.append(None)
        
        original_height, original_width = image_bgr.shape[:2]
        target_width, target_height = 800, 600
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        image_display = cv2.resize(image_bgr, (target_width, target_height))

        if not locations:
            messagebox.showinfo("Sonuç", "Resimde yüz bulunamadı.")
        else:
            for i, (top, right, bottom, left) in enumerate(locations):
                encoding = encodings[i] if i < len(encodings) else None
                
                name, color = "BILINMIYOR", (0, 0, 255)
                
                if encoding is not None:
                    # Use the new prediction logic
                    db_name, score = predict_person(encoding)

                    # Logic: If using classifier, score is (1-prob). If DB, score is distance.
                    # Both cases: Lower is better/more confident match.
                    threshold = 0.5 if classifier_model else settings.RECOGNITION_THRESHOLD

                    if db_name and score < threshold:
                        conf_display = f"{(1 - score) * 100:.1f}%" if classifier_model else f"{score:.2f}"
                        name = f"{db_name.upper()} ({conf_display})"
                        color = (0, 255, 0)

                new_top, new_bottom = int(top * scale_y), int(bottom * scale_y)
                new_left, new_right = int(left * scale_x), int(right * scale_x)

                cv2.rectangle(image_display, (new_left, new_top), (new_right, new_bottom), color, 2)
                cv2.putText(image_display, name, (new_left + 6, new_bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Tanima Sonucu", image_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Hata", f"Resim işlenirken hata oluştu: {e}")

def run_image_app(parent_root):
    # Load classifier when app starts
    load_classifier()
    
    window = ctk.CTkToplevel(parent_root)
    window.title("Resim Tanıma")
    window.geometry("400x220")
    
    window.transient(parent_root)
    window.grab_set()
    window.configure(fg_color=settings.UI_COLORS["bg"])
    window.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(window, text="Resim Analizi", font=ctk.CTkFont(size=16, weight="bold"),
                 text_color=settings.UI_COLORS["text"]).grid(row=0, column=0, pady=(20, 10))

    model_text = f"{settings.ENCODING_MODEL.upper()} + Classifier" if classifier_model else f"{settings.ENCODING_MODEL.upper()} (DB Search)"
    ctk.CTkLabel(window, text=model_text, font=ctk.CTkFont(size=12), text_color=settings.UI_COLORS["hover"]).grid(row=1,
                                                                                                                  column=0,
                                                                                                                  pady=(
                                                                                                                      0,
                                                                                                                      20))

    recognize_btn = ctk.CTkButton(window, text="Analiz İçin Resim Seç", command=select_and_recognize_image, height=40,
                                  fg_color=settings.UI_COLORS["button"], hover_color=settings.UI_COLORS["hover"],
                                  text_color=settings.UI_COLORS["text"])
    recognize_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

    quit_btn = ctk.CTkButton(window, text="Kapat", command=window.destroy, fg_color="transparent", border_width=1,
                             border_color=settings.UI_COLORS["hover"])
    quit_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
    
if __name__ == "__main__":
    app = ctk.CTk()
    Database.initialize_pool()
    run_image_app(app)
    app.mainloop()
    Database.close_all()
