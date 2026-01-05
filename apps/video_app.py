import customtkinter as ctk
from tkinter import messagebox
import cv2
import face_recognition
from threading import Thread, Lock
import time
import pickle
import os
import numpy as np
from collections import deque, Counter

# Updated import paths
from core.database import Database
import config.settings as settings
from core.detector import detect_faces
from deepface import DeepFace

# --- CLASSIFIER LOADING ---
classifier_model = None
label_encoder = None


def load_classifier():
    global classifier_model, label_encoder
    if os.path.exists(settings.CLASSIFIER_PATH):
        try:
            with open(settings.CLASSIFIER_PATH, 'rb') as f:
                data = pickle.load(f)
                classifier_model = data.get("classifier")
                label_encoder = data.get("label_encoder")
            print("Classifier loaded successfully.")
        except Exception as e:
            print(f"Failed to load classifier: {e}")
            classifier_model = None
            label_encoder = None
    else:
        print("Classifier file not found. Falling back to database search.")


def predict_person(encoding, cursor):
    """Predicts the person using the loaded classifier or falls back to DB."""
    if classifier_model and label_encoder:
        try:
            encoding_reshaped = encoding.reshape(1, -1)
            probs = classifier_model.predict_proba(encoding_reshaped)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]

            if confidence > 0.5:
                name = label_encoder.inverse_transform([best_idx])[0]
                return name, (1.0 - confidence)
            else:
                return None, 1.0
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 1.0
    else:
        return find_nearest_face_in_db(encoding, cursor)

def find_nearest_face_in_db(encoding_to_check, cursor):
    try:
        vec_str = str(encoding_to_check.tolist()) if hasattr(encoding_to_check, 'tolist') else str(encoding_to_check)
        
        if settings.ENCODING_MODEL == "dlib":
            column_name, op = "encoding", "<->"
        else:
            column_name, op = f"encoding_{settings.ENCODING_MODEL}", "<=>"

        query = f"SELECT p.name, f.{column_name} {op} %s AS distance FROM people p JOIN face_encodings f ON p.id = f.person_id WHERE f.{column_name} IS NOT NULL ORDER BY distance ASC LIMIT 1;"
        cursor.execute(query, (vec_str,))
        return cursor.fetchone() or (None, None)
    except Exception as e:
        print(f"Veritabanı arama hatası: {e}")
        return None, None

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened(): self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            time.sleep(0.005)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.stream: self.stream.release()


class FaceTracker:
    """Simple tracker to smooth results over multiple frames."""

    def __init__(self, max_history=5):
        self.history = {}  # Key: Face ID (approx location), Value: Deque of names
        self.max_history = max_history
        self.next_id = 0
        self.faces = {}  # Key: Face ID, Value: (top, right, bottom, left)

    def update(self, current_faces):
        """
        current_faces: List of tuples (top, right, bottom, left, name, score)
        Returns: List of stabilized results
        """
        new_faces = {}
        stabilized_results = []

        # Simple tracking by IoU/Distance (Greedy matching)
        used_ids = set()

        for (top, right, bottom, left, name, score) in current_faces:
            best_id = -1
            min_dist = 10000  # Large number

            center_x = (left + right) / 2
            center_y = (top + bottom) / 2

            # Find closest existing face
            for fid, (old_top, old_right, old_bottom, old_left) in self.faces.items():
                if fid in used_ids: continue

                old_cx = (old_left + old_right) / 2
                old_cy = (old_top + old_bottom) / 2

                dist = np.sqrt((center_x - old_cx) ** 2 + (center_y - old_cy) ** 2)

                # Threshold for movement (e.g., 50 pixels)
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    best_id = fid

            if best_id != -1:
                # Update existing face
                fid = best_id
                used_ids.add(fid)
            else:
                # New face
                fid = self.next_id
                self.next_id += 1
                self.history[fid] = deque(maxlen=self.max_history)

            new_faces[fid] = (top, right, bottom, left)

            # Add prediction to history
            if name != "BILINMIYOR":
                self.history[fid].append(name)

            # Vote for best name
            final_name = "BILINMIYOR"
            if len(self.history[fid]) > 0:
                # Get most common name
                counts = Counter(self.history[fid])
                most_common = counts.most_common(1)[0]
                # Only stabilize if we have enough confidence (e.g. > 40% of history agrees)
                if most_common[1] >= 1:
                    final_name = most_common[0]

            # If current detection is unknown but history says "Ahmed", keep "Ahmed"
            # If current is "Ahmed" but history is empty, show "Ahmed"

            # Format score for display
            conf_display = f"{(1 - score) * 100:.0f}%" if classifier_model else f"{score:.2f}"
            display_text = f"{final_name} ({conf_display})" if final_name != "BILINMIYOR" else "BILINMIYOR"
            color = (0, 255, 0) if final_name != "BILINMIYOR" else (0, 0, 255)

            stabilized_results.append((top, right, bottom, left, display_text, color))

        self.faces = new_faces

        # Cleanup old history
        active_ids = set(new_faces.keys())
        keys_to_remove = [k for k in self.history.keys() if k not in active_ids]
        for k in keys_to_remove:
            del self.history[k]

        return stabilized_results

class FaceProcessingThread:
    def __init__(self, video_stream):
        self.video_stream = video_stream
        self.stopped = False
        self.latest_results = [] 
        self.lock = Lock()
        self.tracker = FaceTracker(max_history=8)  # Smooth over 8 frames

    def start(self):
        Thread(target=self.process, args=(), daemon=True).start()
        return self

    def process(self):
        with Database.get_conn() as conn:
            with conn.cursor() as cursor:
                while not self.stopped:
                    frame = self.video_stream.read()
                    if frame is None: 
                        time.sleep(0.01)
                        continue

                    try:
                        if settings.PROCESSING_SCALE < 1.0:
                            small_frame = cv2.resize(frame, (0, 0), fx=settings.PROCESSING_SCALE, fy=settings.PROCESSING_SCALE)
                        else:
                            small_frame = frame
                            
                        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                        face_locations = detect_faces(rgb_frame, settings.FACE_DETECTION_MODEL, confidence=settings.YOLO_CONFIDENCE, yolo_weights=settings.YOLO_WEIGHTS)
                        
                        encodings = []
                        if settings.ENCODING_MODEL == "dlib":
                            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        else:
                            for (top, right, bottom, left) in face_locations:
                                face_img = rgb_frame[top:bottom, left:right]
                                try:
                                    embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                                    if embedding_objs: encodings.append(embedding_objs[0]['embedding'])
                                except:
                                    encodings.append(None)

                        raw_results = []
                        for i, location in enumerate(face_locations):
                            encoding = encodings[i] if i < len(encodings) else None

                            name = "BILINMIYOR"
                            score = 1.0

                            if encoding is not None:
                                db_name, db_score = predict_person(encoding, cursor)
                                threshold = 0.5 if classifier_model else settings.RECOGNITION_THRESHOLD

                                if db_name and db_score < threshold:
                                    name = db_name.upper()
                                    score = db_score
                            
                            top, right, bottom, left = location
                            if settings.PROCESSING_SCALE < 1.0:
                                scale_factor = 1.0 / settings.PROCESSING_SCALE
                                top, right, bottom, left = [int(v * scale_factor) for v in location]

                            raw_results.append((top, right, bottom, left, name, score))

                        # Apply Stabilization
                        stabilized_results = self.tracker.update(raw_results)

                        with self.lock:
                            self.latest_results = stabilized_results
                            
                    except Exception as e:
                        print(f"İşleme Hatası: {e}")

    def get_results(self):
        with self.lock:
            return self.latest_results

    def stop(self):
        self.stopped = True

def run_video_app(parent_root):
    # Load classifier when app starts
    load_classifier()
    
    window = ctk.CTkToplevel(parent_root)
    window.title("Canlı Tanıma")
    window.geometry("400x220")
    
    window.transient(parent_root)
    window.grab_set()
    window.configure(fg_color=settings.UI_COLORS["bg"])
    window.grid_columnconfigure(0, weight=1)

    def start_recognition_program():
        window.withdraw()
        
        video_stream = VideoStream(src=0)
        if not video_stream.stream.isOpened():
            messagebox.showerror("Kamera Hatası", "Web kamerası açılamadı.")
            window.deiconify()
            return
        
        video_stream.start()
        processor = FaceProcessingThread(video_stream).start()

        while True:
            frame = video_stream.read()
            if frame is None: break

            results = processor.get_results()

            for (top, right, bottom, left, name, color) in results:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

            cv2.putText(frame, "Cikis: Q", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Canli Tanima", frame)
            
            # Q tuşu veya pencere kapatma kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty("Canli Tanima", cv2.WND_PROP_VISIBLE) < 1:
                break

        processor.stop()
        video_stream.stop()
        cv2.destroyAllWindows()
        
        # Pencereyi geri getir
        try:
            if window.winfo_exists():
                window.deiconify()
        except: pass

    ctk.CTkLabel(window, text="Canlı Kamera Tanıma", font=ctk.CTkFont(size=16, weight="bold"),
                 text_color=settings.UI_COLORS["text"]).grid(row=0, column=0, pady=(20, 10))

    model_text = f"{settings.ENCODING_MODEL.upper()} + Classifier" if classifier_model else f"{settings.ENCODING_MODEL.upper()} (DB Search)"
    ctk.CTkLabel(window, text=model_text, font=ctk.CTkFont(size=12), text_color=settings.UI_COLORS["hover"]).grid(row=1,
                                                                                                                  column=0,
                                                                                                                  pady=(
                                                                                                                      0,
                                                                                                                      20))

    start_btn = ctk.CTkButton(window, text="Kamerayı Başlat", command=start_recognition_program, height=40,
                              fg_color=settings.UI_COLORS["button"], hover_color=settings.UI_COLORS["hover"],
                              text_color=settings.UI_COLORS["text"])
    start_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

    quit_btn = ctk.CTkButton(window, text="Kapat", command=window.destroy, fg_color="transparent", border_width=1,
                             border_color=settings.UI_COLORS["hover"])
    quit_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
    
if __name__ == "__main__":
    app = ctk.CTk()
    Database.initialize_pool()
    run_video_app(app)
    app.mainloop()
    Database.close_all()
