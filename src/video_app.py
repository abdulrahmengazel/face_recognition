import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
import cv2
import face_recognition
from datetime import datetime
from threading import Thread, Lock
import time
from src.database import Database
import src.config as config
from src.face_detector import detect_faces
from deepface import DeepFace

# --- DATABASE & HELPER FUNCTIONS ---

def add_new_face():
    messagebox.showinfo("Info", "Please use the 'Run Bulk Training' button in the main menu for adding new faces.")

def find_nearest_face_in_db(encoding_to_check, cursor):
    """Searches the DB for the closest face using the correct column and distance operator."""
    try:
        vec_str = str(encoding_to_check.tolist()) if hasattr(encoding_to_check, 'tolist') else str(encoding_to_check)
        
        if config.ENCODING_MODEL == "dlib":
            column_name = "encoding"
            distance_operator = "<->"  # L2 distance for dlib
        else:
            column_name = f"encoding_{config.ENCODING_MODEL}"
            distance_operator = "<=>"  # Cosine distance for FaceNet

        query = f"""
            SELECT p.name, f.{column_name} {distance_operator} %s AS distance
            FROM people p JOIN face_encodings f ON p.id = f.person_id
            WHERE f.{column_name} IS NOT NULL
            ORDER BY distance ASC
            LIMIT 1;
        """
        cursor.execute(query, (vec_str,))
        result = cursor.fetchone()
        return result if result else (None, None)
    except Exception as e:
        print(f"Database search error: {e}")
        return None, None

# --- THREADING CLASSES ---

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

class FaceProcessingThread:
    def __init__(self, video_stream):
        self.video_stream = video_stream
        self.stopped = False
        self.latest_results = [] 
        self.lock = Lock()

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
                        if config.PROCESSING_SCALE < 1.0:
                            small_frame = cv2.resize(frame, (0, 0), fx=config.PROCESSING_SCALE, fy=config.PROCESSING_SCALE)
                        else:
                            small_frame = frame
                            
                        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                        face_locations = detect_faces(rgb_frame, config.FACE_DETECTION_MODEL, confidence=config.YOLO_CONFIDENCE, yolo_weights=config.YOLO_WEIGHTS)
                        
                        if config.ENCODING_MODEL == "dlib":
                            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        else: # facenet
                            face_encodings = []
                            for (top, right, bottom, left) in face_locations:
                                face_img = rgb_frame[top:bottom, left:right]
                                try:
                                    embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                                    if embedding_objs:
                                        face_encodings.append(embedding_objs[0]['embedding'])
                                except:
                                    face_encodings.append(None)

                        results = []
                        for i, location in enumerate(face_locations):
                            face_encoding = face_encodings[i] if i < len(face_encodings) else None
                            
                            name = "UNKNOWN"
                            color = (0, 0, 255)

                            if face_encoding is not None:
                                db_name, distance = find_nearest_face_in_db(face_encoding, cursor)
                                if db_name and distance < config.RECOGNITION_THRESHOLD:
                                    name = db_name.upper()
                                    color = (0, 255, 0)
                            
                            top, right, bottom, left = location
                            if config.PROCESSING_SCALE < 1.0:
                                scale_factor = 1.0 / config.PROCESSING_SCALE
                                top, right, bottom, left = [int(v * scale_factor) for v in location]
                                
                            results.append((top, right, bottom, left, name, color))
                        
                        with self.lock:
                            self.latest_results = results
                            
                    except Exception as e:
                        print(f"Processing Error: {e}")

    def get_results(self):
        with self.lock:
            return self.latest_results

    def stop(self):
        self.stopped = True

# --- MAIN APPLICATION LOGIC ---

def run_video_app(parent_root):
    window = tk.Toplevel(parent_root)
    window.title(f"Live Recognition (Encoding: {config.ENCODING_MODEL}, Detection: {config.FACE_DETECTION_MODEL})")
    window.geometry("400x250")
    
    window.transient(parent_root)
    window.grab_set()

    main_frame = ttk.Frame(window, padding="20")
    main_frame.pack(expand=True, fill="both")

    def start_recognition_program():
        video_stream = VideoStream(src=0).start()
        if video_stream.stopped:
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return
        
        print(f"Starting Face Recognition (Threshold: {config.RECOGNITION_THRESHOLD})...")
        processor = FaceProcessingThread(video_stream).start()

        while True:
            frame = video_stream.read()
            if frame is None: break

            results = processor.get_results()

            for (top, right, bottom, left, name, color) in results:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Live Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        processor.stop()
        video_stream.stop()
        cv2.destroyAllWindows()

    add_face_btn = ttk.Button(main_frame, text="Add New Face", command=add_new_face)
    add_face_btn.pack(pady=10, fill="x")

    start_btn = ttk.Button(main_frame, text="Start Camera", command=start_recognition_program)
    start_btn.pack(pady=10, fill="x")

    quit_btn = ttk.Button(main_frame, text="Close", command=window.destroy)
    quit_btn.pack(pady=10, fill="x")
    
    parent_root.wait_window(window)

if __name__ == "__main__":
    import numpy as np 
    root = tk.Tk()
    Database.initialize_pool()
    run_video_app(root)
    Database.close_all()
    root.destroy()
