import customtkinter as ctk
from tkinter import messagebox
import cv2
import face_recognition
from threading import Thread, Lock
import time
# Updated import paths
from core.database import Database
import config.settings as settings
from core.detector import detect_faces
from deepface import DeepFace

# --- DATABASE & HELPER FUNCTIONS ---

def find_nearest_face_in_db(encoding_to_check, cursor):
    """Searches the DB for the closest face using the correct column and distance operator."""
    try:
        vec_str = str(encoding_to_check.tolist()) if hasattr(encoding_to_check, 'tolist') else str(encoding_to_check)
        
        if settings.ENCODING_MODEL == "dlib":
            column_name = "encoding"
            distance_operator = "<->"
        else:
            column_name = f"encoding_{settings.ENCODING_MODEL}"
            distance_operator = "<=>"

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
                        if settings.PROCESSING_SCALE < 1.0:
                            small_frame = cv2.resize(frame, (0, 0), fx=settings.PROCESSING_SCALE, fy=settings.PROCESSING_SCALE)
                        else:
                            small_frame = frame
                            
                        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                        face_locations = detect_faces(rgb_frame, settings.FACE_DETECTION_MODEL, confidence=settings.YOLO_CONFIDENCE, yolo_weights=settings.YOLO_WEIGHTS)
                        
                        if settings.ENCODING_MODEL == "dlib":
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
                                if db_name and distance < settings.RECOGNITION_THRESHOLD:
                                    name = f"{db_name.upper()} ({distance:.2f})"
                                    color = (0, 255, 0)
                            
                            top, right, bottom, left = location
                            if settings.PROCESSING_SCALE < 1.0:
                                scale_factor = 1.0 / settings.PROCESSING_SCALE
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
    window = ctk.CTkToplevel(parent_root)
    window.title("Live Recognition")
    window.geometry("400x200")
    
    window.transient(parent_root)
    window.grab_set()
    window.grid_columnconfigure(0, weight=1)

    def start_recognition_program():
        window.withdraw() # Hide the menu
        
        video_stream = VideoStream(src=0).start()
        if not video_stream.stream.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            window.deiconify() # Show the menu again
            return
        
        print(f"Starting Face Recognition (Threshold: {settings.RECOGNITION_THRESHOLD})...")
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
        window.deiconify() # Show the menu again on exit

    ctk.CTkLabel(window, text=f"Mode: {settings.ENCODING_MODEL.upper()} | Detection: {settings.FACE_DETECTION_MODEL.upper()}", font=ctk.CTkFont(size=12)).grid(row=0, column=0, pady=(10,5))

    start_btn = ctk.CTkButton(window, text="Start Camera", command=start_recognition_program)
    start_btn.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

    quit_btn = ctk.CTkButton(window, text="Close", command=window.destroy, fg_color="transparent", border_width=2)
    quit_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
    
if __name__ == "__main__":
    app = ctk.CTk()
    Database.initialize_pool()
    run_video_app(app)
    app.mainloop()
    Database.close_all()
