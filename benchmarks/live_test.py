import customtkinter as ctk
from tkinter import messagebox
import cv2
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config.settings as settings
from core.detector import detect_faces_with_score

class LiveDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Detection Benchmark")
        self.root.geometry("400x250")
        
        self.model_name = ctk.StringVar(value="yolo")
        self.is_running = False
        
        self.create_widgets()

    def create_widgets(self):
        self.root.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.root, text="Select Detection Model:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(20,10))
        
        models = ["yolo", "hog", "cnn"]
        for i, model in enumerate(models):
            ctk.CTkRadioButton(self.root, text=model.upper(), variable=self.model_name, value=model).grid(row=i+1, column=0, padx=40, pady=5, sticky="w")
            
        self.start_btn = ctk.CTkButton(self.root, text="Start Camera", command=self.start_camera)
        self.start_btn.grid(row=len(models)+1, column=0, padx=20, pady=20, sticky="ew")

    def start_camera(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.root.withdraw() # Hide the menu
        
        selected_model = self.model_name.get()
        print(f"Starting camera with model: {selected_model.upper()}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.is_running = False
            self.start_btn.configure(state="normal")
            self.root.deiconify()
            return

        prev_frame_time = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detections = detect_faces_with_score(
                rgb_frame, 
                model_name=selected_model,
                confidence=0.3, 
                yolo_weights=settings.YOLO_WEIGHTS
            )

            for (location, score) in detections:
                top, right, bottom, left = location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                text = f"{score:.2f}"
                cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(f"Live Detection - {selected_model.upper()}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        try:
            self.start_btn.configure(state="normal")
            self.root.deiconify()
        except:
            pass

    def on_closing(self):
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    app = ctk.CTk()
    LiveDetectionApp(app)
    # The mainloop is handled by the class now
