import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
# Import the new function
from src.face_detector import detect_faces_with_score

class LiveDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Detection Benchmark")
        self.root.geometry("400x200")
        
        self.model_name = tk.StringVar(value="yolo")
        self.is_running = False
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill="both")

        # Model Selection
        ttk.Label(main_frame, text="Select Detection Model:").pack()
        
        models = ["yolo", "hog", "cnn"]
        for model in models:
            ttk.Radiobutton(main_frame, text=model.upper(), variable=self.model_name, value=model).pack(anchor="w", padx=20)
            
        # Start Button
        self.start_btn = ttk.Button(main_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(pady=10, fill="x")

    def start_camera(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.start_btn.config(state="disabled")
        
        selected_model = self.model_name.get()
        print(f"Starting camera with model: {selected_model.upper()}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.is_running = False
            self.start_btn.config(state="normal")
            return

        prev_frame_time = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detections = detect_faces_with_score(
                rgb_frame, 
                model_name=selected_model,
                confidence=0.3, # Lower confidence to see more boxes
                yolo_weights="../yolo/yolov8l-face.pt"
            )

            # Draw results
            for (location, score) in detections:
                top, right, bottom, left = location
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw text with score
                text = f"{score:.2f}"
                cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and display FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show frame
            cv2.imshow(f"Live Detection - {selected_model.upper()}", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        self.start_btn.config(state="normal")

    def on_closing(self):
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
