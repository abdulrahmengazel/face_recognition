import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import threading

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the shared config
import src.config as config
from src.database import Database
from src.train import run_training_gui  # <-- IMPORT THE NEW GUI FUNCTION
from src.image_app import run_image_app
from src.video_app import run_video_app

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System - Main Menu")
        self.root.geometry("500x650") # Increased height
        
        Database.initialize_pool()
        
        # --- UI Elements ---
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(expand=True, fill="both")

        title_label = ttk.Label(main_frame, text="Face Recognition System", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)
        
        # --- Settings Section ---
        settings_frame = ttk.LabelFrame(main_frame, text="1. System Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=10)

        # Encoding Model Selection
        ttk.Label(settings_frame, text="Encoding Model:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.encoding_model_var = tk.StringVar(value=config.ENCODING_MODEL)
        encoding_model_combo = ttk.Combobox(settings_frame, textvariable=self.encoding_model_var, values=["dlib", "facenet"], state="readonly", width=15)
        encoding_model_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        encoding_model_combo.bind("<<ComboboxSelected>>", self.update_settings)

        # Detection Model Selection
        ttk.Label(settings_frame, text="Detection Model:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.model_var = tk.StringVar(value=config.FACE_DETECTION_MODEL)
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_var, values=["cnn", "hog", "yolo"], state="readonly", width=15)
        model_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        model_combo.bind("<<ComboboxSelected>>", self.update_settings)

        # YOLO Model Selection (Conditional)
        self.yolo_label = ttk.Label(settings_frame, text="YOLO Version:")
        self.yolo_model_var = tk.StringVar(value="YOLOv8 Medium")
        self.yolo_combo = ttk.Combobox(settings_frame, textvariable=self.yolo_model_var, values=list(config.YOLO_MODELS.keys()), state="readonly", width=15)
        self.yolo_combo.bind("<<ComboboxSelected>>", self.update_settings)
        
        # Recognition Threshold
        ttk.Label(settings_frame, text="Rec. Threshold:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.threshold_var = tk.DoubleVar(value=config.RECOGNITION_THRESHOLD)
        threshold_slider = ttk.Scale(settings_frame, from_=0.1, to=1.5, variable=self.threshold_var, orient="horizontal", command=self.update_settings)
        threshold_slider.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        self.threshold_label = ttk.Label(settings_frame, text=f"{config.RECOGNITION_THRESHOLD:.2f}")
        self.threshold_label.grid(row=3, column=2, sticky="w", padx=5, pady=2)

        # Video Scale
        ttk.Label(settings_frame, text="Video Scale:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.scale_var = tk.DoubleVar(value=config.PROCESSING_SCALE)
        scale_slider = ttk.Scale(settings_frame, from_=0.1, to=1.5, variable=self.scale_var, orient="horizontal", command=self.update_settings)
        scale_slider.grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        self.scale_label = ttk.Label(settings_frame, text=f"{config.PROCESSING_SCALE:.2f}")
        self.scale_label.grid(row=4, column=2, sticky="w", padx=5, pady=2)

        # Training Image Size
        ttk.Label(settings_frame, text="Training Size:").grid(row=5, column=0, sticky="w", padx=5, pady=2)
        self.train_size_var = tk.IntVar(value=config.TRAINING_IMAGE_SIZE[0])
        train_size_spinbox = ttk.Spinbox(settings_frame, from_=400, to=1200, increment=100, textvariable=self.train_size_var, width=8, command=self.update_settings)
        train_size_spinbox.grid(row=5, column=1, sticky="w", padx=5, pady=2)

        # --- Training Section ---
        train_frame = ttk.LabelFrame(main_frame, text="2. Training", padding=10)
        train_frame.pack(fill="x", padx=10, pady=10)
        
        train_btn = ttk.Button(train_frame, text="Run Bulk Training", command=self.run_training)
        train_btn.pack(fill="x", pady=5)

        # --- Application Section ---
        app_frame = ttk.LabelFrame(main_frame, text="3. Applications", padding=10)
        app_frame.pack(fill="x", padx=10, pady=10)
        
        image_btn = ttk.Button(app_frame, text="Image Recognition", command=self.run_image_mode)
        image_btn.pack(fill="x", pady=5)
        
        video_btn = ttk.Button(app_frame, text="Live Video Recognition", command=self.run_video_mode)
        video_btn.pack(fill="x", pady=5)

        # --- Quit ---
        quit_btn = ttk.Button(main_frame, text="Exit", command=self.on_closing)
        quit_btn.pack(pady=20)

        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.toggle_yolo_widget()

    def toggle_yolo_widget(self):
        """Shows or hides the YOLO model selection widget."""
        if self.model_var.get() == "yolo":
            self.yolo_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
            self.yolo_combo.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        else:
            self.yolo_label.grid_remove()
            self.yolo_combo.grid_remove()

    def update_settings(self, event=None):
        """Updates the shared config with the selected settings."""
        config.ENCODING_MODEL = self.encoding_model_var.get()
        config.FACE_DETECTION_MODEL = self.model_var.get()
        
        selected_yolo_name = self.yolo_model_var.get()
        config.YOLO_WEIGHTS = config.YOLO_MODELS.get(selected_yolo_name, "yolo/yolov8m-face.pt")
        
        config.RECOGNITION_THRESHOLD = round(self.threshold_var.get(), 2)
        config.PROCESSING_SCALE = round(self.scale_var.get(), 2)
        config.TRAINING_IMAGE_SIZE = (self.train_size_var.get(), self.train_size_var.get())
        
        self.threshold_label.config(text=f"{config.RECOGNITION_THRESHOLD:.2f}")
        self.scale_label.config(text=f"{config.PROCESSING_SCALE:.2f}")
        self.toggle_yolo_widget()
        
        print(f"Settings Updated: Encoding={config.ENCODING_MODEL}, Detection={config.FACE_DETECTION_MODEL}, Threshold={config.RECOGNITION_THRESHOLD}")

    def run_training(self):
        # <-- UPDATED THIS FUNCTION
        if messagebox.askyesno("Confirm", "This will scan the TrainingImages folder and update the database. Continue?"):
            try:
                # Call the new GUI function, passing the main window as the parent
                run_training_gui(self.root)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start training: {e}")

    def run_image_mode(self):
        run_image_app(self.root)

    def run_video_mode(self):
        run_video_app(self.root)

    def on_closing(self):
        print("Closing database connections...")
        Database.close_all()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
