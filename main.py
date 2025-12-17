import customtkinter as ctk
from tkinter import messagebox
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import lightweight modules
import config.settings as settings
from core.database import Database

# --- THEME SETTINGS ---
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition System - Main Menu")
        self.geometry("600x700")
        
        Database.initialize_pool()
        
        # --- UI Elements ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self, corner_radius=15)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ctk.CTkLabel(main_frame, text="Face Recognition System", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # --- Settings Section ---
        settings_frame = ctk.CTkFrame(main_frame)
        settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        settings_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings_frame, text="1. System Settings", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=3, pady=(10, 5), sticky="w", padx=10)

        # Encoding Model
        ctk.CTkLabel(settings_frame, text="Encoding Model:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.encoding_model_var = ctk.StringVar(value=settings.ENCODING_MODEL)
        encoding_model_combo = ctk.CTkComboBox(settings_frame, variable=self.encoding_model_var, values=["dlib", "facenet"], state="readonly", command=self.update_settings)
        encoding_model_combo.grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        # Detection Model
        ctk.CTkLabel(settings_frame, text="Detection Model:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.model_var = ctk.StringVar(value=settings.FACE_DETECTION_MODEL)
        model_combo = ctk.CTkComboBox(settings_frame, variable=self.model_var, values=["cnn", "hog", "yolo"], state="readonly", command=self.update_settings)
        model_combo.grid(row=2, column=1, sticky="ew", padx=10, pady=5)

        # YOLO Model (Conditional)
        self.yolo_label = ctk.CTkLabel(settings_frame, text="YOLO Version:")
        self.yolo_model_var = ctk.StringVar(value="YOLOv8 Medium")
        self.yolo_combo = ctk.CTkComboBox(settings_frame, variable=self.yolo_model_var, values=list(settings.YOLO_MODELS.keys()), state="readonly", command=self.update_settings)
        
        # Recognition Threshold
        ctk.CTkLabel(settings_frame, text="Rec. Threshold:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.threshold_label = ctk.CTkLabel(settings_frame, text=f"{settings.RECOGNITION_THRESHOLD:.2f}")
        self.threshold_label.grid(row=4, column=2, sticky="w", padx=10, pady=5)
        
        self.threshold_slider = ctk.CTkSlider(settings_frame, from_=0.1, to=1.5, number_of_steps=14, command=self.update_threshold_label)
        self.threshold_slider.set(settings.RECOGNITION_THRESHOLD)
        self.threshold_slider.grid(row=4, column=1, sticky="ew", padx=10, pady=5)

        # Video Scale
        ctk.CTkLabel(settings_frame, text="Video Scale:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.scale_label = ctk.CTkLabel(settings_frame, text=f"{settings.PROCESSING_SCALE:.2f}")
        self.scale_label.grid(row=5, column=2, sticky="w", padx=10, pady=5)
        
        self.scale_slider = ctk.CTkSlider(settings_frame, from_=0.1, to=1.5, number_of_steps=14, command=self.update_scale_label)
        self.scale_slider.set(settings.PROCESSING_SCALE)
        self.scale_slider.grid(row=5, column=1, sticky="ew", padx=10, pady=5)

        # Training Image Size (Simplified as Label for now, or Entry)
        ctk.CTkLabel(settings_frame, text="Training Size:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        self.train_size_entry = ctk.CTkEntry(settings_frame, placeholder_text="800")
        self.train_size_entry.insert(0, str(settings.TRAINING_IMAGE_SIZE[0]))
        self.train_size_entry.grid(row=6, column=1, sticky="ew", padx=10, pady=5)
        
        # Apply Button for Settings
        apply_btn = ctk.CTkButton(settings_frame, text="Apply Settings", command=self.update_settings)
        apply_btn.grid(row=7, column=0, columnspan=3, pady=10, padx=10, sticky="ew")

        # --- Training Section ---
        train_frame = ctk.CTkFrame(main_frame)
        train_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        train_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(train_frame, text="2. Training", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        train_btn = ctk.CTkButton(train_frame, text="Run Bulk Training", command=self.run_training, fg_color="#2CC985", hover_color="#229F69")
        train_btn.pack(fill="x", padx=10, pady=10)

        # --- Application Section ---
        app_frame = ctk.CTkFrame(main_frame)
        app_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        app_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(app_frame, text="3. Applications", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        image_btn = ctk.CTkButton(app_frame, text="Image Recognition", command=self.run_image_mode)
        image_btn.pack(fill="x", padx=10, pady=5)
        
        video_btn = ctk.CTkButton(app_frame, text="Live Video Recognition", command=self.run_video_mode, fg_color="#3B8ED0", hover_color="#36719F")
        video_btn.pack(fill="x", padx=10, pady=10)

        # --- Quit ---
        quit_btn = ctk.CTkButton(main_frame, text="Exit", command=self.on_closing, fg_color="#C0392B", hover_color="#922B21")
        quit_btn.grid(row=4, column=0, padx=20, pady=20, sticky="ew")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.toggle_yolo_widget()

    def toggle_yolo_widget(self):
        """Shows or hides the YOLO model selection widget."""
        if self.model_var.get() == "yolo":
            self.yolo_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)
            self.yolo_combo.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        else:
            self.yolo_label.grid_forget()
            self.yolo_combo.grid_forget()

    def update_threshold_label(self, value):
        self.threshold_label.configure(text=f"{value:.2f}")

    def update_scale_label(self, value):
        self.scale_label.configure(text=f"{value:.2f}")

    def update_settings(self, event=None):
        """Updates the shared config with the selected settings."""
        settings.ENCODING_MODEL = self.encoding_model_var.get()
        settings.FACE_DETECTION_MODEL = self.model_var.get()
        
        selected_yolo_name = self.yolo_model_var.get()
        settings.YOLO_WEIGHTS = settings.YOLO_MODELS.get(selected_yolo_name, "assets/yolo/yolov8m-face.pt")
        
        settings.RECOGNITION_THRESHOLD = round(self.threshold_slider.get(), 2)
        settings.PROCESSING_SCALE = round(self.scale_slider.get(), 2)
        
        try:
            size = int(self.train_size_entry.get())
            settings.TRAINING_IMAGE_SIZE = (size, size)
        except ValueError:
            pass
        
        self.toggle_yolo_widget()
        print(f"Settings Updated: Encoding={settings.ENCODING_MODEL}, Detection={settings.FACE_DETECTION_MODEL}, Threshold={settings.RECOGNITION_THRESHOLD}")

    def run_training(self):
        if messagebox.askyesno("Confirm", "This will scan the TrainingImages folder and update the database. Continue?"):
            try:
                from apps.training_app import run_training_gui
                # Note: training_app uses standard tkinter, so we pass self (which is a CTk window)
                # It should work fine as Toplevel
                run_training_gui(self)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start training: {e}")

    def run_image_mode(self):
        try:
            from apps.image_app import run_image_app
            run_image_app(self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start image app: {e}")

    def run_video_mode(self):
        try:
            from apps.video_app import run_video_app
            run_video_app(self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start video app: {e}")

    def on_closing(self):
        print("Closing database connections...")
        Database.close_all()
        self.destroy()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
