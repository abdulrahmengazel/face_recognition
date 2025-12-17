import customtkinter as ctk
from tkinter import messagebox, simpledialog
import sys
import os
import cv2
import threading
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import lightweight modules
import config.settings as settings
from core.database import Database

# --- COLOR PALETTE ---
COLORS = {
    "bg": "#344e41",        # Darkest Green (Main Background)
    "frame": "#3a5a40",     # Hunter Green (Card Background)
    "button": "#588157",    # Fern Green (Primary Action)
    "hover": "#a3b18a",     # Sage Green (Hover State)
    "text": "#dad7cd",      # Light Beige (Text Color)
    "accent": "#dad7cd"     # Accent Color
}

# --- THEME SETTINGS ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green") 

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition System")
        self.geometry("700x850") # Increased height for new button
        self.configure(fg_color=COLORS["bg"]) 
        
        Database.initialize_pool()
        
        # Fonts
        self.header_font = ctk.CTkFont(family="Roboto", size=24, weight="bold")
        self.sub_font = ctk.CTkFont(family="Roboto", size=16, weight="bold")
        self.text_font = ctk.CTkFont(family="Roboto", size=14)
        
        # --- UI Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_scroll.grid_columnconfigure(0, weight=1)

        self.create_header()
        self.create_settings_card()
        self.create_actions_card()
        self.create_footer()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.toggle_yolo_widget()

    def create_header(self):
        header_frame = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        header_frame.grid(row=0, column=0, pady=(30, 20), sticky="ew")
        
        title = ctk.CTkLabel(header_frame, text="Face Recognition System", font=self.header_font, text_color=COLORS["text"])
        title.pack()
        
        subtitle = ctk.CTkLabel(header_frame, text="Advanced AI-Powered Security", font=ctk.CTkFont(family="Roboto", size=12), text_color=COLORS["hover"])
        subtitle.pack()

    def create_settings_card(self):
        card = ctk.CTkFrame(self.main_scroll, fg_color=COLORS["frame"], corner_radius=15)
        card.grid(row=1, column=0, padx=30, pady=10, sticky="ew")
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card, text="⚙️ System Configuration", font=self.sub_font, text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2, padx=20, pady=(15, 15), sticky="w")

        # Encoding Model
        ctk.CTkLabel(card, text="Encoding Model:", font=self.text_font, text_color=COLORS["text"]).grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.encoding_model_var = ctk.StringVar(value=settings.ENCODING_MODEL)
        enc_combo = ctk.CTkComboBox(card, variable=self.encoding_model_var, values=["dlib", "facenet"], 
                                    fg_color=COLORS["bg"], button_color=COLORS["button"], button_hover_color=COLORS["hover"],
                                    text_color=COLORS["text"], dropdown_fg_color=COLORS["frame"], command=self.update_settings)
        enc_combo.grid(row=1, column=1, padx=20, pady=10, sticky="ew")

        # Detection Model
        ctk.CTkLabel(card, text="Detection Model:", font=self.text_font, text_color=COLORS["text"]).grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.model_var = ctk.StringVar(value=settings.FACE_DETECTION_MODEL)
        det_combo = ctk.CTkComboBox(card, variable=self.model_var, values=["cnn", "hog", "yolo"], 
                                    fg_color=COLORS["bg"], button_color=COLORS["button"], button_hover_color=COLORS["hover"],
                                    text_color=COLORS["text"], dropdown_fg_color=COLORS["frame"], command=self.update_settings)
        det_combo.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

        # YOLO Version
        self.yolo_label = ctk.CTkLabel(card, text="YOLO Version:", font=self.text_font, text_color=COLORS["text"])
        self.yolo_model_var = ctk.StringVar(value="YOLOv8 Medium")
        self.yolo_combo = ctk.CTkComboBox(card, variable=self.yolo_model_var, values=list(settings.YOLO_MODELS.keys()), 
                                          fg_color=COLORS["bg"], button_color=COLORS["button"], button_hover_color=COLORS["hover"],
                                          text_color=COLORS["text"], dropdown_fg_color=COLORS["frame"], command=self.update_settings)

        # Threshold
        ctk.CTkLabel(card, text="Sensitivity:", font=self.text_font, text_color=COLORS["text"]).grid(row=4, column=0, padx=20, pady=10, sticky="w")
        self.threshold_slider = ctk.CTkSlider(card, from_=0.1, to=1.5, number_of_steps=14, command=self.update_threshold_label,
                                              button_color=COLORS["button"], button_hover_color=COLORS["hover"], progress_color=COLORS["button"])
        self.threshold_slider.set(settings.RECOGNITION_THRESHOLD)
        self.threshold_slider.grid(row=4, column=1, padx=(20, 5), pady=10, sticky="ew")
        self.threshold_label = ctk.CTkLabel(card, text=f"{settings.RECOGNITION_THRESHOLD:.2f}", text_color=COLORS["text"], width=30)
        self.threshold_label.grid(row=4, column=2, padx=(0, 20), pady=10)

        # Video Scale
        ctk.CTkLabel(card, text="Performance:", font=self.text_font, text_color=COLORS["text"]).grid(row=5, column=0, padx=20, pady=10, sticky="w")
        self.scale_slider = ctk.CTkSlider(card, from_=0.1, to=1.0, number_of_steps=9, command=self.update_scale_label,
                                          button_color=COLORS["button"], button_hover_color=COLORS["hover"], progress_color=COLORS["button"])
        self.scale_slider.set(settings.PROCESSING_SCALE)
        self.scale_slider.grid(row=5, column=1, padx=(20, 5), pady=10, sticky="ew")
        self.scale_label = ctk.CTkLabel(card, text=f"{settings.PROCESSING_SCALE:.2f}", text_color=COLORS["text"], width=30)
        self.scale_label.grid(row=5, column=2, padx=(0, 20), pady=10)

        apply_btn = ctk.CTkButton(card, text="Apply Changes", command=self.update_settings, 
                                  fg_color=COLORS["bg"], hover_color=COLORS["hover"], text_color=COLORS["text"], border_width=1, border_color=COLORS["text"])
        apply_btn.grid(row=6, column=0, columnspan=3, padx=20, pady=20, sticky="ew")

    def create_actions_card(self):
        card = ctk.CTkFrame(self.main_scroll, fg_color=COLORS["frame"], corner_radius=15)
        card.grid(row=2, column=0, padx=30, pady=10, sticky="ew")
        card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(card, text="🚀 Operations", font=self.sub_font, text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2, padx=20, pady=(15, 15), sticky="w")

        # Add New Person Button (NEW)
        add_btn = ctk.CTkButton(card, text="➕ Add New Person", command=self.add_new_person, height=50,
                                fg_color=COLORS["button"], hover_color=COLORS["hover"], text_color=COLORS["text"], font=self.sub_font)
        add_btn.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

        # Training Button
        train_btn = ctk.CTkButton(card, text="Run Bulk Training", command=self.run_training, height=50,
                                  fg_color=COLORS["bg"], hover_color=COLORS["hover"], text_color=COLORS["text"], font=self.sub_font, border_width=1, border_color=COLORS["button"])
        train_btn.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

        # Image App Button
        img_btn = ctk.CTkButton(card, text="Analyze Image", command=self.run_image_mode, height=50,
                                fg_color=COLORS["bg"], hover_color=COLORS["hover"], text_color=COLORS["text"], font=self.sub_font)
        img_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Video App Button
        vid_btn = ctk.CTkButton(card, text="Live Camera", command=self.run_video_mode, height=50,
                                fg_color=COLORS["bg"], hover_color=COLORS["hover"], text_color=COLORS["text"], font=self.sub_font)
        vid_btn.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

    def create_footer(self):
        footer_frame = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        footer_frame.grid(row=3, column=0, pady=20, sticky="ew")
        
        quit_btn = ctk.CTkButton(footer_frame, text="Exit Application", command=self.on_closing, 
                                 fg_color="#8B0000", hover_color="#A52A2A", text_color="white")
        quit_btn.pack(ipadx=20)

    # --- Logic Functions ---
    def toggle_yolo_widget(self):
        if self.model_var.get() == "yolo":
            self.yolo_label.grid(row=3, column=0, padx=20, pady=10, sticky="w")
            self.yolo_combo.grid(row=3, column=1, padx=20, pady=10, sticky="ew")
        else:
            self.yolo_label.grid_forget()
            self.yolo_combo.grid_forget()

    def update_threshold_label(self, value):
        self.threshold_label.configure(text=f"{value:.2f}")

    def update_scale_label(self, value):
        self.scale_label.configure(text=f"{value:.2f}")

    def update_settings(self, event=None):
        settings.ENCODING_MODEL = self.encoding_model_var.get()
        settings.FACE_DETECTION_MODEL = self.model_var.get()
        selected_yolo_name = self.yolo_model_var.get()
        settings.YOLO_WEIGHTS = settings.YOLO_MODELS.get(selected_yolo_name, "assets/yolo/yolov8m-face.pt")
        settings.RECOGNITION_THRESHOLD = round(self.threshold_slider.get(), 2)
        settings.PROCESSING_SCALE = round(self.scale_slider.get(), 2)
        self.toggle_yolo_widget()
        print(f"Settings Updated: {settings.ENCODING_MODEL} | {settings.FACE_DETECTION_MODEL}")

    def add_new_person(self):
        # Ask for name
        name = ctk.CTkInputDialog(text="Enter Name:", title="New Person").get_input()
        if not name: return
        
        # Create directory
        person_dir = os.path.join("data/TrainingImages", name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Open Camera to capture images
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return
            
        count = 0
        max_images = 10
        
        while count < max_images:
            ret, frame = cap.read()
            if not ret: break
            
            cv2.putText(frame, f"Capturing: {count+1}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture Face", frame)
            
            # Auto capture every 0.5 seconds or on key press
            if cv2.waitKey(500): # Wait 500ms
                img_name = os.path.join(person_dir, f"{name}_{count}.jpg")
                cv2.imwrite(img_name, frame)
                count += 1
                print(f"Saved {img_name}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Ask to train immediately
        if messagebox.askyesno("Success", f"Captured {max_images} images for {name}. Train now?"):
            self.run_training()

    def run_training(self):
        try:
            from apps.training_app import run_training_gui
            run_training_gui(self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def run_image_mode(self):
        try:
            from apps.image_app import run_image_app
            run_image_app(self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def run_video_mode(self):
        try:
            from apps.video_app import run_video_app
            run_video_app(self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def on_closing(self):
        Database.close_all()
        self.destroy()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
