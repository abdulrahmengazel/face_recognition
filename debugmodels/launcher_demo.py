import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys

class NotebookLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Launcher")
        self.root.geometry("500x350")
        
        # Title
        title_label = ttk.Label(root, text="Facial Recognition System", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=20)
        
        subtitle_label = ttk.Label(root, text="Choose a module to launch:", font=("Helvetica", 10))
        subtitle_label.pack(pady=5)

        # --- Image Processing Section ---
        img_frame = ttk.LabelFrame(root, text="Image Processing (Uploaded Images)", padding=10)
        img_frame.pack(fill="x", padx=20, pady=10)

        btn_img_basic = ttk.Button(img_frame, text="Open Basic Version", 
                                   command=lambda: self.launch_notebook("notebooks/Image_Processing/Facial_Recognition_System_on_Uploaded_Images.ipynb"))
        btn_img_basic.pack(fill="x", pady=2)

        btn_img_opt = ttk.Button(img_frame, text="Open Optimized Version", 
                                 command=lambda: self.launch_notebook("notebooks/Image_Processing/Facial_Recognition_System_on_Uploaded_Images_Optimized.ipynb"))
        btn_img_opt.pack(fill="x", pady=2)

        # --- Video Processing Section ---
        vid_frame = ttk.LabelFrame(root, text="Video Processing (Live Stream)", padding=10)
        vid_frame.pack(fill="x", padx=20, pady=10)

        btn_vid_basic = ttk.Button(vid_frame, text="Open Basic Stream", 
                                   command=lambda: self.launch_notebook("notebooks/Video_Processing/stream_facial_recognition_system.ipynb"))
        btn_vid_basic.pack(fill="x", pady=2)

        btn_vid_opt = ttk.Button(vid_frame, text="Open Optimized Stream", 
                                 command=lambda: self.launch_notebook("notebooks/Video_Processing/stream_facial_recognition_system_optimized.ipynb"))
        btn_vid_opt.pack(fill="x", pady=2)

        # Quit Button
        btn_quit = ttk.Button(root, text="Exit", command=root.destroy)
        btn_quit.pack(pady=20)

    def launch_notebook(self, relative_path):
        """Launches the selected Jupyter Notebook in the default browser."""
        try:
            # Get absolute path
            base_path = os.path.dirname(os.path.abspath(__file__))
            notebook_path = os.path.join(base_path, relative_path)
            
            if not os.path.exists(notebook_path):
                messagebox.showerror("Error", f"File not found:\n{notebook_path}")
                return

            # Command to run jupyter notebook
            # We use sys.executable to ensure we use the python interpreter of the current venv
            # -m jupyter notebook: runs the notebook module
            cmd = [sys.executable, "-m", "jupyter", "notebook", notebook_path]
            
            # Run in a subprocess so it doesn't freeze the GUI
            subprocess.Popen(cmd)
            
            # Optional: Feedback to user
            # messagebox.showinfo("Launching", f"Opening {os.path.basename(relative_path)} in Jupyter...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch notebook: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NotebookLauncher(root)
    root.mainloop()
