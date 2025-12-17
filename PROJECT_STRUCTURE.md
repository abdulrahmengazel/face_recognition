# 📂 Project Structure Documentation

This document explains the structure of the Face Recognition System project, detailing the purpose of each folder and key file.

---

### 🌳 Root Directory

- **`main.py`**: The main entry point of the application. It launches the main GUI window and handles user interactions.
- **`requirements.txt`**: Lists all the Python libraries required to run the project.
- **`PROJECT_STRUCTURE.md`**: This file.

---

### ⚙️ `config/`

This package contains all global settings and configurations.

- **`settings.py`**: A central file for all adjustable parameters, such as model paths, recognition thresholds, database credentials, and performance tuning options.

---

### 🧠 `core/`

This package contains the fundamental logic (the "brain") of the system.

- **`database.py`**: Manages all interactions with the PostgreSQL database, including connection pooling, table creation, and `pgvector` index setup.
- **`detector.py`**: A unified layer for face detection. It contains the logic to load and run different detection models like HOG, CNN, and YOLO.

---

### 🖥️ `apps/`

This package contains the user-facing applications and GUI components.

- **`training_app.py`**: Handles the logic for "training" (enrolling faces). It reads images, extracts embeddings for both dlib and FaceNet, and saves them to the database. Includes a progress bar GUI.
- **`image_app.py`**: The application for recognizing faces in a static image file.
- **`video_app.py`**: The application for real-time face recognition using a live camera feed.

---

### 📊 `benchmarks/`

This package contains all scripts related to testing, evaluation, and performance analysis.

- **`suite.py`**: A comprehensive benchmark suite that runs static tests (comparing detection and recognition models) and live tests (real-time camera analysis). It generates reports and plots.
- **`detection_only.py`**: A focused script to test and report the performance (speed, success rate) of a single, user-selected detection model.
- **`live_test.py`**: A simple GUI application to visually test and compare detection models in real-time, showing bounding boxes, confidence scores, and FPS.
- **`check_gpu.py`**: A diagnostic tool to check if the deep learning libraries (PyTorch, TensorFlow) are utilizing the GPU.

---

### 🖼️ `assets/`

This folder stores static asset files required by the project.

- **`yolo/`**: Contains the pre-trained `.pt` model files for the YOLO face detector.

---

### 🗃️ `data/`

This folder is used for all user-provided data.

- **`TrainingImages/`**: Contains subfolders for each person, with images used to train the system.
- **`TestImages/`--**: Contains subfolders for each person, with new images used for evaluating the system's accuracy.
