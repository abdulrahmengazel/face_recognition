# 🤖 Akıllı Okul Yüz Tanıma Sistemi (Smart School Face Recognition System)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100.0+-009688.svg" alt="FastAPI Version">
  <img src="https://img.shields.io/badge/PostgreSQL-15+-336791.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

### 🌟 Overview (Genel Bakış)

An end-to-end, high-performance face recognition system designed for modern school management. It combines a *
*Tkinter-based Desktop GUI** for administrative tasks with a **FastAPI REST backend** for mobile and web integration.

Built with scalability in mind, the system uses **PostgreSQL** with the **pgvector** extension for ultra-fast similarity
searches, and advanced **Machine Learning Classifiers (MLP Neural Network)** for state-of-the-art recognition accuracy.

---

### 🚀 Key Features

- **🖥️ Dual Interface:** Admin Desktop App (CustomTkinter) & Mobile-friendly API (FastAPI).
- **🧠 Advanced AI Models:**
    - **Detection:** Support for HOG, CNN, and YOLOv8.
    - **Recognition:** dlib and FaceNet embeddings.
    - **Classification:** **MLP (Multi-Layer Perceptron)** for high-accuracy identification.
- **⚡ High Performance:**
    - Database-driven storage using `pgvector`.
    - Real-time **Face Tracking & Stabilization** in video streams.
    - Fast re-training capabilities (train classifier in seconds).
- **📸 Flexible Recognition:** Supports static images, live webcam streams, and batch training.
- **📊 Robust Benchmarking:** Built-in tools to evaluate model accuracy and speed.
- **⚙️ Configurable:** Easily adjustable thresholds, scaling, and training parameters via GUI.

---

### 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Backend Framework:** FastAPI, Uvicorn
- **GUI Framework:** CustomTkinter, OpenCV
- **AI/ML Libraries:**
    - `ultralytics` (YOLOv8)
    - `face_recognition` (dlib)
    - `deepface` (FaceNet)
    - `scikit-learn` (MLP Classifier)
    - `tensorflow` & `torch`
- **Database:** PostgreSQL + `pgvector`
- **Infrastructure:** CUDA/cuDNN support for GPU acceleration

---

### 📋 Requirements

- **Operating System:** Windows/Linux/MacOS
- **Python:** 3.9 or higher
- **Database:** PostgreSQL (v15+) with `pgvector` extension installed.
- **Hardware:** NVIDIA GPU recommended for optimal performance (YOLO & CNN models).
- **Other:** [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (for Windows users
  experiencing DLL errors).

---

### 📥 Setup & Installation

#### 1. Clone the Repository
```powershell
git clone <repository-url>
cd PythonProject
```

#### 2. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Database Configuration
1. Ensure PostgreSQL is running.
2. Create a database (default name: `postgres`).
3. Enable `pgvector`: `CREATE EXTENSION IF NOT EXISTS vector;`
4. Update connection details in `config/settings.py` (see [Environment Variables](#-environment-variables-config)).

#### 4. Prepare Training Data
Organize your images in `data/TrainingImages/` with one folder per person:
```text
data/TrainingImages/
├── Ali/
│   ├── img1.jpg
│   └── img2.jpg
└── Ayse/
    ├── img1.jpg
    └── img2.jpg
```

---

### 🚀 Run Commands

#### Desktop Application (GUI)
```powershell
python main.py
```

*Use the GUI to configure models, run batch training, retrain classifier, and test recognition.*

#### API Server
```powershell
python api.py
```
*The API will be available at `http://localhost:8000`. Access Swagger docs at `/docs`.*

---

### 📜 Scripts

| Script                          | Description                                                                                                |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------|
| `main.py`                       | Primary entry point for the Desktop Management GUI.                                                        |
| `api.py`                        | Primary entry point for the FastAPI REST server.                                                           |
| `benchmarks/evaluate_models.py` | Evaluates model performance (Accuracy, Precision, Recall) using a train-test split on the current dataset. |

---

### ⚙️ Environment Variables & Config

Configuration is primarily managed in `config/settings.py`. Key parameters include:

- **Database:** `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`.
- **AI Models:**
    - `ENCODING_MODEL`: "facenet" or "dlib".
    - `FACE_DETECTION_MODEL`: "yolo", "hog", or "cnn".
    - `YOLO_WEIGHTS`: Path to the selected YOLO model.
- **Classifier:**
    - `hidden_layers`, `max_iter`, `solver`, etc.
- **Thresholds:** `RECOGNITION_THRESHOLD` (default: 0.4).
- **UI Colors:** Customizable theme in `UI_COLORS`.

---

### 🧪 Tests & Benchmarking

To run the model evaluation script:
```powershell
python benchmarks/evaluate_models.py
```
This script will:
1. Load images from `data/TrainingImages/`.
2. Generate embeddings using the configured models.
3. Train the MLP Classifier.
4. Output accuracy metrics and a Confusion Matrix.

---

### 📂 Project Structure

```text
PythonProject/
├── api.py              # FastAPI server entry point
├── main.py             # Desktop GUI entry point
├── apps/               # GUI application modules (training, image, video apps)
├── assets/             # Static assets (YOLO weights, classifiers)
├── benchmarks/         # Performance evaluation scripts
├── config/             # System-wide settings & database config
├── core/               # Core logic (database handlers, face detectors)
├── data/               # Datasets for training and testing
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

### 🚧 TODO / Upcoming Improvements

- [ ] Implement a dynamic `.env` file loader for database credentials.
- [ ] Add Docker support for easy deployment of the API and Database.
- [ ] Implement more robust unit tests for core detection logic.
- [ ] Expand the API to include user management endpoints.

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.