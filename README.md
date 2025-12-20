# 🤖 Smart School Face Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100.0+-009688.svg" alt="FastAPI Version">
  <img src="https://img.shields.io/badge/PostgreSQL-15+-336791.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

### 🌟 Overview

An end-to-end, high-performance face recognition system designed for modern school management. It combines a **Tkinter-based Desktop GUI** for administrative tasks and a **FastAPI REST backend** for seamless mobile and web integration.

Built with scalability in mind, the system leverages **PostgreSQL** with the **pgvector** extension for ultra-fast similarity searches, supporting thousands of identities with ease.

---

### 🚀 Key Features

- **🖥️ Dual Interface:** Admin Desktop App (Tkinter) & Mobile-Ready API (FastAPI).
- **🧠 Advanced AI Models:**
  - **Detection:** HOG, CNN, and YOLOv8 support.
  - **Recognition:** dlib and FaceNet embeddings.
- **⚡ High Performance:** Database-driven similarity search using `pgvector`.
- **📸 Flexible Recognition:** Supports static images, live webcam feeds, and bulk training.
- **📊 Robust Benchmarking:** Built-in tools to evaluate model accuracy and speed.
- **⚙️ Configurable:** Easily adjustable thresholds, scaling, and training parameters.

---

### 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **GUI:** Tkinter, OpenCV
- **AI/ML:** Ultralytics (YOLO), Face Recognition (dlib), DeepFace (FaceNet)
- **Database:** PostgreSQL + `pgvector`
- **Infrastructure:** CUDA/cuDNN support for GPU acceleration

---

### 📥 Quick Start

#### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Prepare Training Data
Organize your images in `data/TrainingImages/` using a folder-per-person structure:
```text
data/TrainingImages/
├── 👤 Alice/
│   ├── img1.jpg
│   └── img2.jpg
└── 👤 Bob/
    ├── img1.jpg
    └── img2.jpg
```

#### 3. Run the Desktop App
```powershell
python main.py
```
*Use the GUI to configure models, run bulk training, and test recognition.*

#### 4. Launch the API Server
```powershell
python api.py
```
*The API will be available at `http://localhost:8000`. Access docs at `/docs`.*

---

### 📂 Project Structure

```text
PythonProject/
├── 📱 api.py              # FastAPI server entry point
├── 🖥️ main.py             # Desktop GUI entry point
├── 📂 apps/               # GUI application modules
├── 📂 assets/             # Static assets (YOLO weights)
├── 📂 benchmarks/         # Performance testing scripts
├── 📂 config/             # Global configurations
├── 📂 core/               # Database & Detector logic
└── 📂 data/               # Training & Test datasets
```

---

### 💡 Notes & Tips

- **GPU Acceleration:** For YOLO and CNN models, ensure CUDA and cuDNN are correctly configured.
- **Database:** Requires a PostgreSQL instance with the `pgvector` extension installed.
- **YOLO Weights:** Place your `.pt` files in `assets/yolo/`.
- **Troubleshooting:**
  - *DLL Errors:* Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).
  - *Database:* Check connection strings in `config/settings.py`.

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
