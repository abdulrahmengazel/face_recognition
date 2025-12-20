# Face Recognition System (Smart School API & GUI)

An end‑to‑end face recognition system with both a Tkinter GUI for desktop use and a FastAPI backend for mobile/web integration. It supports bulk training from a folder of images and multiple recognition modes: single image, live video (webcam), and REST API.

## Features

- GUI main menu to configure and run the system
- FastAPI REST API for integration with mobile apps and web platforms
- Detection models: `hog`, `cnn`, and `yolo`
- Encoding models: `dlib` and `facenet`
- Selectable YOLO variants (weights placed under the `assets/yolo/` folder)
- Adjustable recognition threshold, processing scale, and training image size
- Bulk training from `data/TrainingImages/` (folder-per-person layout)
- PostgreSQL database with `pgvector` for fast similarity search

## Quick Start

1) Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) (Optional) Validate your environment

```powershell
python benchmarks\check_gpu.py   # Checks GPU utilization
```

4) Prepare training data

Place images in `data/TrainingImages/` using a folder-per-person structure:

```
data/TrainingImages/
  Alice/
    img1.jpg
    img2.jpg
  Bob/
    img1.jpg
```

5) Run the Desktop App

```powershell
python main.py
```

In the GUI:
- Choose encoding and detection models
- Adjust threshold/scale/training size as needed
- Click “Run Bulk Training” once your `TrainingImages/` are ready
- Use “Image Recognition” to test on a still image
- Use “Live Video Recognition” to run via webcam

6) Run the API Server

```powershell
python api.py
```
The API will be available at `http://localhost:8000`. You can access the auto-generated documentation at `http://localhost:8000/docs`.

## Project Layout

```
PythonProject/
  main.py                 # Tkinter main menu (Entry point)
  api.py                  # FastAPI server for mobile/web integration
  requirements.txt        # Project dependencies
  PROJECT_STRUCTURE.md    # Detailed folder/file descriptions
  apps/                   # GUI application logic (Training, Image, Video)
  benchmarks/             # Performance testing and diagnostic scripts
  config/                 # Global settings and configurations
  core/                   # Core logic (Database, Face Detector)
  data/                   # User data (Training and Test images)
  assets/                 # Static assets (YOLO weights)
```

## YOLO Weights

- Put your YOLO face detection `.pt` weights in the `assets/yolo/` directory.
- The default weights are configured in `config/settings.py`.

## Notes & Tips

- Python version: 3.9–3.11 is recommended.
- Database: Requires PostgreSQL with the `pgvector` extension.
- GPU: Ensure CUDA/cuDNN are installed for GPU acceleration with YOLO or CNN models.
- Threshold: Adjust the threshold in `config/settings.py` or via the GUI if you get too many false positives or misses.

## Troubleshooting

- DLL load failures (Windows): Ensure Visual C++ Redistributable is installed.
- Database Connection: Verify PostgreSQL credentials in `config/settings.py`.
- Webcam not opening: Close other apps using the camera.

## License

This project’s license was not specified. Add your preferred license here.
