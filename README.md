# Face Recognition System (Tkinter GUI)

An end‑to‑end face recognition desktop application with a simple Tkinter GUI. It supports bulk training from a folder of images and two recognition modes: single image and live video (webcam). You can switch between multiple detection and encoding backends at runtime.

## Features

- GUI main menu to configure and run the system
- Detection models: `hog`, `cnn`, and `yolo`
- Encoding models: `dlib` and `facenet`
- Selectable YOLO variants (weights placed under the `yolo/` folder)
- Adjustable recognition threshold, processing scale, and training image size
- Bulk training from `TrainingImages/` (folder-per-person layout)

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
python check_env.py
python debugmodels\check_dlib_cuda.py   # Checks dlib/CUDA availability
```

4) Prepare training data

Place images in `TrainingImages/` using a folder-per-person structure:

```
TrainingImages/
  Alice/
    img1.jpg
    img2.jpg
  Bob/
    img1.jpg
```

5) Run the app

```powershell
python main.py
```

In the GUI:
- Choose encoding and detection models (YOLO shows an extra selector)
- Adjust threshold/scale/training size as needed
- Click “Run Bulk Training” once your `TrainingImages/` are ready
- Use “Image Recognition” to test on a still image
- Use “Live Video Recognition” to run via webcam

## YOLO Weights

- Put your YOLO face detection `.pt` weights in the `yolo/` directory.
- The available presets and defaults are configured in `src/config.py` via `YOLO_MODELS` and `YOLO_WEIGHTS`.
- In the GUI, pick the desired YOLO model name; the app updates the selected weights on the fly.

## Project Layout

```
PythonProject/
  main.py                 # Tkinter main menu
  requirements.txt
  check_env.py            # Environment sanity checks
  debugmodels/
    check_dlib_cuda.py    # dlib/CUDA diagnostic
  src/
    config.py             # Global settings (updated by GUI)
    train.py              # Bulk training pipeline
    image_app.py          # Image recognition flow
    video_app.py          # Live video recognition flow (webcam)
    face_detector.py      # Detection backends (HOG/CNN/YOLO)
    database.py           # Persistence/connection pooling utilities
  TrainingImages/         # Your training dataset (folder-per-person)
  yolo/                   # YOLO weights (.pt)
```

## Notes & Tips

- Python version: For best compatibility with `dlib`/`face_recognition`, use a commonly supported Python (e.g., 3.9–3.11).
- GPU: If you plan to use GPU acceleration (dlib or PyTorch/YOLO), make sure the correct CUDA/cuDNN versions are installed and match your installed wheels.
- First run: Perform “Run Bulk Training” after placing images — recognition quality depends on your dataset.
- Threshold: If you see many false positives, reduce the recognition threshold; if too many misses, increase it slightly.
- Processing scale: Lowering scale speeds up video but may reduce accuracy.

## Troubleshooting

- DLL load failures (Windows): Ensure you’re using wheels compatible with your Python version and that Visual C++ Redistributable is installed.
- dlib not found / no CUDA: Use the provided diagnostic scripts and consider CPU-only if CUDA isn’t available.
- Webcam not opening: Close other apps using the camera; try a different index (if supported in `video_app.py`).
- YOLO weight missing: Confirm the selected weight file exists under `yolo/` and that `src/config.py` maps the chosen model name correctly.

## License

This project’s license was not specified. Add your preferred license here.
