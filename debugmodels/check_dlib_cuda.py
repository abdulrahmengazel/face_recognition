# python
# File: `debugmodels/check_dlib_cuda.py`
# Prints actionable instructions if dlib is missing and safely checks CUDA support.

import sys

try:
    import dlib
except ModuleNotFoundError:
    print("Error: dlib is not installed in the current Python environment.")
    print(r"Activate your virtualenv in PyCharm terminal (PowerShell):")
    print(r"  .\venv\Scripts\Activate.ps1")
    print("Or (cmd.exe):")
    print(r"  venv\Scripts\activate")
    print("\nThen install prerequisites and dlib:")
    print("  python -m pip install --upgrade pip setuptools wheel cmake")
    print("  pip install dlib            # try prebuilt wheel first")
    print("If wheel is unavailable, build from source in dlib repo:")
    print("  cd path\\to\\dlib")
    print("  pip install .               # requires CMake + MSVC (and CUDA if desired)")
    print("\nIf you need CUDA support, ensure CUDA toolkit (nvcc) is on PATH and build with CMake option DLIB_USE_CUDA=ON.")
    sys.exit(1)

try:
    print(f"Dlib Version: {dlib.__version__}")
    cuda_flag = getattr(dlib, "DLIB_USE_CUDA", False)
    print("✅ Dlib compiled with CUDA support." if cuda_flag else "❌ Dlib NOT compiled with CUDA support (CPU only).")

    if hasattr(dlib, "cuda") and hasattr(dlib.cuda, "get_num_devices"):
        num_devices = dlib.cuda.get_num_devices()
        print(f"Number of GPU devices detected: {num_devices}")
        if num_devices > 0 and hasattr(dlib.cuda, "get_device_name"):
            print(f"Device Name: {dlib.cuda.get_device_name(0)}")
    else:
        print("dlib.cuda API not available in this build.")

except Exception as e:
    print(f"Error checking CUDA: {e}")
    sys.exit(1)
