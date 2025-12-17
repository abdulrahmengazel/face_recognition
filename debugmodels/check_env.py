import sys
import importlib.metadata

def check_environment():
    print("--- Environment Check ---")
    
    # 1. Check Python Version
    py_version = sys.version_info
    print(f"Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("❌ CRITICAL: Python version is too old. DeepFace requires Python 3.8+.")
        print("   Please install Python 3.10 or 3.11 and recreate your virtual environment.")
    else:
        print("✅ Python version is compatible.")

    print("-" * 30)

    # 2. Check Packages
    required_packages = {
        "opencv-python": "Any",
        "face_recognition": "Any",
        "psycopg2-binary": "Any",
        "ultralytics": "8.0.0+",
        "deepface": "Any",
        "numpy": "1.22+"
    }

    for package, min_version in required_packages.items():
        try:
            version = importlib.metadata.version(package)
            print(f"✅ {package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ {package}: NOT INSTALLED")

    print("-" * 30)
    print("Check complete.")

if __name__ == "__main__":
    check_environment()
