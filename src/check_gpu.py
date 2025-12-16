import dlib
import torch
import tensorflow as tf



def check_hardware():
    print("\n" + "="*50)
    print("HARDWARE ACCELERATION DIAGNOSTIC")
    print("="*50)

    # --- 1. Dlib Check (Used for 'dlib' encoding) ---
    print(f"\n[1] Dlib (Used for 'dlib' model)")
    try:
        # Check if dlib was compiled with CUDA support
        is_cuda = dlib.DLIB_USE_CUDA
        num_devices = dlib.cuda.get_num_devices()

        print(f" - Compiled with CUDA support: {is_cuda}")
        print(f" - CUDA Devices found: {num_devices}")

        if is_cuda and num_devices > 0:
            print(" -> STATUS: ✅ RUNNING ON GPU")
        else:
            print(" -> STATUS: ⚠️ RUNNING ON CPU (Standard pip install is usually CPU-only)")
    except Exception as e:
        print(f" - Error checking dlib: {e}")

    # --- 2. PyTorch Check (Used for YOLO detection) ---
    print(f"\n[2] PyTorch / YOLO (Used for 'yolo' detection)")
    try:
        gpu_available = torch.cuda.is_available()
        print(f" - CUDA Available: {gpu_available}")

        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            print(f" - Device Name: {device_name}")
            print(" -> STATUS: ✅ RUNNING ON GPU")
        else:
            print(" -> STATUS: ⚠️ RUNNING ON CPU")
    except Exception as e:
        print(f" - Error checking PyTorch: {e}")

    # --- 3. TensorFlow Check (Used for DeepFace/FaceNet) ---
    print(f"\n[3] TensorFlow / DeepFace (Used for 'facenet' model)")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print(f" - GPUs Detected: {len(gpus)}")

        if gpus:
            for gpu in gpus:
                print(f"   * {gpu.name}")
            print(" -> STATUS: ✅ RUNNING ON GPU")
        else:
            print(" -> STATUS: ⚠️ RUNNING ON CPU")
    except Exception as e:
        print(f" - Error checking TensorFlow: {e}")

    print("\n" + "="*50)
    print("NOTE: If you have an NVIDIA card but see 'CPU',")
    print("you might need to install CUDA Toolkit and cuDNN.")
    print("="*50 + "\n")

if __name__ == "__main__":
    check_hardware()
