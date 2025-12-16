import face_recognition
import cv2
import os
import dlib
import time

def test_cnn():
    print("--- CNN Diagnostic Tool ---")
    
    # 1. Check CUDA
    print(f"Dlib Version: {dlib.__version__}")
    print(f"CUDA Enabled: {dlib.DLIB_USE_CUDA}")
    print(f"GPUs: {dlib.cuda.get_num_devices()}")
    
    # 2. Find a test image
    training_dir = "TrainingImages"
    test_image_path = None
    
    if os.path.exists(training_dir):
        for root, dirs, files in os.walk(training_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image_path = os.path.join(root, file)
                    break
            if test_image_path: break
    
    if not test_image_path:
        print("❌ Error: No images found in TrainingImages to test with.")
        return

    print(f"\nTesting with image: {test_image_path}")
    
    try:
        # Load image
        image = face_recognition.load_image_file(test_image_path)
        
        # Test HOG (Baseline)
        print("\n1. Testing HOG (CPU)...")
        start = time.time()
        hog_faces = face_recognition.face_locations(image, model="hog")
        print(f"   - Time: {time.time() - start:.4f}s")
        print(f"   - Faces found: {len(hog_faces)}")
        
        # Test CNN (GPU)
        print("\n2. Testing CNN (GPU)...")
        start = time.time()
        # IMPORTANT: We set number_of_times_to_upsample=0 to save VRAM
        cnn_faces = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=0)
        print(f"   - Time: {time.time() - start:.4f}s")
        print(f"   - Faces found: {len(cnn_faces)}")
        
        if len(cnn_faces) == 0:
            print("\n⚠️ Warning: CNN found 0 faces.")
            print("Possible causes:")
            print("1. Image is too large for VRAM (Try resizing image).")
            print("2. Face is too small (Upsample=0 might miss small faces).")
        else:
            print("\n✅ CNN is working correctly!")

    except Exception as e:
        print(f"\n❌ Critical Error: {e}")

if __name__ == "__main__":
    test_cnn()
