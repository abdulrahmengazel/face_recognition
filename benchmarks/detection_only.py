import os
import cv2
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config.settings as settings
from core.detector import detect_faces

# Configuration
TEST_DIR = "../data/TestImages"

def run_detection_test():
    print("\n" + "="*50)
    print("   FACE DETECTION BENCHMARK TOOL")
    print("="*50)
    
    # 1. Select Model
    print("\nAvailable Models:")
    print("1. HOG  (Fast, CPU-friendly, less accurate)")
    print("2. CNN  (Slow, very accurate, heavy)")
    print("3. YOLO (Very Fast on GPU, very accurate)")
    
    choice = input("\nSelect model (1-3) or type name: ").strip().lower()
    
    if choice in ['1', 'hog']:
        model_name = "hog"
    elif choice in ['2', 'cnn']:
        model_name = "cnn"
    elif choice in ['3', 'yolo']:
        model_name = "yolo"
    else:
        print("Invalid choice. Defaulting to YOLO.")
        model_name = "yolo"

    print(f"\n>>> Starting Benchmark for: {model_name.upper()} <<<\n")

    # 2. Collect Images
    all_images = []
    if os.path.exists(TEST_DIR):
        for root, dirs, files in os.walk(TEST_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(root, file))
    
    if not all_images:
        print(f"Error: No images found in '{TEST_DIR}'.")
        return

    # 3. Run Test
    total_images = len(all_images)
    detected_count = 0
    times = []
    results = []

    for i, img_path in enumerate(all_images):
        filename = os.path.basename(img_path)
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            print(f"[{i+1}/{total_images}] ⚠️ Could not read {filename}")
            continue
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Measure Time
        start_time = time.perf_counter()
        
        try:
            locations = detect_faces(
                rgb_image, 
                model_name=model_name, 
                confidence=settings.YOLO_CONFIDENCE, 
                yolo_weights=settings.YOLO_WEIGHTS
            )
            duration = time.perf_counter() - start_time
            times.append(duration)
            
            num_faces = len(locations)
            is_detected = num_faces > 0
            
            if is_detected:
                detected_count += 1
                status = "✅ DETECTED"
            else:
                status = "❌ FAILED"
                
            print(f"[{i+1}/{total_images}] {filename:<30} | {status} ({num_faces} faces) | {duration:.4f}s")
            
            results.append({
                "Image": filename,
                "Detected": 1 if is_detected else 0,
                "Time": duration
            })
            
        except Exception as e:
            print(f"[{i+1}/{total_images}] Error processing {filename}: {e}")

    # 4. Final Report
    if not times: return

    avg_time = np.mean(times)
    detection_rate = (detected_count / total_images) * 100
    
    print("\n" + "="*50)
    print(f"   FINAL REPORT: {model_name.upper()}")
    print("="*50)
    print(f"Total Images Processed : {total_images}")
    print(f"Faces Detected         : {detected_count}")
    print(f"Detection Rate         : {detection_rate:.2f}%")
    print(f"Average Time per Image : {avg_time:.4f} seconds")
    print(f"Estimated FPS          : {1/avg_time:.2f} FPS")
    print("="*50)

    # 5. Visualization
    try:
        plt.figure(figsize=(10, 6))
        
        # Pie Chart for Detection Rate
        plt.subplot(1, 2, 1)
        labels = ['Detected', 'Failed']
        sizes = [detected_count, total_images - detected_count]
        colors = ['#4CAF50', '#F44336'] # Green, Red
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title(f'Detection Rate ({model_name.upper()})')

        # Histogram for Processing Time
        plt.subplot(1, 2, 2)
        sns.histplot(times, kde=True, color='skyblue')
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not show plots: {e}")

if __name__ == "__main__":
    run_detection_test()
