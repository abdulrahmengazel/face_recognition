import cv2
import face_recognition
import time
import numpy as np

def benchmark_cnn(source=0, num_frames=50, scale=0.5):
    print(f"--- Starting CNN Benchmark ---")
    print(f"Source: Camera {source}")
    print(f"Frames to process: {num_frames}")
    print(f"Scale: {scale}")
    print("-" * 30)

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

    # Warmup
    print("Warming up camera...")
    for _ in range(10):
        cap.read()

    print("Benchmarking started...")
    start_time = time.time()
    processed_frames = 1
    
    detection_times = []

    try:
        while processed_frames < num_frames:
            ret, frame = cap.read()
            if not ret: break

            frame_start = time.time()

            # 1. Resize
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            # 2. Convert Color
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 3. Detect Faces (CNN)
            # number_of_times_to_upsample=0 is critical for speed
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn", number_of_times_to_upsample=0)
            
            # 4. Encode Faces (Optional - uncomment to test full pipeline)
            # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            frame_end = time.time()
            detection_times.append(frame_end - frame_start)
            
            processed_frames += 1
            print(f"Frame {processed_frames}/{num_frames}: Found {len(face_locations)} faces. Time: {(frame_end - frame_start):.4f}s")

    except KeyboardInterrupt:
        print("Benchmark interrupted.")
    finally:
        cap.release()

    total_time = time.time() - start_time
    avg_time = np.mean(detection_times)
    avg_fps = 1.0 / avg_time

    print("-" * 30)
    print(f"--- Results ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Processed Frames: {processed_frames}")
    print(f"Average Processing Time per Frame: {avg_time:.4f}s")
    print(f"Estimated FPS (Processing Only): {avg_fps:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    # Test with Scale 0.25 (Fastest)
    benchmark_cnn(scale=1.00)
    
    # Uncomment to test other scales
    # benchmark_cnn(scale=0.5)
    # benchmark_cnn(scale=1.0)
