import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import face_recognition
from deepface import DeepFace

# Add project root to path to import settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.settings as settings
from core.detector import detect_faces


def get_encoding(image_rgb):
    """Extracts encoding based on current settings."""
    locs = detect_faces(image_rgb, settings.FACE_DETECTION_MODEL, confidence=settings.YOLO_CONFIDENCE,
                        yolo_weights=settings.YOLO_WEIGHTS)

    if not locs:
        return None

    top, right, bottom, left = locs[0]

    if settings.ENCODING_MODEL == "dlib":
        # Dlib Encoding
        dlib_encs = face_recognition.face_encodings(image_rgb, [locs[0]],
                                                    num_jitters=1)  # Keep jitter low for benchmark speed
        if dlib_encs:
            return dlib_encs[0]

    elif settings.ENCODING_MODEL == "facenet":
        # FaceNet Encoding
        face_img = image_rgb[top:bottom, left:right]
        if face_img.shape[0] > 20 and face_img.shape[1] > 20:
            try:
                embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                if embedding_objs:
                    return np.array(embedding_objs[0]['embedding'])
            except:
                pass
    return None


def load_dataset(data_dir):
    print(f"Loading dataset from {data_dir}...")
    print(f"Using Model: {settings.ENCODING_MODEL.upper()} + {settings.FACE_DETECTION_MODEL.upper()}")

    X = []
    y = []

    people = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for person_name in people:
        person_path = os.path.join(data_dir, person_name)
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Processing {person_name}: {len(images)} images found.")

        for img_name in images:
            img_path = os.path.join(person_path, img_name)
            try:
                # Read and resize
                img = cv2.imread(img_path)
                if img is None: continue

                # Resize for consistency
                height, width = img.shape[:2]
                scale = min(800 / width, 800 / height)
                new_w, new_h = int(width * scale), int(height * scale)
                img = cv2.resize(img, (new_w, new_h))

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get Encoding
                encoding = get_encoding(rgb)

                if encoding is not None:
                    X.append(encoding)
                    y.append(person_name)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    return np.array(X), np.array(y)


def main():
    data_dir = os.path.join(settings.PROJECT_ROOT, "data", "TrainingImages")

    # 1. Load Data
    X, y = load_dataset(data_dir)

    if len(X) < 10:
        print("\n[ERROR] Not enough data to perform split. You need at least 10-20 images total.")
        return

    print(f"\nTotal Samples: {len(X)}")
    print(f"Classes: {np.unique(y)}")

    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. Split Data (70% Train, 15% Val, 15% Test)
    # First split: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded,
                                                        random_state=42)

    # Second split: Split the 30% Temp into 50% Val (15% total) and 50% Test (15% total)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"\n--- Data Split ---")
    print(f"Training Set:   {len(X_train)} samples")
    print(f"Validation Set: {len(X_val)} samples")
    print(f"Testing Set:    {len(X_test)} samples")

    # 4. Train Classifier
    print("\nTraining MLP Classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        solver='adam',
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 5. Evaluate on Validation Set (Optional - usually for tuning)
    val_acc = clf.score(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.2f}")

    # 6. Evaluate on Test Set (Final Metrics)
    print("\n--- TEST SET RESULTS ---")
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Final Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Confusion Matrix\nModel: {settings.ENCODING_MODEL} | Acc: {acc:.2f}")
    plt.show()


if __name__ == "__main__":
    main()
