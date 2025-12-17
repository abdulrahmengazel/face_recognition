import os
import cv2
import numpy as np
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# إضافة مسار المشروع
sys.path.append(os.path.abspath('.'))

from src.database import Database
import src.config as config
from src.face_detector import detect_faces
from deepface import DeepFace
import face_recognition

# إعدادات العرض
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# مجلد الاختبار
TEST_DIR = "TestImages"

# --- 1. FONKSİYONLAR: TESPİT (DETECTION) ---

def run_detection_benchmark():
    """HOG, CNN ve YOLO modellerini hız ve başarı oranı açısından karşılaştırır."""
    print("\n" + "="*40)
    print("AŞAMA 1: YÜZ TESPİT MODELLERİ KARŞILAŞTIRMASI")
    print("="*40)
    
    models = ["hog", "cnn", "yolo"]
    results = []
    raw_times_data = [] # Box plot için ham veriler
    
    # Tüm resimleri önceden listele
    all_images = []
    if os.path.exists(TEST_DIR):
        for root, dirs, files in os.walk(TEST_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(root, file))
    
    if not all_images:
        print("Hata: Test görseli bulunamadı!")
        return None, None

    for model in models:
        print(f"Testing Model: {model.upper()}...")
        times = []
        success_count = 0
        
        for img_path in all_images:
            image = cv2.imread(img_path)
            if image is None: continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            start_time = time.perf_counter()
            
            try:
                locations = detect_faces(
                    rgb_image, 
                    model_name=model, 
                    confidence=config.YOLO_CONFIDENCE, 
                    yolo_weights=config.YOLO_WEIGHTS
                )
                duration = time.perf_counter() - start_time
                times.append(duration)
                
                # Box plot verisi için ekle
                raw_times_data.append({"Model": model.upper(), "Süre (sn)": duration})
                
                if len(locations) > 0:
                    success_count += 1
            except Exception as e:
                print(f"  Hata ({model}): {e}")
        
        avg_time = np.mean(times) if times else 0
        success_rate = (success_count / len(all_images)) * 100 if all_images else 0
        
        results.append({
            "Model": model.upper(),
            "Ort. Süre (sn)": avg_time,
            "Başarı Oranı (%)": success_rate,
            "Toplam Resim": len(all_images),
            "Bulunan Yüz": success_count
        })
        print(f"  > {model.upper()}: {avg_time:.4f}s | Başarı: {success_rate:.1f}%")

    return pd.DataFrame(results), pd.DataFrame(raw_times_data)

# --- 2. FONKSİYONLAR: TANIMA (RECOGNITION) ---

def get_encoding(image_rgb, model_name):
    locations = detect_faces(
        image_rgb, 
        config.FACE_DETECTION_MODEL, 
        confidence=config.YOLO_CONFIDENCE, 
        yolo_weights=config.YOLO_WEIGHTS
    )
    
    if not locations:
        return None, 0

    top, right, bottom, left = locations[0]
    encoding = None
    
    start_time = time.perf_counter()

    if model_name == "dlib":
        encs = face_recognition.face_encodings(image_rgb, [locations[0]])
        if encs: encoding = encs[0]
    
    elif model_name == "facenet":
        face_img = image_rgb[top:bottom, left:right]
        if face_img.size > 0:
            try:
                embedding_objs = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)
                if embedding_objs:
                    encoding = np.array(embedding_objs[0]['embedding'])
            except:
                pass
    
    end_time = time.perf_counter()
    return encoding, end_time - start_time

def find_nearest_face(encoding, cursor, model_name):
    if encoding is None: return "BİLİNMİYOR"
    
    vec_str = str(encoding.tolist()) if hasattr(encoding, 'tolist') else str(encoding)
    
    if model_name == "dlib":
        column_name = "encoding"
        distance_operator = "<->"
        threshold = 0.6
    else:
        column_name = "encoding_facenet"
        distance_operator = "<=>"
        threshold = 0.4

    try:
        query = f"""
            SELECT p.name, f.{column_name} {distance_operator} %s AS distance
            FROM people p JOIN face_encodings f ON p.id = f.person_id
            WHERE f.{column_name} IS NOT NULL
            ORDER BY distance ASC
            LIMIT 1;
        """
        cursor.execute(query, (vec_str,))
        result = cursor.fetchone()
        
        if result:
            name, distance = result
            if distance < threshold:
                return name
        return "BİLİNMİYOR"
    except:
        return "HATA"

def run_recognition_benchmark(model_name, cursor):
    print(f"\nTest Ediliyor: {model_name.upper()}...")
    
    y_true = []
    y_pred = []
    processing_times = []
    
    people_dirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    for true_name in people_dirs:
        person_path = os.path.join(TEST_DIR, true_name)
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in images:
            img_path = os.path.join(person_path, img_file)
            image = cv2.imread(img_path)
            if image is None: continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            encoding, duration = get_encoding(rgb_image, model_name)
            processing_times.append(duration)
            
            predicted_name = find_nearest_face(encoding, cursor, model_name)
            
            y_true.append(true_name)
            y_pred.append(predicted_name)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    avg_time = np.mean(processing_times) if processing_times else 0
    
    return {
        "Model": model_name.upper(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Ort. Süre (sn)": avg_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "raw_times": processing_times # Box plot için
    }

# --- MAIN ---

def main():
    # 1. Detection Benchmark
    det_df, det_raw_times = run_detection_benchmark()
    
    # 2. Recognition Benchmark
    print("\n" + "="*40)
    print("AŞAMA 2: YÜZ TANIMA MODELLERİ KARŞILAŞTIRMASI")
    print("="*40)
    
    Database.initialize_pool()
    rec_results = []
    rec_raw_times = []
    confusion_data = {}

    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            res_dlib = run_recognition_benchmark("dlib", cursor)
            if res_dlib: 
                confusion_data["DLIB"] = (res_dlib.pop("y_true"), res_dlib.pop("y_pred"))
                times = res_dlib.pop("raw_times")
                for t in times: rec_raw_times.append({"Model": "DLIB", "Süre (sn)": t})
                rec_results.append(res_dlib)
            
            res_facenet = run_recognition_benchmark("facenet", cursor)
            if res_facenet: 
                confusion_data["FACENET"] = (res_facenet.pop("y_true"), res_facenet.pop("y_pred"))
                times = res_facenet.pop("raw_times")
                for t in times: rec_raw_times.append({"Model": "FACENET", "Süre (sn)": t})
                rec_results.append(res_facenet)

    Database.close_all()
    rec_df = pd.DataFrame(rec_results)
    rec_times_df = pd.DataFrame(rec_raw_times)

    # --- VISUALIZATION ---
    
    # --- GRAFİK 1: TESPİT HIZI DAĞILIMI (BOX PLOT) ---
    if det_raw_times is not None and not det_raw_times.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y='Süre (sn)', data=det_raw_times, palette='Set3')
        plt.title('Yüz Tespit Hızı Kararlılığı (Box Plot)', fontsize=14)
        plt.ylabel('Süre (Saniye)')
        plt.show()

    # --- GRAFİK 2: TESPİT BAŞARISI (PIE CHARTS) ---
    if det_df is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Yüz Tespit Başarı Oranları', fontsize=16)
        
        for i, (idx, row) in enumerate(det_df.iterrows()):
            labels = ['Bulundu', 'Bulunamadı']
            sizes = [row['Bulunan Yüz'], row['Toplam Resim'] - row['Bulunan Yüz']]
            colors = ['#66b3ff', '#ff9999']
            axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[i].set_title(row['Model'])
        plt.show()

    # --- GRAFİK 3: TANIMA PERFORMANSI (DETAYLI) ---
    if not rec_df.empty:
        # Veriyi "melt" ederek seaborn için uygun hale getir
        melted_df = rec_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], var_name="Metrik", value_name="Değer")
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Metrik", y="Değer", hue="Model", data=melted_df, palette="viridis")
        plt.title("Yüz Tanıma Performans Metrikleri (Detaylı)", fontsize=14)
        plt.ylim(0, 1.1)
        plt.legend(loc='lower right')
        plt.show()

        # --- GRAFİK 4: TANIMA HIZI DAĞILIMI (BOX PLOT) ---
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Süre (sn)', data=rec_times_df, palette='Pastel1')
        plt.title('Yüz Kodlama (Encoding) Hızı Kararlılığı', fontsize=14)
        plt.ylabel('Süre (Saniye)')
        plt.show()

        # --- GRAFİK 5: KARMAŞIKLIK MATRİSLERİ ---
        for model_name, (y_true, y_pred) in confusion_data.items():
            plt.figure(figsize=(8, 6))
            unique_labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
            plt.title(f'{model_name} - Karmaşıklık Matrisi')
            plt.xlabel('Tahmin Edilen')
            plt.ylabel('Gerçek')
            plt.show()

if __name__ == "__main__":
    main()
