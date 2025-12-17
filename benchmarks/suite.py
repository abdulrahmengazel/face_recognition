# -*- coding: utf-8 -*-
"""
Yüz Tanıma Sistemi Karşılaştırma Raporu (Benchmark Suite)
=========================================================

Bu dosya, yüz tanıma sisteminin performansını ölçmek için iki mod sunar:
1. Statik Test: TestImages klasöründeki resimler üzerinde toplu analiz ve grafikler.
2. Canlı Test: Kamera üzerinden gerçek zamanlı tespit ve güven skoru (confidence score) analizi.

Kullanılan Modellerin Tanımları:
--------------------------------

### Yüz Tespit Modelleri (Detection Models) ###

1.  **HOG (Histogram of Oriented Gradients):**
    *   **Tanım:** Klasik bir makine öğrenmesi tekniğidir. CPU üzerinde çok hızlı çalışır 
      ancak yüzleri farklı açılardan veya kötü ışıkta bulmakta zorlanabilir.

2.  **CNN (Convolutional Neural Network):**
    *   **Tanım:** Derin öğrenme tabanlı bir modeldir. Çok yüksek doğrulukla yüz bulur 
      ancak CPU üzerinde çok yavaştır. Genellikle GPU gerektirir.

3.  **YOLO (You Only Look Once):**
    *   **Tanım:** Modern ve çok hızlı bir derin öğrenme modelidir. GPU üzerinde gerçek 
      zamanlı çalışır ve yüksek doğruluk sunar.

### Yüz Tanıma Modelleri (Recognition Models) ###

1.  **dlib (ResNet tabanlı):**
    *   **Tanım:** Yüzü 128 sayıdan oluşan bir vektöre (embedding) dönüştüren popüler 
      bir modeldir. Genellikle L2 (Euclidean) mesafesi ile kullanılır.

2.  **FaceNet:**
    *   **Tanım:** Google tarafından geliştirilen ve çok yüksek doğruluk sunan bir modeldir. 
      Yüzü 128 boyutlu bir vektöre dönüştürür ve genellikle Kosinüs (Cosine) 
      mesafesi ile en iyi sonucu verir.
"""

import os
import cv2
import numpy as np
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# Updated import paths
from core.database import Database
import config.settings as settings
from core.detector import detect_faces, detect_faces_with_score
from deepface import DeepFace
import face_recognition

# إعدادات العرض
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# مجلد الاختبار
TEST_DIR = "data/TestImages"

# ==========================================
# BÖLÜM 1: STATİK TEST FONKSİYONLARI
# ==========================================

def run_detection_benchmark():
    """HOG, CNN ve YOLO modellerini hız ve başarı oranı açısından karşılaştırır."""
    print("\n" + "="*40)
    print("AŞAMA 1: YÜZ TESPİT MODELLERİ KARŞILAŞTIRMASI")
    print("="*40)
    
    models = ["hog", "cnn", "yolo"]
    results = []
    raw_times_data = [] 
    
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
                    confidence=settings.YOLO_CONFIDENCE, 
                    yolo_weights=settings.YOLO_WEIGHTS
                )
                duration = time.perf_counter() - start_time
                times.append(duration)
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

def get_encoding(image_rgb, model_name):
    locations = detect_faces(
        image_rgb, 
        settings.FACE_DETECTION_MODEL, 
        confidence=settings.YOLO_CONFIDENCE, 
        yolo_weights=settings.YOLO_WEIGHTS
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
        "raw_times": processing_times
    }

def run_static_benchmark_suite():
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
    if det_raw_times is not None and not det_raw_times.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y='Süre (sn)', data=det_raw_times, palette='Set3')
        plt.title('Yüz Tespit Hızı Kararlılığı (Box Plot)', fontsize=14)
        plt.ylabel('Süre (Saniye)')
        plt.show()

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

    if not rec_df.empty:
        melted_df = rec_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], var_name="Metrik", value_name="Değer")
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Metrik", y="Değer", hue="Model", data=melted_df, palette="viridis")
        plt.title("Yüz Tanıma Performans Metrikleri (Detaylı)", fontsize=14)
        plt.ylim(0, 1.1)
        plt.legend(loc='lower right')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Süre (sn)', data=rec_times_df, palette='Pastel1')
        plt.title('Yüz Kodlama (Encoding) Hızı Kararlılığı', fontsize=14)
        plt.ylabel('Süre (Saniye)')
        plt.show()

        for model_name, (y_true, y_pred) in confusion_data.items():
            plt.figure(figsize=(8, 6))
            unique_labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
            plt.title(f'{model_name} - Karmaşıklık Matrisi')
            plt.xlabel('Tahmin Edilen')
            plt.ylabel('Gerçek')
            plt.show()

# ==========================================
# BÖLÜM 2: CANLI TEST (GUI)
# ==========================================

class CanliTespitUygulamasi:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Canlı Yüz Tespit Testi (Live Benchmark)")
        self.root.geometry("400x250")
        
        self.model_name = tk.StringVar(value="yolo")
        self.is_running = False
        
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill="both")

        ttk.Label(main_frame, text="Tespit Modeli Seçiniz:", font=("Helvetica", 10, "bold")).pack(pady=5)
        
        models = ["yolo", "hog", "cnn"]
        for model in models:
            ttk.Radiobutton(main_frame, text=model.upper(), variable=self.model_name, value=model).pack(anchor="w", padx=40)
            
        self.start_btn = ttk.Button(main_frame, text="Kamerayı Başlat", command=self.start_camera)
        self.start_btn.pack(pady=20, fill="x")

    def start_camera(self):
        if self.is_running: return
            
        self.is_running = True
        self.start_btn.config(state="disabled")
        
        selected_model = self.model_name.get()
        print(f"Kamera başlatılıyor... Model: {selected_model.upper()}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Web kamerası açılamadı.")
            self.is_running = False
            self.start_btn.config(state="normal")
            return

        prev_frame_time = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Yeni fonksiyonu kullan (Skorlu tespit)
            detections = detect_faces_with_score(
                rgb_frame, 
                model_name=selected_model,
                confidence=0.3, 
                yolo_weights=settings.YOLO_WEIGHTS
            )

            for (location, score) in detections:
                top, right, bottom, left = location
                
                # Dikdörtgen çiz
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Skor yaz
                text = f"Skor: {score:.2f}"
                cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(f"Canli Tespit - {selected_model.upper()}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        try:
            self.start_btn.config(state="normal")
        except:
            pass

    def on_closing(self):
        self.is_running = False
        self.root.destroy()

# ==========================================
# ANA MENÜ
# ==========================================

def main():
    print("\n" + "="*50)
    print("   YÜZ TANIMA SİSTEMİ - PERFORMANS TEST ARACI")
    print("="*50)
    print("1. Statik Test (Resim Klasörü Analizi)")
    print("2. Canlı Test (Kamera ile Gerçek Zamanlı Analiz)")
    print("="*50)
    
    choice = input("\nLütfen bir işlem seçiniz (1 veya 2): ").strip()
    
    if choice == '1':
        run_static_benchmark_suite()
    elif choice == '2':
        CanliTespitUygulamasi()
    else:
        print("Geçersiz seçim. Program kapatılıyor.")

if __name__ == "__main__":
    main()
