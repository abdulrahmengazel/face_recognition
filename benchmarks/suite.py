# -*- coding: utf-8 -*-
"""
Yüz Tanıma Sistemi Karşılaştırma Raporu (Benchmark Suite)
GUI Versiyonu
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import numpy as np
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import threading
import customtkinter as ctk
from tkinter import messagebox

# --- DİNAMİK YOL AYARI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from core.database import Database
import config.settings as settings
from core.detector import detect_faces, detect_faces_with_score
from deepface import DeepFace
import face_recognition

# Ayarlar
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "TestImages")

# ==========================================
# YARDIMCI SINIF: ÇIKTI YÖNLENDİRME (REDIRECT)
# ==========================================
class TextRedirector(object):
    """Konsol çıktılarını (print) GUI'deki Textbox'a yönlendirir."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
        # GUI'nin donmasını engellemek için update
        self.widget.update_idletasks()

    def flush(self):
        pass

# ==========================================
# BÖLÜM 1: MANTIK FONKSİYONLARI (LOGIC)
# ==========================================

def run_detection_benchmark():
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
        print(f"Hata: '{TEST_DIR}' klasöründe test görseli bulunamadı!")
        return None, None

    for model in models:
        print(f"Model Test Ediliyor: {model.upper()}...")
        times = []
        success_count = 0
        
        for i, img_path in enumerate(all_images):
            # İlerleme göstergesi (basit)
            if i % 5 == 0: print(f"  > İşleniyor... {i}/{len(all_images)}")
            
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
        print(f"  SONUÇ > {model.upper()}: {avg_time:.4f}s | Başarı: {success_rate:.1f}%")

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
            
            # print(f"  > {img_file}: {predicted_name}") # Çok fazla çıktı olmaması için kapattım

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    avg_time = np.mean(processing_times) if processing_times else 0
    
    print(f"  {model_name.upper()} Tamamlandı. Doğruluk: {accuracy:.2%}")
    
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
    # Detection
    det_df, det_raw_times = run_detection_benchmark()
    
    # Recognition
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

    print("\n" + "="*40)
    print("TEST TAMAMLANDI. GRAFİKLER HAZIRLANIYOR...")
    print("="*40)

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
    def __init__(self, parent_root=None):
        if parent_root:
            self.root = ctk.CTkToplevel(parent_root)
            self.root.transient(parent_root)
        else:
            self.root = ctk.CTk()
            
        self.root.title("Canlı Yüz Tespit Testi")
        self.root.geometry("400x300")
        
        self.model_name = ctk.StringVar(value="yolo")
        self.is_running = False
        
        self.create_widgets()
        if not parent_root:
            self.root.mainloop()

    def create_widgets(self):
        self.root.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.root, text="Tespit Modeli Seçiniz:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(20,10))
        
        models = ["yolo", "hog", "cnn"]
        for i, model in enumerate(models):
            ctk.CTkRadioButton(self.root, text=model.upper(), variable=self.model_name, value=model).grid(row=i+1, column=0, padx=40, pady=5, sticky="w")
            
        self.start_btn = ctk.CTkButton(self.root, text="Kamerayı Başlat", command=self.start_camera)
        self.start_btn.grid(row=len(models)+1, column=0, padx=20, pady=20, sticky="ew")

    def start_camera(self):
        if self.is_running: return
            
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.root.withdraw()
        
        selected_model = self.model_name.get()
        print(f"Kamera başlatılıyor... Model: {selected_model.upper()}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Web kamerası açılamadı.")
            self.is_running = False
            self.start_btn.configure(state="normal")
            self.root.deiconify()
            return

        prev_frame_time = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detections = detect_faces_with_score(
                rgb_frame, 
                model_name=selected_model,
                confidence=0.3, 
                yolo_weights=settings.YOLO_WEIGHTS
            )

            for (location, score) in detections:
                top, right, bottom, left = location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                text = f"Skor: {score:.2f}"
                cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(f"Canli Tespit - {selected_model.upper()}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        try:
            self.start_btn.configure(state="normal")
            self.root.deiconify()
        except:
            pass

# ==========================================
# ANA GUI UYGULAMASI (BENCHMARK APP)
# ==========================================

class BenchmarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Benchmark Suite - Performans Testi")
        self.geometry("800x600")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # Log alanı genişlesin

        # Başlık
        ctk.CTkLabel(self, text="Yüz Tanıma Sistemi - Performans Test Aracı", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, pady=20)

        # Log Alanı (Textbox)
        self.log_box = ctk.CTkTextbox(self, font=("Consolas", 12))
        self.log_box.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.log_box.configure(state="disabled")

        # Butonlar Alanı
        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_static = ctk.CTkButton(btn_frame, text="1. Statik Test (Resim Klasörü)", command=self.start_static_test, height=40)
        self.btn_static.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.btn_live = ctk.CTkButton(btn_frame, text="2. Canlı Test (Kamera)", command=self.start_live_test, height=40, fg_color="#E67E22", hover_color="#D35400")
        self.btn_live.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Çıktı Yönlendirme
        sys.stdout = TextRedirector(self.log_box)
        sys.stderr = TextRedirector(self.log_box) # Hataları da yakala

        print("Sistem hazır. Lütfen bir test seçiniz...")

    def start_static_test(self):
        self.btn_static.configure(state="disabled")
        self.btn_live.configure(state="disabled")
        print("\n--- Statik Test Başlatılıyor ---\n")
        
        # Thread içinde çalıştır ki GUI donmasın
        threading.Thread(target=self._run_static_thread, daemon=True).start()

    def _run_static_thread(self):
        try:
            run_static_benchmark_suite()
        except Exception as e:
            print(f"\nHATA OLUŞTU: {e}")
        finally:
            self.btn_static.configure(state="normal")
            self.btn_live.configure(state="normal")
            print("\n--- İşlem Tamamlandı ---")

    def start_live_test(self):
        # Canlı test zaten kendi penceresini açıyor
        CanliTespitUygulamasi(self)

if __name__ == "__main__":
    app = BenchmarkApp()
    app.mainloop()
