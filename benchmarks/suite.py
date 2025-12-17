# -*- coding: utf-8 -*-
"""
Yüz Tanıma Sistemi Karşılaştırma Raporu (Benchmark Suite)
GUI Versiyonu - v2.0 (İyileştirilmiş ve Hata Ayıklanmış)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import numpy as np
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
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

# --- RENK PALETİ ---
COLORS = {
    "bg": "#344e41", "frame": "#3a5a40", "button": "#588157",
    "hover": "#a3b18a", "text": "#dad7cd", "accent1": "#fde4cf", "accent2": "#90e0ef",
}

# --- GRAFİK AYARLARI ---
sns.set_style("darkgrid", {"axes.facecolor": COLORS["frame"], "axes.edgecolor": COLORS["text"], "axes.labelcolor": COLORS["text"], "text.color": COLORS["text"], "xtick.color": COLORS["text"], "ytick.color": COLORS["text"], "grid.color": COLORS["bg"]})
plt.rcParams.update({'figure.facecolor': COLORS["bg"], 'figure.edgecolor': COLORS["bg"], 'savefig.facecolor': COLORS["bg"], 'savefig.edgecolor': COLORS["bg"], 'font.size': 12, 'font.family': 'sans-serif'})

TEST_DIR = os.path.join(PROJECT_ROOT, "data", "TestImages")

# ==========================================
# YARDIMCI SINIF: ÇIKTI YÖNLENDİRME
# ==========================================
class TextRedirector:
    def __init__(self, widget): self.widget = widget
    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str)
        self.widget.see("end")
        self.widget.configure(state="disabled")
        self.widget.update_idletasks()
    def flush(self): pass

# ==========================================
# GELİŞMİŞ GRAFİK FONKSİYONLARI
# ==========================================

def plot_threshold_analysis(model_name, y_true, y_distances):
    """Farklı eşik değerlerine göre doğruluk değişimini analiz eder."""
    print(f"  > {model_name} için Eşik Değeri Analizi yapılıyor...")
    thresholds = np.arange(0.1, 1.0, 0.05)
    accuracies = []
    
    for thresh in thresholds:
        correct = 0
        total = 0
        for true_name, (pred_name, dist) in zip(y_true, y_distances):
            if dist == float('inf'): continue
            total += 1
            final_pred = pred_name if dist < thresh else "BİLİNMİYOR"
            if final_pred == true_name:
                correct += 1
        accuracies.append(correct / total if total > 0 else 0)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, accuracies, marker='o', linestyle='-', color=COLORS["accent2"], linewidth=2, label="Doğruluk")
    plt.title(f'Eşik Değeri Analizi - {model_name}', color='white', fontsize=16)
    plt.xlabel('Eşik Değeri (Mesafe Limiti)', color='white')
    plt.ylabel('Doğruluk (Accuracy)', color='white')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if accuracies:
        best_idx = np.argmax(accuracies)
        best_thresh, best_acc = thresholds[best_idx], accuracies[best_idx]
        plt.axvline(best_thresh, color='red', linestyle='--', label=f'En İyi Eşik: {best_thresh:.2f}')
        print(f"  > {model_name} İçin En İyi Eşik Değeri: {best_thresh:.2f} (Doğruluk: {best_acc:.2%})")
    
    plt.legend()
    plt.show()

def plot_roc_curve(model_name, y_true, y_distances):
    """Modelin genel performansını ölçmek için ROC eğrisini çizer."""
    print(f"  > {model_name} için ROC Eğrisi hazırlanıyor...")
    
    y_true_binary, y_scores = [], []
    for true_name, (pred_name, dist) in zip(y_true, y_distances):
        if dist == float('inf'): continue
        y_true_binary.append(1 if true_name == pred_name else 0)
        y_scores.append(1.0 / (1.0 + dist))

    if len(set(y_true_binary)) < 2:
        print(f"  UYARI: ROC eğrisi çizilemiyor. Veride sadece tek bir sınıf var.")
        return

    try:
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color=COLORS["accent1"], lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı', color='white')
        plt.ylabel('Doğru Pozitif Oranı', color='white')
        plt.title(f'ROC Eğrisi - {model_name}', color='white', fontsize=16)
        plt.legend(loc="lower right")
        plt.show()
    except Exception as e:
        print(f"  HATA: ROC çiziminde hata oluştu: {e}")

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name):
    """Hata matrisini çizer."""
    try:
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, cbar=False)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, color='white', pad=20)
        plt.xlabel('Tahmin Edilen', color='white', fontsize=12)
        plt.ylabel('Gerçek', color='white', fontsize=12)
        plt.xticks(rotation=45); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Heatmap hatası: {e}")

def plot_radar_chart(df, title):
    """Radar grafiği ile modelleri karşılaştırır."""
    try:
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_facecolor(COLORS["frame"])
        
        colors = ['#00ff00', '#ff00ff', '#00ffff']
        for idx, row in df.iterrows():
            values = row[labels].tolist() + [row[labels][0]]
            color = colors[idx % len(colors)]
            ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=row['Model'])
            ax.fill(angles, values, color=color, alpha=0.25)
        
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], labels, color='white', size=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
        plt.ylim(0, 1.0)
        plt.title(title, size=16, color='white', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor=COLORS["frame"], labelcolor='white')
        plt.show()
    except Exception as e:
        print(f"Radar chart hatası: {e}")

# ==========================================
# BÖLÜM 1: MANTIK FONKSİYONLARI
# ==========================================

def run_detection_benchmark():
    print("\n" + "="*40 + "\nAŞAMA 1: YÜZ TESPİT MODELLERİ KARŞILAŞTIRMASI\n" + "="*40)
    models, results, raw_times_data = ["hog", "cnn", "yolo"], [], []
    all_images = [os.path.join(r, f) for r, _, fs in os.walk(TEST_DIR) for f in fs if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not all_images:
        print(f"Hata: '{TEST_DIR}' klasöründe test görseli bulunamadı!"); return None, None

    for model in models:
        print(f"Model Test Ediliyor: {model.upper()}...")
        times, success_count = [], 0
        for i, img_path in enumerate(all_images):
            if i % 10 == 0: print(f"  > İşleniyor... {i}/{len(all_images)}")
            image = cv2.imread(img_path)
            if image is None: continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            start_time = time.perf_counter()
            try:
                locations = detect_faces(rgb_image, model, settings.YOLO_CONFIDENCE, settings.YOLO_WEIGHTS)
                duration = time.perf_counter() - start_time
                times.append(duration)
                raw_times_data.append({"Model": model.upper(), "Süre (sn)": duration})
                if len(locations) > 0: success_count += 1
            except Exception as e: print(f"  Hata ({model}): {e}")
        
        avg_time = np.mean(times) if times else 0
        success_rate = (success_count / len(all_images)) * 100 if all_images else 0
        results.append({"Model": model.upper(), "Ort. Süre (sn)": avg_time, "Başarı Oranı (%)": success_rate})
        print(f"  SONUÇ > {model.upper()}: {avg_time:.4f}s | Başarı: {success_rate:.1f}%")
    return pd.DataFrame(results), pd.DataFrame(raw_times_data)

def get_encoding(image_rgb, model_name):
    locations = detect_faces(image_rgb, settings.FACE_DETECTION_MODEL, settings.YOLO_CONFIDENCE, settings.YOLO_WEIGHTS)
    if not locations: return None, 0
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
                if embedding_objs: encoding = np.array(embedding_objs[0]['embedding'])
            except: pass
    return encoding, time.perf_counter() - start_time

def find_nearest_face(encoding, cursor, model_name):
    if encoding is None: return "BİLİNMİYOR", float('inf')
    vec_str = str(encoding.tolist())
    column_name, op = ("encoding", "<->") if model_name == "dlib" else ("encoding_facenet", "<=>")
    try:
        query = f"SELECT p.name, f.{column_name} {op} %s AS distance FROM people p JOIN face_encodings f ON p.id = f.person_id WHERE f.{column_name} IS NOT NULL ORDER BY distance ASC LIMIT 1;"
        cursor.execute(query, (vec_str,))
        result = cursor.fetchone()
        return (result[0], result[1]) if result else ("BİLİNMİYOR", float('inf'))
    except: return "HATA", float('inf')

def run_recognition_benchmark(model_name, cursor):
    print(f"\nTest Ediliyor: {model_name.upper()}...")
    y_true, y_pred, y_distances, processing_times = [], [], [], []
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
            pred_name, distance = find_nearest_face(encoding, cursor, model_name)
            y_true.append(true_name)
            default_thresh = 0.6 if model_name == 'dlib' else 0.4
            y_pred.append(pred_name if distance < default_thresh else "BİLİNMİYOR")
            y_distances.append((pred_name, distance))
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    avg_time = np.mean(processing_times) if processing_times else 0
    return {"Model": model_name.upper(), "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1, "Ort. Süre (sn)": avg_time, "y_true": y_true, "y_pred": y_pred, "y_distances": y_distances, "raw_times": processing_times}

def run_static_benchmark_suite():
    det_df, det_raw_times = run_detection_benchmark()
    
    print("\n" + "="*40 + "\nAŞAMA 2: YÜZ TANIMA MODELLERİ KARŞILAŞTIRMASI\n" + "="*40)
    Database.initialize_pool()
    rec_results, rec_raw_times, model_analysis_data = [], [], {}
    with Database.get_conn() as conn:
        with conn.cursor() as cursor:
            for model_name in ["dlib", "facenet"]:
                res = run_recognition_benchmark(model_name, cursor)
                if res:
                    y_true, y_pred, y_distances = res.pop("y_true"), res.pop("y_pred"), res.pop("y_distances")
                    model_analysis_data[model_name.upper()] = (y_true, y_pred, y_distances)
                    for t in res.pop("raw_times"): rec_raw_times.append({"Model": model_name.upper(), "Süre (sn)": t})
                    rec_results.append(res)
    Database.close_all()
    rec_df = pd.DataFrame(rec_results)
    rec_times_df = pd.DataFrame(rec_raw_times)

    print("\n" + "="*40 + "\nTEST TAMAMLANDI. GELİŞMİŞ GRAFİKLER HAZIRLANIYOR...\n" + "="*40)

    # --- VISUALIZATION ---
    if det_raw_times is not None and not det_raw_times.empty:
        plt.figure(); sns.violinplot(x='Model', y='Süre (sn)', data=det_raw_times, palette="viridis", inner="quartile"); plt.yscale('log'); plt.title('Yüz Tespit Hızı Dağılımı (Log Scale)', color='white', fontsize=16); plt.show()
    if det_df is not None:
        plt.figure(); ax = sns.barplot(x='Model', y='Başarı Oranı (%)', data=det_df, palette="coolwarm"); plt.title('Yüz Tespit Başarı Oranları', color='white', fontsize=16); plt.ylim(0, 115)
        for p in ax.patches: ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='center', fontsize=12, color='white', xytext=(0, -10), textcoords='offset points', weight='bold')
        plt.show()
    if not rec_df.empty:
        plot_radar_chart(rec_df, "Model Performans Karşılaştırması (Radar Chart)")
    if not rec_times_df.empty:
        plt.figure(); sns.violinplot(x='Model', y='Süre (sn)', data=rec_times_df, palette="magma"); plt.title('Yüz Kodlama (Encoding) Süreleri', color='white', fontsize=16); plt.show()
    for model_name, (y_true, y_pred, y_distances) in model_analysis_data.items():
        print(f"\n--- {model_name} İÇİN DETAYLI ANALİZ ---")
        plot_threshold_analysis(model_name, y_true, y_distances)
        plot_roc_curve(model_name, y_true, y_distances)
        plot_confusion_matrix_heatmap(y_true, y_pred, model_name)

# ==========================================
# BÖLÜM 2: CANLI TEST (GUI)
# ==========================================
class CanliTespitUygulamasi:
    def __init__(self, parent_root=None):
        self.root = ctk.CTkToplevel(parent_root) if parent_root else ctk.CTk()
        if parent_root: self.root.transient(parent_root)
        self.root.title("Canlı Yüz Tespit Testi"); self.root.geometry("400x300"); self.root.configure(fg_color=COLORS["bg"])
        self.model_name, self.is_running = ctk.StringVar(value="yolo"), False
        self.create_widgets()
        if not parent_root: self.root.mainloop()
    def create_widgets(self):
        self.root.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.root, text="Tespit Modeli Seçiniz:", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLORS["text"]).grid(row=0, column=0, padx=20, pady=(20,10))
        for i, model in enumerate(["yolo", "hog", "cnn"]):
            ctk.CTkRadioButton(self.root, text=model.upper(), variable=self.model_name, value=model, text_color=COLORS["text"], fg_color=COLORS["button"], hover_color=COLORS["hover"]).grid(row=i+1, column=0, padx=40, pady=5, sticky="w")
        self.start_btn = ctk.CTkButton(self.root, text="Kamerayı Başlat", command=self.start_camera, fg_color=COLORS["button"], hover_color=COLORS["hover"], text_color=COLORS["text"])
        self.start_btn.grid(row=len(["yolo", "hog", "cnn"])+1, column=0, padx=20, pady=20, sticky="ew")
    def start_camera(self):
        if self.is_running: return
        self.is_running, selected_model = True, self.model_name.get()
        self.start_btn.configure(state="disabled"); self.root.withdraw()
        print(f"Kamera başlatılıyor... Model: {selected_model.upper()}")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Web kamerası açılamadı."); self.is_running = False; self.start_btn.configure(state="normal"); self.root.deiconify(); return
        prev_frame_time = 0
        while self.is_running:
            ret, frame = cap.read()
            if not ret: break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detect_faces_with_score(rgb_frame, selected_model, 0.3, settings.YOLO_WEIGHTS)
            for (location, score) in detections:
                top, right, bottom, left = location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Skor: {score:.2f}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f"Canli Tespit - {selected_model.upper()}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release(); cv2.destroyAllWindows(); self.is_running = False
        try: self.start_btn.configure(state="normal"); self.root.deiconify()
        except: pass
    def on_closing(self): self.is_running = False; self.root.destroy()

# ==========================================
# ANA GUI UYGULAMASI
# ==========================================
class BenchmarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Benchmark Suite - Performans Testi"); self.geometry("900x700"); self.configure(fg_color=COLORS["bg"])
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Yüz Tanıma Sistemi - Performans Test Aracı", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLORS["text"]).grid(row=0, column=0, pady=20)
        self.log_box = ctk.CTkTextbox(self, font=("Consolas", 12), fg_color=COLORS["frame"], text_color=COLORS["text"], border_color=COLORS["button"], border_width=1)
        self.log_box.grid(row=1, column=0, padx=20, pady=10, sticky="nsew"); self.log_box.configure(state="disabled")
        btn_frame = ctk.CTkFrame(self, fg_color="transparent"); btn_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew"); btn_frame.grid_columnconfigure((0, 1), weight=1)
        self.btn_static = ctk.CTkButton(btn_frame, text="1. Statik Test (Raporlar)", command=self.start_static_test, height=50, font=ctk.CTkFont(size=16), fg_color=COLORS["button"], hover_color=COLORS["hover"], text_color=COLORS["text"])
        self.btn_static.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.btn_live = ctk.CTkButton(btn_frame, text="2. Canlı Test (Kamera)", command=self.start_live_test, height=50, font=ctk.CTkFont(size=16), fg_color=COLORS["button"], hover_color=COLORS["hover"], text_color=COLORS["text"])
        self.btn_live.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        sys.stdout = TextRedirector(self.log_box)
        print("Sistem hazır. Lütfen bir test seçiniz...")
    def start_static_test(self):
        self.btn_static.configure(state="disabled", text="Çalışıyor..."); self.btn_live.configure(state="disabled")
        print("\n--- Statik Test Başlatılıyor ---\n")
        threading.Thread(target=self._run_static_thread, daemon=True).start()
    def _run_static_thread(self):
        try: run_static_benchmark_suite()
        except Exception as e: print(f"\nHATA OLUŞTU: {e}")
        finally:
            self.btn_static.configure(state="normal", text="1. Statik Test (Raporlar)"); self.btn_live.configure(state="normal")
            print("\n--- İşlem Tamamlandı ---")
    def start_live_test(self):
        try: CanliTespitUygulamasi(self)
        except Exception as e: messagebox.showerror("Hata", f"Canlı test başlatılamadı: {e}")

if __name__ == "__main__":
    app = BenchmarkApp()
    app.mainloop()
