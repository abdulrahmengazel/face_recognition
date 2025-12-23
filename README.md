# 🤖 Akıllı Okul Yüz Tanıma Sistemi (Smart School Face Recognition System)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100.0+-009688.svg" alt="FastAPI Version">
  <img src="https://img.shields.io/badge/PostgreSQL-15+-336791.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

### 🌟 Genel Bakış (Overview)

Modern okul yönetimi için tasarlanmış uçtan uca, yüksek performanslı bir yüz tanıma sistemi. İdari görevler için **Tkinter tabanlı Masaüstü GUI** ile mobil ve web entegrasyonu için **FastAPI REST arka ucunu** birleştirir.

Ölçeklenebilirlik göz önünde bulundurularak oluşturulan sistem, binlerce kimliği kolaylıkla destekleyen ultra hızlı benzerlik aramaları için **pgvector** uzantılı **PostgreSQL** kullanır.

---

### 🚀 Temel Özellikler

- **🖥️ Çift Arayüz:** Yönetici Masaüstü Uygulaması (Tkinter) & Mobil Uyumlu API (FastAPI).
- **🧠 Gelişmiş Yapay Zeka Modelleri:**
  - **Tespit (Detection):** HOG, CNN ve YOLOv8 desteği.
  - **Tanıma (Recognition):** dlib ve FaceNet gömüleri (embeddings).
- **⚡ Yüksek Performans:** `pgvector` kullanarak veritabanı tabanlı benzerlik araması.
- **📸 Esnek Tanıma:** Statik resimleri, canlı web kamerası akışlarını ve toplu eğitimi destekler.
- **📊 Güçlü Performans Testleri:** Model doğruluğunu ve hızını değerlendirmek için yerleşik araçlar.
- **⚙️ Yapılandırılabilir:** Kolayca ayarlanabilen eşik değerleri, ölçeklendirme ve eğitim parametreleri.

---

### 🛠️ Teknoloji Yığını (Tech Stack)

- **Backend:** Python, FastAPI, Uvicorn
- **GUI:** Tkinter, OpenCV
- **AI/ML:** Ultralytics (YOLO), Face Recognition (dlib), DeepFace (FaceNet)
- **Veritabanı:** PostgreSQL + `pgvector`
- **Altyapı:** GPU hızlandırması için CUDA/cuDNN desteği

---

### 📥 Hızlı Başlangıç (Quick Start)

#### 1. Ortam Kurulumu
```powershell
# Sanal ortamı oluştur ve etkinleştir
python -m venv .venv
.\.venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

#### 2. Eğitim Verilerini Hazırlama
Resimlerinizi `data/TrainingImages/` içinde kişi başına bir klasör yapısı kullanarak düzenleyin:
```text
data/TrainingImages/
├── 👤 Ali/
│   ├── img1.jpg
│   └── img2.jpg
└── 👤 Ayse/
    ├── img1.jpg
    └── img2.jpg
```

#### 3. Masaüstü Uygulamasını Çalıştırın
```powershell
python main.py
```
*Modelleri yapılandırmak, toplu eğitim çalıştırmak ve tanımayı test etmek için GUI'yi kullanın.*

#### 4. API Sunucusunu Başlatın
```powershell
python api.py
```
*API `http://localhost:8000` adresinde mevcut olacaktır. Dokümanlara `/docs` adresinden erişebilirsiniz.*

---

### 📂 Proje Yapısı

```text
PythonProject/
├── 📱 api.py              # FastAPI sunucu giriş noktası
├── 🖥️ main.py             # Masaüstü GUI giriş noktası
├── 📂 apps/               # GUI uygulama modülleri
├── 📂 assets/             # Statik varlıklar (YOLO ağırlıkları)
├── 📂 benchmarks/         # Performans test betikleri
├── 📂 config/             # Genel yapılandırmalar
├── 📂 core/               # Veritabanı & Dedektör mantığı
└── 📂 data/               # Eğitim & Test veri setleri
```

---

### 💡 Notlar & İpuçları

- **GPU Hızlandırma:** YOLO ve CNN modelleri için CUDA ve cuDNN'in doğru yapılandırıldığından emin olun.
- **Veritabanı:** `pgvector` uzantısı yüklü bir PostgreSQL örneği gerektirir.
- **YOLO Ağırlıkları:** `.pt` dosyalarınızı `assets/yolo/` içine yerleştirin.
- **Sorun Giderme:**
  - *DLL Hataları:* [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) yükleyin.
  - *Veritabanı:* `config/settings.py` içindeki bağlantı dizelerini kontrol edin.

---

### 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.