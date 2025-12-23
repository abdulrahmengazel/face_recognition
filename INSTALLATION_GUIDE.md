# 🛠️ Kurulum ve Yapılandırma Kılavuzu (Installation Guide)

Bu proje, yüksek performanslı yüz tanıma işlemleri için **GPU (Ekran Kartı)** gücünden yararlanır. Sistemin tam performansla çalışması için aşağıdaki adımları eksiksiz uygulamanız gerekmektedir.

---

## 📋 1. Temel Gereksinimler (Prerequisites)

*   **İşletim Sistemi:** Windows 10/11 (veya Linux/macOS)
*   **Python:** Sürüm 3.9 veya 3.10 (3.11+ bazı kütüphanelerle uyumsuzluk çıkarabilir).
*   **Veritabanı:** PostgreSQL 15 veya daha yeni bir sürüm.
*   **Donanım:** NVIDIA Ekran Kartı (CUDA destekli) önerilir.

---

## 🎮 2. NVIDIA CUDA ve cuDNN Kurulumu (GPU Hızlandırma İçin)

YOLOv8 ve Dlib'in hızlı çalışması için bu adım **kritiktir**.

### Adım 2.1: Ekran Kartı Sürücüsü
NVIDIA GeForce Experience veya resmi web sitesinden en güncel ekran kartı sürücüsünü (Game Ready Driver) yükleyin.

### Adım 2.2: CUDA Toolkit Kurulumu
1.  Komut satırını (CMD) açın ve `nvidia-smi` yazın. Sağ üstte **CUDA Version: 12.x** gibi bir yazı göreceksiniz.
2.  Bu sürümle uyumlu (veya bir alt sürüm, örn: 11.8 veya 12.1) **CUDA Toolkit** indirin.
    *   [CUDA Toolkit İndirme Sayfası](https://developer.nvidia.com/cuda-downloads)
3.  İndirilen `.exe` dosyasını kurun (Express kurulum seçebilirsiniz).

### Adım 2.3: PyTorch Kurulumu (CUDA Destekli)
Projeyi kurmadan önce PyTorch'un GPU sürümünü yüklemelisiniz.
Proje klasöründe terminali açın ve şu komutu çalıştırın (CUDA 11.8 için örnek):

```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*(Eğer CUDA 12.1 kurduysanız `cu121` kullanın)*

---

## 🗄️ 3. Veritabanı Kurulumu (PostgreSQL & pgvector)

Bu proje, yüz vektörlerini (embeddings) saklamak ve aramak için **pgvector** eklentisini kullanır.

1.  **PostgreSQL'i Yükleyin:** [Resmi sitesinden](https://www.postgresql.org/download/) indirip kurun. Kurulum sırasında şifreyi unutmayın (Varsayılan: `postgres`).
2.  **pgAdmin 4'ü Açın** ve yeni bir veritabanı oluşturun (Örn: `postgres`).
3.  **pgvector Eklentisini Kurun:**
    *   Windows için: PostgreSQL kurulum klasöründeki "Stack Builder" uygulamasını çalıştırın ve `pgvector` eklentisini seçip yükleyin.
    *   Veya SQL Sorgu aracını açıp şu komutu çalıştırın:
    ```sql
    CREATE EXTENSION vector;
    ```
    *(Hata alırsanız, pgvector'ün sisteminizde kurulu olduğundan emin olun. Windows'ta bazen manuel derleme veya hazır binary gerekebilir).*

---

## 📦 4. Proje Bağımlılıklarının Yüklenmesi

1.  Proje klasörüne gidin.
2.  Sanal ortam oluşturun (Önerilen):
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
3.  Kütüphaneleri yükleyin:
    ```powershell
    pip install -r requirements.txt
    ```

---

## ⚙️ 5. Dlib GPU Desteği (İsteğe Bağlı ama Önerilir)

`face_recognition` kütüphanesi varsayılan olarak CPU kullanır. GPU kullanması için:

1.  Visual Studio (Community Edition) yükleyin ve **"Desktop development with C++"** seçeneğini işaretleyin.
2.  Mevcut dlib'i kaldırın: `pip uninstall dlib`
3.  GPU desteğiyle tekrar derleyin:
    ```powershell
    pip install dlib --no-binary dlib
    ```
    *(Bu işlem birkaç dakika sürebilir)*

---

## ✅ 6. Kurulumu Doğrulama

Her şeyin doğru çalıştığını test etmek için `benchmarks/check_gpu.py` dosyasını çalıştırın:

```powershell
python benchmarks/check_gpu.py
```

Çıktıda şunları görmelisiniz:
*   `Torch (YOLO) GPU: True`
*   `Dlib GPU: True` (Eğer Adım 5'i yaptıysanız)

---

## 🚀 7. Çalıştırma

Artık sistemi başlatabilirsiniz:

*   **Masaüstü Uygulaması:** `python main.py`
*   **API Sunucusu:** `python api.py`
