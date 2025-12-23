# 📂 Performans Testleri Klasörü (Benchmarks)

Bu klasör, sistemin performansını ve hızını ölçmek için araçlar içerir.

## 📄 Dosyalar

### 1. `check_gpu.py`
*   **İşlev:** `torch` (YOLO için) ve `dlib` kütüphanelerinin ekran kartı (GPU/CUDA) üzerinde çalışıp çalışmadığını kontrol eder.
*   **Önemi:** Sistemin mümkün olan en yüksek hızda çalıştığından emin olmak.

### 2. `detection_only.py`
*   **İşlev:** Sadece yüz tespit hızını ölçer (tanıma işlemi olmadan).
*   **Amaç:** YOLO'nun hızını HOG veya CNN ile karşılaştırmak.

### 3. `live_test.py`
*   **İşlev:** Tam işlem sırasında saniyedeki kare sayısını (FPS) ölçmek için sistemi canlı ortamda (Live) test eder.

### 4. `suite.py`
*   **İşlev:** `data/TestImages` klasörü üzerinde kapsamlı testler çalıştırır.
*   **Not:** Çalışması için belirtilen klasörde resimlerin olması gerekir, aksi takdirde `Error: No images found` hatası görünür.