# 📘 Proje Dokümantasyonu (Project Documentation)

Bu dosya, projenin genel yapısını ve her klasörün amacını açıklar.

## 📂 Klasör Yapısı

### 1. `apps/` (Uygulamalar)
Sistemi farklı modlarda (resim, video, eğitim) çalıştırmak için ana komut dosyalarını içerir.

### 2. `benchmarks/` (Performans Testleri)
Sistemin hızını ve doğruluğunu ölçmek ve GPU uyumluluğunu test etmek için araçlar içerir.
**Not:** `Error: No images found in 'data/TestImages'` hatası alıyorsanız, test için ayrılan resim klasörü boş demektir.

### 3. `config/` (Ayarlar)
Veritabanı ayarları ve model yolları gibi yapılandırma dosyalarını ve sabit değişkenleri içerir.

### 4. `core/` (Çekirdek)
Diğer uygulamaların dayandığı temel kodları ve iş mantığını (veritabanı, yüz dedektörü) içerir.

### 5. `data/` (Veri)
Eğitim (Training) ve Test (Testing) için kullanılan resimlerin saklandığı klasördür.

### 6. `assets/` (Kaynaklar)
Önceden eğitilmiş yapay zeka modellerini (örneğin `yolov8n-face.pt`) içerir.

---
Her bölüm hakkında daha fazla ayrıntı için ilgili alt klasördeki `README.md` dosyasına bakın.