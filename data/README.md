# 📂 Veri Klasörü (Data)

Bu klasör, sistemin dayandığı resimleri saklamak için ayrılmıştır.

## 📂 Alt Klasörler

### 1. `TrainingImages/`
*   **Amaç:** Sisteme kaydetmek istediğiniz kişilerin resimlerini buraya koyun.
*   **İsimlendirme:** Dosyayı kişinin adıyla isimlendirmeniz önerilir (örneğin: `ahmed.jpg`, `sara.png`). `training_app.py`, dosya adını öğrenci adı olarak kullanacaktır.

### 2. `TestImages/`
*   **Amaç:** `benchmarks` betiklerini kullanarak sistemin doğruluğunu ve hızını test etmek için rastgele resimler koyun.
*   **Sorun Çözümü:** Eğer `Error: No images found` hatası alırsanız, bu klasör boş demektir. Testin çalışması için buraya bazı resimler (.jpg/.png) ekleyin.