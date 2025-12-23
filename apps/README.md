# 📂 Uygulamalar Klasörü (Apps)

Bu klasör, kullanıcının doğrudan çalıştırabileceği yürütülebilir uygulamaları içerir.

## 📄 Dosyalar

### 1. `image_app.py`
*   **İşlev:** Sabit resimlerden yüz tanıma uygulaması.
*   **Kullanım:** Diskten bir resim okur, yüzleri tespit eder ve kişilerin kimliğini belirlemek için veritabanıyla karşılaştırır.

### 2. `video_app.py`
*   **İşlev:** Canlı videodan (Webcam) veya video dosyasından yüz tanıma uygulaması.
*   **Kullanım:** Video akışını görüntüler ve tespit edilen yüzlerin etrafına kareler çizerek kişinin adını ve durumunu (mevcut/yok) yazar.

### 3. `training_app.py`
*   **İşlev:** Sistemi yeni yüzler için eğitme aracı.
*   **Mekanizma:**
    1. `data/TrainingImages` klasöründen resimleri okur.
    2. Her kişi için yüz kodlamasını (Encoding) çıkarır.
    3. Verileri (İsim + Kodlama) PostgreSQL veritabanına kaydeder.