# 📂 Çekirdek Klasörü (Core)

Uygulamaların dayandığı temel iş mantığını (Business Logic) içerir.

## 📄 Dosyalar

### 1. `database.py`

* **İşlev:** PostgreSQL veritabanı bağlantısını ve şema yönetimini sağlar.
*   **Görevler:**
    *   Bağlantı havuzu (Connection Pool) oluşturma.
    * Tabloları (`people`, `face_encodings`) ve indeksleri (HNSW) otomatik oluşturma.
    * Çoklu model desteği için veritabanı şemasını yönetme (dlib ve facenet sütunları).

### 2. `detector.py`

* **İşlev:** Yüz tespit algoritmaları için birleşik bir arayüz sağlar.
*   **Görevler:**
    * **YOLOv8:** `ultralytics` kütüphanesini kullanarak hızlı ve hassas tespit.
    * **HOG & CNN:** `face_recognition` kütüphanesi üzerinden alternatif tespit yöntemleri.
    * `detect_faces` fonksiyonu ile modelden bağımsız olarak yüz koordinatlarını (top, right, bottom, left) döndürür.