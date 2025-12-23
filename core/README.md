# 📂 Çekirdek Klasörü (Core)

Uygulamaların dayandığı temel iş mantığını (Business Logic) içerir.

## 📄 Dosyalar

### 1. `database.py`
*   **İşlev:** PostgreSQL veritabanı bağlantısını yönetir.
*   **Görevler:**
    *   Bağlantı havuzu (Connection Pool) oluşturma.
    *   Sorguları yürütme (öğrenci ekleme, yüz arama).
    *   `pgvector` işlemlerini yönetme.

### 2. `detector.py`
*   **İşlev:** Yüz tespit algoritmaları için bir sarmalayıcı (Wrapper).
*   **Görevler:**
    *   YOLO modelini yükleme.
    *   Resmi kabul eden ve yüz koordinatlarını döndüren birleşik bir `detect_faces` fonksiyonu sağlama.
    *   Tespit öncesi gerekli resim dönüşümlerini yapma.