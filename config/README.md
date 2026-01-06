# 📂 Ayarlar Klasörü (Config)

Projenin merkezi yapılandırma dosyalarını içerir.

## 📄 Dosyalar

### 1. `settings.py`

* **İşlev:** Tüm sabitleri, ortam değişkenlerini ve model parametrelerini içerir.
*   **İçerik:**
    * **Veritabanı:** Bağlantı ayarları (DB_HOST, DB_USER, ...).
    * **Yollar:** Proje kök dizini, YOLO model yolları, Sınıflandırıcı dosya yolu.
    * **Modeller:**
        * `ENCODING_MODEL`: "facenet" veya "dlib".
        * `FACE_DETECTION_MODEL`: "yolo", "hog", "cnn".
    * **Eğitim Ayarları (`TRAINING_CONFIG`):**
        * `classifier`: Sınıflandırıcı tipi ("mlp", "xgboost") ve hiperparametreleri (hidden_layers, learning_rate,
          n_estimators...).
        * `dlib`: Jitter sayısı.
        * `detection`: Upsample oranları.
    * **Arayüz:** `UI_COLORS` teması.
    * **Performans:** `PROCESSING_SCALE` ve `TRAINING_IMAGE_SIZE`.