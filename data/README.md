# 📂 Veri Klasörü (Data)

Bu klasör, sistemin eğitimi ve testi için kullanılan resim verilerini saklar.

## 📂 Alt Klasörler

### 1. `TrainingImages/`

* **Amaç:** Sisteme kaydetmek (Enrollment) istediğiniz kişilerin resimlerini içerir.
* **Yapı:** Her kişi için ayrı bir klasör oluşturulmalıdır.
  ```text
  TrainingImages/
  ├── Ali/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── Ayse/
      ├── img1.jpg
      └── ...
  ```
* **Öneri:** Her kişi için en az 10-20 farklı açıdan çekilmiş resim kullanılması, sınıflandırıcının (Classifier)
  doğruluğunu artırır.

### 2. `TestImages/` (Opsiyonel)

* **Amaç:** Sistemin performansını manuel olarak test etmek için kullanılan karışık resimler.
* **Kullanım:** `image_app.py` ile bu klasörden rastgele resimler seçip sistemin kimi tanıdığını görebilirsiniz.