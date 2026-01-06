# 📂 Uygulamalar Klasörü (Apps)

Bu klasör, kullanıcının doğrudan çalıştırabileceği yürütülebilir uygulamaları içerir.

## 📄 Dosyalar

### 1. `image_app.py`
*   **İşlev:** Sabit resimlerden yüz tanıma uygulaması.
* **Kullanım:** Diskten bir resim okur, yüzleri tespit eder ve eğitilmiş **MLP/XGBoost Sınıflandırıcıyı** kullanarak
  kişilerin kimliğini belirler.
* **Özellikler:**
    - Yüz tespiti için YOLOv8 kullanır.
    - Sınıflandırıcı dosyası (`classifier.pkl`) yoksa otomatik olarak veritabanı aramasına geçer.

### 2. `video_app.py`

* **İşlev:** Canlı videodan (Webcam) yüz tanıma uygulaması.
* **Kullanım:** Video akışını görüntüler ve tespit edilen yüzlerin etrafına kareler çizerek kişinin adını yazar.
* **Özellikler:**
    - **Face Tracking:** Yüzleri kareler arasında takip ederek titremeyi önler.
    - **Stabilizasyon:** Sonuçları son birkaç kareye göre ortalayarak (Voting) daha kararlı bir tanıma sağlar.
    - Gerçek zamanlı performans için optimize edilmiştir.

### 3. `training_app.py`

* **İşlev:** Sistemi yeni yüzler için eğitme ve sınıflandırıcıyı güncelleme aracı.
*   **Mekanizma:**
    1. `data/TrainingImages` klasöründen resimleri okur.
    2. Seçilen modele (Dlib/FaceNet) göre yüz kodlamasını (Encoding) çıkarır.
    3. Verileri (İsim + Kodlama) PostgreSQL veritabanına kaydeder (Her resim için ayrı kayıt).
    4. **Sınıflandırıcı Eğitimi:** Veritabanındaki tüm verileri kullanarak bir **MLP** veya **XGBoost** modeli eğitir ve
       `assets/classifier.pkl` olarak kaydeder.
* **Özellikler:**
    - Veri normalizasyonu (StandardScaler).
    - Eğitim sırasında doğrulama (Validation) seti ile performans izleme.
    - Sadece sınıflandırıcıyı yeniden eğitme (Retrain Classifier Only) seçeneği.