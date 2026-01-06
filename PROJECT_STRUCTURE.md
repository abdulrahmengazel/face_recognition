# 🧠 Detaylı Sistem Çalışma Mimarisi (Detailed System Architecture)

Bu belge, Yüz Tanıma Sistemi'nin (Smart School Face Recognition) arka planında çalışan algoritmaları, veri akışlarını ve
karar mekanizmalarını **derinlemesine** teknik detaylarla açıklar.

---

## 1. 📂 Eğitim Aşaması (Training Phase) - "Veri Toplama ve İşleme"

Bu aşama, ham piksellerden oluşan görüntülerin, yapay zekanın anlayabileceği matematiksel vektörlere dönüştürülüp *
*PostgreSQL** veritabanında yapılandırılmış bir şekilde saklanması sürecidir.

### 🛠️ Teknik Süreç Detayları:

#### A. Görüntü Okuma ve Ön İşleme (Preprocessing)

1. **Dosya Okuma:** `cv2.imdecode` kullanılarak dosya sisteminden resimler okunur (Türkçe karakterli dosya yollarını
   desteklemek için).
2. **Renk Dönüşümü:** OpenCV resimleri varsayılan olarak **BGR** (Blue-Green-Red) formatında okur. Modellerin doğru
   çalışması için bu, **RGB** formatına dönüştürülür.
3. **Yeniden Boyutlandırma (Resizing):** İşlem yükünü optimize etmek için resimler, en-boy oranı korunarak `800x800`
   piksel sınırlarına ölçeklenir.

#### B. Yüz Tespiti (Face Detection - YOLOv8)

* **Model:** `yolov8l-face.pt` (Large model) kullanılır.
* **Algoritma:** YOLO (You Only Look Once), resmi tek seferde tarar ve yüzlerin bulunduğu koordinatları (
  `Bounding Box: [x1, y1, x2, y2]`) ve bir **Güven Skoru (Confidence Score)** döndürür.
* **Filtreleme:** Sadece güven skoru `%50` (0.5) üzerinde olan yüzler işleme alınır.

#### C. Özellik Çıkarımı (Feature Extraction / Embedding)
Tespit edilen yüz bölgesi kesilir ve seçilen modele gönderilir:

* **FaceNet (Google):** Yüzü 128 boyutlu bir hiper-küre (hypersphere) üzerinde bir noktaya eşler.
* **Dlib (ResNet):** Yüzü 128 boyutlu bir vektöre dönüştürür.
* **Jittering (Sadece Dlib):** Resim rastgele bozulmalara (döndürme, kaydırma) uğratılarak 10 kez işlenir ve ortalaması
  alınır. Bu, gürültüye karşı dayanıklılığı artırır.
* **Sonuç:** Her yüz için `[0.123, -0.45, 0.88, ...]` şeklinde 128 adet ondalıklı sayıdan oluşan bir liste elde edilir.

#### D. Veritabanı Yönetimi (PostgreSQL + pgvector)

1. **Kişi Kaydı:** Kişi ismi `people` tablosuna eklenir ve benzersiz bir `person_id` (Primary Key) üretilir.
2. **Temizlik:** Kişinin seçilen model (örn: FaceNet) için daha önce kaydedilmiş eski vektörleri silinir (Duplicate
   önleme).
3. **Vektör Kaydı:** Her bir resimden çıkarılan vektör, `face_encodings` tablosuna **ayrı bir satır** olarak eklenir.
    * *Neden?* Ortalama (Mean) almak yerine tüm varyasyonları saklamak, sınıflandırıcının kişinin farklı hallerini (
      gözlüklü, sakallı, yandan) öğrenmesini sağlar.

---

## 2. 🧠 Sınıflandırıcı Aşaması (Classifier Phase) - "Model Eğitimi"

Bu aşama, veritabanındaki binlerce vektörü analiz ederek, hangi vektörün kime ait olduğunu öğrenen bir "Yapay Beyin"
oluşturma sürecidir.

### 🛠️ Teknik Süreç Detayları:

#### A. Veri Hazırlığı (Data Preparation)

1. **Fetch:** Veritabanından `(İsim, Vektör)` çiftleri çekilir.
2. **Label Encoding:** İsimler (String), makine öğrenmesi için tamsayılara (Integer) çevrilir.
    * `Ahmet` -> `0`, `Mehmet` -> `1`, `Zeynep` -> `2`.
3. **Scaling (Kritik Adım):** `StandardScaler` kullanılır.
    * Her özellik (feature) için: `z = (x - u) / s` formülü uygulanır.
    * Bu işlem, verilerin ortalamasını 0, varyansını 1 yapar. Sinir ağlarının (Neural Networks) yakınsaması (
      convergence) için zorunludur.

#### B. Model Eğitimi: MLP Classifier (Multi-Layer Perceptron)

* **Mimari:** Derin Yapay Sinir Ağı.
    * **Girdi Katmanı:** 128 Nöron (Yüz vektörü).
    * **Gizli Katmanlar:** 1024 -> 512 -> 256 Nöron (ReLU aktivasyon fonksiyonu ile).
    * **Çıktı Katmanı:** Sınıf sayısı kadar nöron (Softmax aktivasyonu ile olasılık dağılımı).
* **Optimizasyon:** `Adam` algoritması, ağırlıkları güncelleyerek hatayı (Log-Loss) minimize eder.
* **Avantajı:** Karmaşık, doğrusal olmayan ilişkileri çok iyi öğrenir ve yüz tanıma için en kararlı sonuçları verir.

#### C. Doğrulama (Validation)

* Veri seti `%90 Eğitim`, `%10 Doğrulama` olarak ayrılır (eğer yeterli veri varsa).
* Eğitim sırasında modelin performansı (Loss değeri) canlı olarak izlenir.

#### D. Serileştirme (Serialization)

* Eğitilen Model + Label Encoder + Scaler, `pickle` kütüphanesi ile tek bir `.pkl` dosyasına paketlenir. Bu dosya,
  uygulamanın "beyni"dir.

---

## 3. 👁️ Tanıma ve İşleme Aşaması (Inference Phase) - "Canlı Analiz"

Bu aşama, sistemin gerçek dünyada çalıştığı, kameradan gelen görüntüyü saniyeler içinde analiz edip kimlik tespiti
yaptığı andır.

### 🛠️ Teknik Süreç Detayları:

#### A. Pipeline (İşlem Hattı)

1. **Frame Capture:** Kameradan görüntü alınır.
2. **Detection:** YOLOv8 ile yüzler bulunur.
3. **Encoding:** Yüzlerden 128d vektör çıkarılır.
4. **Preprocessing:** Vektör, eğitimde kaydedilen `Scaler` ile normalize edilir.

#### B. Tahmin (Prediction)

Model (MLP), normalize edilmiş vektörü alır ve bir olasılık dizisi döndürür:

* `[0.01, 0.98, 0.01]` -> Bu, %98 ihtimalle 1. indeksteki kişi demektir.
* **Güven Eşiği (Threshold):** Eğer en yüksek olasılık `%30` (0.3) altındaysa, sonuç reddedilir ve "BİLİNMİYOR" yazılır.

#### C. Yüz Takibi ve Stabilizasyon (Face Tracking & Smoothing)
Video akışındaki titremeyi önlemek için özel bir algoritma çalışır:

1. **Eşleştirme (Matching):**
    * Şu anki karedeki yüzün merkezi ile bir önceki karedeki yüzlerin merkezleri arasındaki **Öklid Mesafesi**
      hesaplanır.
    * Mesafe kısaysa (örn: < 100 piksel), bu yüzlerin aynı kişiye ait olduğu varsayılır ve aynı `Face ID` atanır.

2. **Oylama (Voting):**
    * Her `Face ID` için son 8 karenin tahmin sonuçları bir hafızada (Deque) tutulur.
    * Örn: `['Ali', 'Ali', 'Bilinmiyor', 'Ali', 'Mehmet', 'Ali', 'Ali', 'Ali']`
    * **Mod (Mode) Hesabı:** En çok tekrar eden isim ('Ali') ekrana yazdırılır.
    * Bu sayede anlık hatalı tahminler (glitch) filtrelenir ve ekrandaki isim sabit kalır.

#### D. Fallback (Yedek Plan)

* Eğer `classifier.pkl` dosyası bulunamazsa veya bozuksa, sistem otomatik olarak **Veritabanı Arama Moduna** (Legacy
  Mode) geçer.
* Bu modda, vektör ile veritabanındaki tüm kayıtlar arasındaki mesafe tek tek hesaplanır (Daha yavaş ama güvenilir).

---

### 🚀 Performans Özeti

| Modül      | Teknoloji          | Görevi              | Hız            |
|:-----------|:-------------------|:--------------------|:---------------|
| **Tespit** | YOLOv8 Large       | Yüzü bulma          | ~30-50ms (GPU) |
| **Vektör** | FaceNet            | Yüzü sayıya çevirme | ~100-200ms     |
| **Karar**  | MLP Neural Network | Kimliği bulma       | **< 1ms**      |
| **Takip**  | Euclidean Tracker  | Yüzü izleme         | **< 1ms**      |

Bu mimari, sistemin binlerce kişiyi tanısa bile gerçek zamanlı (Real-Time) çalışabilmesini sağlar.
