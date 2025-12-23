import sys
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# --- Çekirdek modülleri içe aktarmak için yol ayarı ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from core.database import Database
from core.detector import detect_faces
import face_recognition
import config.settings as settings

# --- Uygulama Yaşam Döngüsü (Başlatma & Kapatma) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Sunucu Başlatılıyor... Veritabanı Havuzuna Bağlanılıyor...")
    Database.initialize_pool()
    yield
    print("🛑 Sunucu Durduruluyor... Veritabanı Havuzu Kapatılıyor...")
    Database.close_all()

# FastAPI Uygulamasını Başlat
app = FastAPI(title="Akıllı Okul Yüz Tanıma API", lifespan=lifespan)

# --- CORS Middleware Ekle ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" herkesin erişimine izin verir (Mobil, Web vb.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------------

# --- Yardımcı Fonksiyonlar ---

def get_face_encodings(image_rgb):
    """
    YOLO (GPU) kullanarak yüzleri tespit eder ve Dlib (GPU) kullanarak 128D kodlamalar (encodings) oluşturur.
    """
    # 1. YOLO kullanarak yüzleri tespit et (GPU üzerinde En Hızlı & Hassas)
    locations = detect_faces(
        image_rgb, 
        model_name="yolo", 
        confidence=0.5, 
        yolo_weights=settings.YOLO_WEIGHTS
    )
    
    if not locations:
        return []

    # 2. Dlib kullanarak kodlamaları oluştur
    # Not: HOG ile tekrar tespit yapmamak için 'locations' parametresi verilir
    encodings = face_recognition.face_encodings(image_rgb, locations)
    return encodings

def identify_student(encoding):
    """
    En yakın yüz eşleşmesini bulmak için PostgreSQL veritabanını sorgular.
    Öklid mesafesi (pgvector <-> operatörü) kullanır.
    """
    vec_str = str(encoding.tolist())
    try:
        with Database.get_conn() as conn:
            with conn.cursor() as cursor:
                # 0.5 mesafeden (Eşik Değeri) daha yakın olan en iyi eşleşmeyi bul
                query = """
                SELECT p.name, p.id 
                FROM people p 
                JOIN face_encodings f ON p.id = f.person_id 
                WHERE f.encoding <-> %s < 0.5 
                ORDER BY f.encoding <-> %s ASC 
                LIMIT 1;
                """
                cursor.execute(query, (vec_str, vec_str))
                result = cursor.fetchone()
                
                if result:
                    return {"name": result[0], "id": result[1], "status": "Mevcut"}
                
                return {"name": "Bilinmiyor", "id": None, "status": "Bilinmiyor"}
                
    except Exception as e:
        print(f"❌ Veritabanı Hatası: {e}")
        return None

# --- API Uç Noktaları (Endpoints) ---

@app.get("/")
def home():
    return {"message": "Akıllı Okul API, GPU Desteği ile Çalışıyor! 🚀"}

@app.post("/scan-attendance")
async def scan_attendance(file: UploadFile = File(...)):
    """
    Flutter Uygulaması tarafından çağrılan uç nokta.
    1. Bir resim dosyası alır.
    2. Yüzleri tespit eder.
    3. Öğrencileri tanımlar.
    4. Mevcut öğrencilerin JSON listesini döndürür.
    """
    try:
        # 1. Resmi Oku ve Çöz (Decode)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Geçersiz resim dosyası")

        # 2. BGR'den RGB'ye çevir (dlib/face_recognition için gerekli)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Resmi İşle
        encodings = get_face_encodings(rgb_img)
        
        results = []
        if len(encodings) > 0:
            print(f"📸 {len(encodings)} yüz bulundu. Kimlik tespiti yapılıyor...")
            for encoding in encodings:
                student = identify_student(encoding)
                if student:
                    results.append(student)
        else:
            print("⚠️ Resimde yüz bulunamadı.")

        # 4. Yanıtı Döndür
        return {
            "success": True,
            "total_faces": len(encodings),
            "students": results
        }

    except Exception as e:
        print(f"❌ İstek işlenirken hata oluştu: {e}")
        return {"success": False, "error": str(e)}

# --- Giriş Noktası ---
if __name__ == "__main__":
    # '0.0.0.0' ana bilgisayarı, ağdaki diğer cihazlardan (Mobil gibi) erişime izin verir
    uvicorn.run(app, host="0.0.0.0", port=8000)