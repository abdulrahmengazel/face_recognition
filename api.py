import sys
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# --- 1. Setup Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from core.database import Database
from core.detector import detect_faces
import face_recognition
import config.settings as settings


# --- 2. Database Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server Starting... Connecting to Database Pool...")
    try:
        Database.initialize_pool()
        yield
    except Exception as e:
        print(f"❌ Startup Error: {e}")
    finally:
        print("🛑 Server Stopping... Closing Database Pool...")
        Database.close_all()


# --- 3. Initialize App ---
app = FastAPI(title="Smart School Face ID API", lifespan=lifespan)

# --- 4. CORS Setup (مهم جداً للمتصفح) ---
# هذا الجزء يسمح للمتصفح بالاتصال دون حظر (Cross-Origin Error)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # السماح لأي مصدر (موبايل، متصفح، سيرفر آخر)
    allow_credentials=True,
    allow_methods=["*"],  # السماح بكل أنواع الطلبات (GET, POST, etc)
    allow_headers=["*"],  # السماح بكل الهيدرز
)


# --- 5. Helper Functions ---
def get_face_encodings(image_rgb):
    """Detect faces (YOLO) and encode (Dlib)."""
    locations = detect_faces(
        image_rgb,
        model_name="yolo",
        confidence=0.5,
        yolo_weights=settings.YOLO_WEIGHTS
    )
    if not locations:
        return []
    return face_recognition.face_encodings(image_rgb, locations)


def identify_student(encoding):
    """Identify student and check Late/Present status."""
    vec_str = str(encoding.tolist())
    try:
        with Database.get_conn() as conn:
            with conn.cursor() as cursor:
                # البحث عن أقرب وجه
                query = """
                        SELECT p.name, p.id
                        FROM people p
                                 JOIN face_encodings f ON p.id = f.person_id
                        WHERE f.encoding <-> %s < 0.5
                        ORDER BY f.encoding <-> %s ASC
                        LIMIT 1; \
                        """
                cursor.execute(query, (vec_str, vec_str))
                result = cursor.fetchone()

                if result:
                    # منطق التأخير (Late Logic)
                    now = datetime.now()
                    cutoff_time = now.replace(hour=8, minute=30, second=0, microsecond=0)
                    status = "Late" if now > cutoff_time else "Present"

                    return {"name": result[0], "id": result[1], "status": status}

                return {"name": "Unknown", "id": None, "status": "Unknown"}
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return None


# --- 6. API Endpoints ---

@app.get("/")
def home():
    """Test Endpoint for Browser"""
    return {
        "message": "Smart School API is Online! 🟢",
        "access": "Accessible from Browser & Mobile",
        "time": datetime.now().strftime("%H:%M:%S")
    }


@app.post("/api/attendance/scan")
async def scan_attendance(file: UploadFile = File(...)):
    """Main Attendance Endpoint"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = get_face_encodings(rgb_img)

        results = []
        if len(encodings) > 0:
            print(f"📸 Found {len(encodings)} faces.")
            for encoding in encodings:
                student = identify_student(encoding)
                if student:
                    results.append(student)

        return {
            "success": True,
            "total_faces": len(encodings),
            "students": results
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}


# --- 7. Run Server (مهم للشبكة) ---
if __name__ == "__main__":
    # host="0.0.0.0" تعني: اسمع للكل (Localhost + Wi-Fi IP)
    uvicorn.run(app, host="0.0.0.0", port=8000)