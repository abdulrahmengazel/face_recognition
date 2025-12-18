import sys
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# --- Setup Paths to import core modules ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from core.database import Database
from core.detector import detect_faces
import face_recognition
import config.settings as settings

# --- Application Lifespan (Startup & Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server Starting... Connecting to Database Pool...")
    Database.initialize_pool()
    yield
    print("🛑 Server Stopping... Closing Database Pool...")
    Database.close_all()

# Initialize FastAPI App
app = FastAPI(title="Smart School Face ID API", lifespan=lifespan)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" تعني السماح للجميع (موبايل، ويب، أي شيء)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------------

# --- Helper Functions ---

def get_face_encodings(image_rgb):
    """
    Detects faces using YOLO (GPU) and generates 128D encodings using Dlib (GPU).
    """
    # 1. Detect faces using YOLO (Fastest & Accurate on GPU)
    locations = detect_faces(
        image_rgb, 
        model_name="yolo", 
        confidence=0.5, 
        yolo_weights=settings.YOLO_WEIGHTS
    )
    
    if not locations:
        return []

    # 2. Generate encodings using Dlib
    # Note: 'locations' are passed to avoid re-detecting with HOG
    encodings = face_recognition.face_encodings(image_rgb, locations)
    return encodings

def identify_student(encoding):
    """
    Queries the PostgreSQL database to find the nearest face match.
    Uses Euclidean distance (pgvector <-> operator).
    """
    vec_str = str(encoding.tolist())
    try:
        with Database.get_conn() as conn:
            with conn.cursor() as cursor:
                # Find the closest face vector with distance < 0.5 (Threshold)
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
                    return {"name": result[0], "id": result[1], "status": "Present"}
                
                return {"name": "Unknown", "id": None, "status": "Unknown"}
                
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return None

# --- API Endpoints ---

@app.get("/")
def home():
    return {"message": "Smart School API is Running with GPU Support! 🚀"}

@app.post("/api/attendance/scan")
async def scan_attendance(file: UploadFile = File(...)):
    """
    Endpoint called by Flutter App.
    1. Receives an image file.
    2. Detects faces.
    3. Identifies students.
    4. Returns JSON list of present students.
    """
    try:
        # 1. Read and Decode Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 2. Convert BGR to RGB (required for dlib/face_recognition)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Process Image
        encodings = get_face_encodings(rgb_img)
        
        results = []
        if len(encodings) > 0:
            print(f"📸 Found {len(encodings)} faces. Identifying...")
            for encoding in encodings:
                student = identify_student(encoding)
                if student:
                    results.append(student)
        else:
            print("⚠️ No faces found in the image.")

        # 4. Return Response
        return {
            "success": True,
            "total_faces": len(encodings),
            "students": results
        }

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return {"success": False, "error": str(e)}

# --- Entry Point ---
if __name__ == "__main__":
    # Host '0.0.0.0' allows access from other devices (like Mobile) on the network
    uvicorn.run(app, host="0.0.0.0", port=8000)