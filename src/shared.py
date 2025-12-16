import os
import sqlite3
import pickle
import cv2
import numpy as np
from datetime import datetime

# --- Configuration & Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "TrainingImages")
DB_FILE = os.path.join(PROJECT_ROOT, "face_encodings.db")
RECORDS_FILE = os.path.join(PROJECT_ROOT, "records.csv")

# Define the current model name being used by the system
# If you switch libraries later (e.g., to FaceNet), change this constant or pass it dynamically
CURRENT_MODEL_NAME = "dlib_face_recognition"

# --- Database Functions ---
def init_database():
    """Initializes the database with the new relational schema."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 1. Table for People (Stores unique identities)
    c.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT
        )
    ''')

    # 2. Table for Encodings (Stores vector data linked to a person and a model)
    c.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            encoding BLOB NOT NULL,
            created_at TEXT,
            FOREIGN KEY(person_id) REFERENCES people(id)
        )
    ''')

    # 3. Table for Records (Logs)
    c.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def load_encodings_from_db(model_name=CURRENT_MODEL_NAME):
    """
    Loads face encodings and labels for a specific model.
    Returns: (encodings_list, labels_list)
    """
    encodings = []
    labels = []
    if not os.path.exists(DB_FILE):
        return encodings, labels
        
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Join people and face_encodings to get name + encoding for the specific model
        query = '''
            SELECT p.name, e.encoding 
            FROM people p 
            JOIN face_encodings e ON p.id = e.person_id 
            WHERE e.model_name = ?
        '''
        c.execute(query, (model_name,))
        rows = c.fetchall()
        conn.close()
        
        if rows:
            labels = [row[0] for row in rows]
            encodings = [pickle.loads(row[1]) for row in rows]
            
    except Exception as e:
        print(f"Error loading database: {e}")
        
    return encodings, labels

def save_encoding_to_db(name, encoding, model_name=CURRENT_MODEL_NAME):
    """
    Saves a face encoding linked to a person.
    1. Checks if person exists in 'people'. If not, creates them.
    2. Inserts the encoding into 'face_encodings' linked to that person.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Get or Create Person ID
        c.execute("SELECT id FROM people WHERE name = ?", (name,))
        result = c.fetchone()
        
        if result:
            person_id = result[0]
        else:
            c.execute("INSERT INTO people (name, created_at) VALUES (?, ?)", (name, now))
            person_id = c.lastrowid

        # 2. Insert Encoding
        c.execute('''
            INSERT INTO face_encodings (person_id, model_name, encoding, created_at) 
            VALUES (?, ?, ?, ?)
        ''', (person_id, model_name, pickle.dumps(encoding), now))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

def log_recognition(name):
    """Logs a recognition event to the database and CSV."""
    capture_date = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    
    # Log to DB
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO records (name, date, time) VALUES (?, ?, ?)", (name, capture_date, now))
        conn.commit()
        conn.close()
    except Exception:
        pass

    # Log to CSV (Backup)
    try:
        if not os.path.isfile(RECORDS_FILE):
            with open(RECORDS_FILE, "w", encoding="utf-8") as f:
                f.write("Name,Date,Time\n")
        with open(RECORDS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{name},{capture_date},{now}\n")
    except Exception:
        pass

# --- Utility Functions ---
def detect_gpu():
    """Checks if CUDA/GPU is available for dlib."""
    try:
        import dlib
        if hasattr(dlib, 'cuda') and hasattr(dlib.cuda, 'get_num_devices'):
            return int(dlib.cuda.get_num_devices()) > 0
        if hasattr(dlib, 'DLIB_USE_CUDA'):
            return bool(dlib.DLIB_USE_CUDA)
    except:
        pass
    return False

def resize_image_aspect_ratio(image, target_size=(800, 800)):
    """Resizes an image maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    return cv2.resize(image, (int(w*scale), int(h*scale)))
