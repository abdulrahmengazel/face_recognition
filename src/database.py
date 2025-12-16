import psycopg2
from psycopg2 import pool, extras
import numpy as np
import os
from contextlib import contextmanager
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

# Tuning HNSW Index for Speed vs Accuracy
INDEX_M = 16
INDEX_EF = 64

class Database:
    _pool = None

    @classmethod
    def initialize_pool(cls):
        if cls._pool is None:
            try:
                cls._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10, 
                    host=DB_HOST,
                    port=DB_PORT,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASS
                )
                print("Database Connection Pool Initialized.")
            except Exception as e:
                print(f"Error initializing connection pool: {e}")

    @classmethod
    @contextmanager
    def get_conn(cls):
        """Context manager to get a connection from the pool and return it automatically."""
        if cls._pool is None:
            cls.initialize_pool()
        
        conn = cls._pool.getconn()
        try:
            yield conn
        finally:
            cls._pool.putconn(conn)

    @classmethod
    def close_all(cls):
        if cls._pool:
            cls._pool.closeall()
            print("Connection Pool Closed.")

    @staticmethod
    def init_tables():
        """Creates tables and indexes with optimized settings."""
        with Database.get_conn() as conn:
            with conn.cursor() as cursor:
                # 1. Enable pgvector
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # 2. Create Tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS people (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                ''')

                # Updated table definition to support multiple encoding models
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_encodings (
                        id SERIAL PRIMARY KEY,
                        person_id INTEGER NOT NULL,
                        model_name TEXT NOT NULL,
                        encoding vector(128), -- Changed to NULLABLE
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT fk_person FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                    );
                ''')
                
                # Add FaceNet column if it doesn't exist
                cursor.execute('''
                    ALTER TABLE face_encodings 
                    ADD COLUMN IF NOT EXISTS encoding_facenet vector(128);
                ''')
                
                # IMPORTANT: Make the old 'encoding' column nullable to allow FaceNet-only records
                cursor.execute('''
                    ALTER TABLE face_encodings 
                    ALTER COLUMN encoding DROP NOT NULL;
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS records (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL
                    );
                ''')

                # 3. Create Optimized Indexes (HNSW)
                # Index for dlib
                cursor.execute("DROP INDEX IF EXISTS face_encodings_idx;")
                cursor.execute(f'''
                    CREATE INDEX IF NOT EXISTS face_encodings_idx ON face_encodings 
                    USING hnsw (encoding vector_l2_ops) 
                    WITH (m = {INDEX_M}, ef_construction = {INDEX_EF});
                ''')
                
                # Index for FaceNet
                cursor.execute("DROP INDEX IF EXISTS face_encodings_facenet_idx;")
                cursor.execute(f'''
                    CREATE INDEX IF NOT EXISTS face_encodings_facenet_idx ON face_encodings 
                    USING hnsw (encoding_facenet vector_l2_ops) 
                    WITH (m = {INDEX_M}, ef_construction = {INDEX_EF});
                ''')
                
                conn.commit()
                print("Database Tables & Indexes Optimized (Multi-Model Support).")

# --- ADAPTERS ---
def adapt_numpy_array(numpy_array):
    return psycopg2.extensions.AsIs(f"'{numpy_array.tolist()}'")

psycopg2.extensions.register_adapter(np.ndarray, adapt_numpy_array)
