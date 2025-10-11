import psycopg2
import os
from dotenv import load_dotenv
import json

load_dotenv()

def get_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )
    return conn

def insert_video(cur, filename, storage_path, fps=None, resolution=None, width=None, height=None, duration_sec=None):
    cur.execute("""
        INSERT INTO videos (filename, storage_path, fps, resolution, width, height, duration_sec)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        RETURNING id;
    """, (filename, storage_path, fps, resolution, width, height, duration_sec))
    return cur.fetchone()[0]

def insert_landmarks(cur, video_id, frame_data, person_id=None):
    for f in frame_data:
        cur.execute("""
            INSERT INTO landmarks (video_id, person_id, frame_number, timestamp_sec, landmarks)
            VALUES (%s,%s,%s,%s,%s)
        """, (video_id, person_id, f["frame_index"], f["timestamp"], json.dumps(f["landmarks"])))

def insert_annotations(cur, video_id, annotations_json):
    for ann in annotations_json:
        cur.execute("""
            INSERT INTO annotations (video_id, label, start_time_sec, end_time_sec, extra_data)
            VALUES (%s,%s,%s,%s,%s)
        """, (video_id, ann["label"], ann["start"], ann["end"], json.dumps(ann)))
