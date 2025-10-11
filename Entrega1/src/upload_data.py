import os, json
from db_utils import get_connection, insert_video, insert_landmarks, insert_annotations
from mediapipe_extract import extract_landmarks

DATA_PATH = "../Entrega1/data"

def process_video(video_path, annotations_path=None):
    filename = os.path.basename(video_path)
    storage_path = f"local://{video_path}"   # simulando storage
    frames, fps = extract_landmarks(video_path)

    conn = get_connection()
    cur = conn.cursor()

    video_id = insert_video(cur, filename, storage_path, fps=fps)
    insert_landmarks(cur, video_id, frames)

    if annotations_path and os.path.exists(annotations_path):
        with open(annotations_path, "r", encoding="utf8") as f:
            annotations_json = json.load(f)
        insert_annotations(cur, video_id, annotations_json)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ {filename} insertado correctamente con {len(frames)} frames")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python upload_data.py path_al_video [path_annotations]")
        exit()
    video = sys.argv[1]
    annotations = sys.argv[2] if len(sys.argv) > 2 else None
    process_video(video, annotations)
