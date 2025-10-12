# file: load_video_info_to_supabase.py
# Requisitos:
#   pip install -r requirements.txt
#
# .env:
#   SUPABASE_URL=https://xxxxx.supabase.co
#   SUPABASE_ANON_KEY=eyJhbGciOi...
#   VIDEOS_DIR=../../videos
#   MAX_SAMPLES=60
#   SAMPLE_STRIDE_SEC=0
#   INSERT_LANDMARKS=1
#   LANDMARKS_SAMPLE_EVERY=1
#   LANDMARKS_MAX_FRAMES=0

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
import cv2
import numpy as np
import math
from mediapipe_extract import extract_landmarks 

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
VIDEOS_DIR = os.getenv("VIDEOS_DIR", "../../videos")
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "60"))
SAMPLE_STRIDE_SEC = float(os.getenv("SAMPLE_STRIDE_SEC", "0"))
INSERT_LANDMARKS = os.getenv("INSERT_LANDMARKS", "1") not in ("0", "false", "False", "")
LANDMARKS_SAMPLE_EVERY = int(os.getenv("LANDMARKS_SAMPLE_EVERY", "1"))
LANDMARKS_MAX_FRAMES = int(os.getenv("LANDMARKS_MAX_FRAMES", "0"))

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("❌ Faltan SUPABASE_URL o SUPABASE_ANON_KEY en el .env")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MOV", ".MP4"}


def get_duration(cap, fps):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return frame_count / fps if fps and fps > 0 else 0.0


def sample_times(duration_sec: float, fps: float) -> list[float]:
    if duration_sec <= 0:
        return [0.0]
    if SAMPLE_STRIDE_SEC > 0:
        times = np.arange(0, duration_sec, SAMPLE_STRIDE_SEC, dtype=float).tolist()
    else:
        n = max(1, min(MAX_SAMPLES, int(math.ceil(duration_sec))))
        n = min(n, MAX_SAMPLES)
        times = [0.0] if n == 1 else np.linspace(0, max(0.0, duration_sec - 1e-3), n, dtype=float).tolist()
    if times and times[0] > 0.05:
        times = [0.0] + times
    return times[:MAX_SAMPLES]


def video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = get_duration(cap, fps)

    lighting_vals = []
    for t in sample_times(duration_sec, fps):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lighting_vals.append(float(np.mean(gray)))
    cap.release()

    lighting = float(np.mean(lighting_vals)) if lighting_vals else None
    resolution = f"{width}x{height}" if width and height else None
    return fps, width, height, duration_sec, lighting, resolution


def find_videos(base: Path):
    if not base.exists():
        print(f"⚠️  Carpeta {base} no existe.")
        return
    for action_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        action = action_dir.name
        for file in action_dir.rglob("*"):
            if file.is_file() and file.suffix in ALLOWED_EXT:
                yield action, file


def insert_video_row_return_id(filename: str,
                               label: str,
                               fps=None, width=None, height=None,
                               duration_sec=None, lighting=None, resolution=None) -> str:
    payload = {
        "filename": filename,
        "label": label,  # ✅ nueva columna
        "fps": float(fps) if fps is not None else None,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "duration_sec": float(duration_sec) if duration_sec is not None else None,
        "lighting": float(lighting) if lighting is not None else None,
        "resolution": resolution
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    res = supabase.table("videos").insert(payload).execute()

    data = getattr(res, "data", None) or []
    if not data or "id" not in data[0]:
        raise RuntimeError("No se recibió 'id' del insert en videos (checa RLS/permissions).")
    return data[0]["id"]


def maybe_downsample_frames(frames: list) -> list:
    if not frames:
        return frames
    if LANDMARKS_SAMPLE_EVERY > 1:
        frames = frames[::LANDMARKS_SAMPLE_EVERY]
    if LANDMARKS_MAX_FRAMES and LANDMARKS_MAX_FRAMES > 0:
        frames = frames[:LANDMARKS_MAX_FRAMES]
    return frames


def insert_landmarks_row(video_uuid: str, frames_json: list, fps_val):
    to_store = {"fps": fps_val, "frames": frames_json}
    res = supabase.table("landmarks").insert({
        "video_id": video_uuid,
        "landmarks": to_store
    }).execute()
    if hasattr(res, "error") and res.error:
        raise RuntimeError(res.error)
    return getattr(res, "data", None)


def main():
    base = Path(VIDEOS_DIR)
    total = ok_v = err_v = ok_lm = err_lm = 0

    for action, path in find_videos(base):
        total += 1
        try:
            fps, w, h, dur, light, reso = video_metadata(path)
            # 👇 ahora pasamos el label basado en la carpeta
            video_id = insert_video_row_return_id(
                filename=path.name,
                label=action,  # ✅ el nombre de la carpeta como label
                fps=fps, width=w, height=h,
                duration_sec=dur, lighting=light, resolution=reso
            )
            ok_v += 1
            print(f"🎬 Video OK: {action}/{path.name} | label={action} | id={video_id} | {reso} | {fps:.2f} fps | {dur:.2f}s | light≈{(light or 0):.1f}")

            if INSERT_LANDMARKS:
                try:
                    frames, fps_mp = extract_landmarks(str(path))
                    frames = maybe_downsample_frames(frames)
                    insert_landmarks_row(video_id, frames, fps_mp)
                    ok_lm += 1
                    print(f"🧍 Landmarks OK: {action}/{path.name} | frames_guardados={len(frames)}")
                except KeyboardInterrupt:
                    print("\n⏭️  Landmarks cancelado por el usuario; continúo…")
                    err_lm += 1
                except Exception as e_lm:
                    err_lm += 1
                    print(f"❌ Landmarks ERROR {action}/{path.name}: {e_lm}")
        except KeyboardInterrupt:
            print("\n⏭️  Cancelado por el usuario; continúo con el siguiente…")
            err_v += 1
            continue
        except Exception as e:
            err_v += 1
            print(f"❌ Video ERROR {action}/{path.name}: {e}")

    print("\nResumen ->")
    print(f"  videos:     total={total}, ok={ok_v}, error={err_v}")
    if INSERT_LANDMARKS:
        print(f"  landmarks:  ok={ok_lm}, error={err_lm}")


if __name__ == "__main__":
    main()
