# file: load_video_info_to_supabase.py
# Requisitos:
#   pip install -r requirements.txt
#
# .env esperado:
#   SUPABASE_URL=https://xxxxx.supabase.co
#   SUPABASE_ANON_KEY=eyJhbGciOi...
#   VIDEOS_DIR=../../videos                # opcional
#   MAX_SAMPLES=60                     # opcional (número máx de frames a muestrear)
#   SAMPLE_STRIDE_SEC=0                # opcional; si >0 fuerza salto fijo en segundos

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
import cv2
import numpy as np
import math
import traceback

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
VIDEOS_DIR = os.getenv("VIDEOS_DIR", "../../videos")
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "60"))
SAMPLE_STRIDE_SEC = float(os.getenv("SAMPLE_STRIDE_SEC", "0"))

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("❌ Faltan SUPABASE_URL o SUPABASE_ANON_KEY en el .env")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MOV", ".MP4"}

def get_duration(cap, fps):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps and fps > 0:
        return frame_count / fps
    # fallback por tiempo (algunos contenedores reportan mejor el tiempo)
    dur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    return float(frame_count / fps) if fps and fps > 0 else 0.0

def sample_times(duration_sec: float, fps: float) -> list[float]:
    """Devuelve los tiempos (en segundos) a muestrear para estimar 'lighting'."""
    if duration_sec <= 0:
        return [0.0]
    if SAMPLE_STRIDE_SEC > 0:
        times = np.arange(0, duration_sec, SAMPLE_STRIDE_SEC, dtype=float).tolist()
    else:
        # Distribuye hasta MAX_SAMPLES puntos uniformemente en [0, duration]
        n = max(1, min(MAX_SAMPLES, int(math.ceil(duration_sec))))  # a lo sumo 1/s
        n = min(n, MAX_SAMPLES)
        if n == 1:
            times = [min(0.0, duration_sec)]
        else:
            times = np.linspace(0, max(0.0, duration_sec - 1e-3), n, dtype=float).tolist()
    # Garantiza que 0 esté incluido
    if times and times[0] > 0.05:
        times = [0.0] + times
    return times[:MAX_SAMPLES]

def video_metadata(video_path: Path):
    """Devuelve (fps, width, height, duration_sec, lighting, resolution)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Duración estimada
    duration_sec = get_duration(cap, fps)

    # Muestreo por tiempo (seek), muy barato comparado con iterar frames
    lighting_vals = []
    for t in sample_times(duration_sec, fps):
        # Posicionar por milisegundos
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
    """Itera videos bajo 'base/accion/archivo.ext' devolviendo (accion, path)"""
    if not base.exists():
        print(f"⚠️  Carpeta {base} no existe.")
        return
    for action_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        action = action_dir.name
        for file in action_dir.rglob("*"):
            if file.is_file() and file.suffix in ALLOWED_EXT:
                yield action, file

def insert_video_row(filename: str,
                     fps=None, width=None, height=None,
                     duration_sec=None, lighting=None, resolution=None):
    # Importante: NO incluimos 'storage_path' porque no existe en tu tabla.
    payload = {
        "filename": filename,
        "fps": float(fps) if fps is not None else None,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "duration_sec": float(duration_sec) if duration_sec is not None else None,
        "lighting": float(lighting) if lighting is not None else None,
        "resolution": resolution
    }
    # Elimina claves cuyo valor es None para evitar choques con NOT NULL/defauls
    payload = {k: v for k, v in payload.items() if v is not None}
    res = supabase.table("videos").insert(payload).execute()
    # supabase-py v2 retorna .data y .error en .model
    if hasattr(res, "error") and res.error:
        raise RuntimeError(res.error)
    return getattr(res, "data", None)

def main():
    base = Path(VIDEOS_DIR)
    total = 0
    ok = 0
    skipped = 0

    for action, path in find_videos(base):
        total += 1
        try:
            fps, w, h, dur, light, reso = video_metadata(path)

            _ = insert_video_row(
                filename=path.name,
                fps=fps,
                width=w,
                height=h,
                duration_sec=dur,
                lighting=light,
                resolution=reso
            )
            ok += 1
            print(f"✅ Insertado: {action}/{path.name} | {reso} | {fps:.2f} fps | {dur:.2f}s | light≈{(light or 0):.1f}")
        except KeyboardInterrupt:
            print("\n⏭️  Cancelado por el usuario en este archivo, continúo con el siguiente…")
            skipped += 1
            continue
        except Exception as e:
            skipped += 1
            # Intenta mostrar mensaje legible
            msg = str(e)
            if hasattr(e, "args") and e.args:
                msg = e.args[0]
            print(f"❌ Error con {action}/{path.name}: {msg}")
            # Debug opcional:
            # traceback.print_exc()

    print(f"\nResumen -> total: {total}, insertados: {ok}, con error: {skipped}")

if __name__ == "__main__":
    main()
