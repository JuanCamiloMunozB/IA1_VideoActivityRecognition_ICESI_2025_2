"""
Cargadores y utilidades de datos para Entrega 2.
- Opción A: cargar desde Supabase (tablas `videos` y `landmarks`).
- Opción B: cargar desde CSV exportado previamente.

Todas las funciones devuelven DataFrames listos para preprocesar.
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

# Supabase client (v2)
try:
    from supabase import create_client, Client  # type: ignore
except Exception as e:
    create_client = None
    Client = None


def ensure_env_loaded() -> None:
    """Carga .env si existe (raíz del repo o Entrega2)."""
    load_dotenv()  # carga ./.env si existe


def get_supabase_client() -> Any:
    """Crea cliente de Supabase usando SUPABASE_URL y SUPABASE_ANON_KEY del entorno."""
    if create_client is None:
        raise RuntimeError("Supabase client no disponible. Instala dependencias (supabase==2.*).")
    ensure_env_loaded()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Faltan SUPABASE_URL/SUPABASE_ANON_KEY en el entorno (.env).")
    return create_client(url, key)


def load_from_supabase(
    videos_cols: Optional[List[str]] = None,
    landmarks_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Carga tablas `videos` y `landmarks` desde Supabase y hace merge.
    landmarks.landmarks se deja como dict (json) en columna 'landmarks_json'.
    """
    sb = get_supabase_client()

    v_cols = "*" if videos_cols is None else ",".join(videos_cols)
    l_cols = "*" if landmarks_cols is None else ",".join(landmarks_cols)

    # fetch
    v_res = sb.table("videos").select(v_cols).execute()
    l_res = sb.table("landmarks").select(l_cols).execute()
    df_v = pd.DataFrame(v_res.data or [])
    df_l = pd.DataFrame(l_res.data or [])

    if df_v.empty or df_l.empty:
        print("⚠️ Aviso: alguna tabla viene vacía. videos:", len(df_v), "landmarks:", len(df_l))

    # parse json
    if "landmarks" in df_l.columns:
        df_l["landmarks_json"] = df_l["landmarks"]
    elif "landmarks_json" not in df_l.columns:
        df_l["landmarks_json"] = None

    # merge: landmarks.video_id -> videos.id
    df = df_l.merge(df_v, left_on="video_id", right_on="id", how="inner", suffixes=("_lm", "_v"))
    return df


def load_from_csv(videos_csv: str, landmarks_csv: str) -> pd.DataFrame:
    """
    Carga `videos` y `landmarks` desde CSV y hace merge.
    Espera que `landmarks.landmarks` sea JSON (string) o ya dict en 'landmarks_json'.
    """
    df_v = pd.read_csv(videos_csv)
    df_l = pd.read_csv(landmarks_csv)

    if "landmarks_json" not in df_l.columns:
        if "landmarks" in df_l.columns:
            df_l["landmarks_json"] = df_l["landmarks"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
        else:
            raise ValueError("El CSV de landmarks no trae ni 'landmarks_json' ni 'landmarks'")

    df = df_l.merge(df_v, left_on="video_id", right_on="id", how="inner", suffixes=("_lm", "_v"))
    return df


def explode_frames_table(df: pd.DataFrame, frames_col: str = "landmarks_json") -> pd.DataFrame:
    """
    Explota el JSON por frame y retorna una tabla con una fila por (video, frame_index).
    Estructura esperada: landmarks_json = {"fps": <float>, "frames": [{ "frame_index": int, "timestamp": float, "landmarks": {...}}]}
    """
    rows = []
    for _, row in df.iterrows():
        data = row.get(frames_col) or {}
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = {}
        frames = data.get("frames", []) if isinstance(data, dict) else []
        fps = data.get("fps", None) if isinstance(data, dict) else None

        for fr in frames:
            rows.append(
                {
                    "video_id": row["video_id"],
                    "label": row.get("label"),
                    "fps": fps,
                    "frame_index": fr.get("frame_index"),
                    "timestamp": fr.get("timestamp"),
                    "landmarks": fr.get("landmarks"),
                    # Metadatos útiles del video:
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "duration_sec": row.get("duration_sec"),
                    "lighting": row.get("lighting"),
                    "resolution": row.get("resolution"),
                }
            )
    out = pd.DataFrame(rows)
    # Orden básico
    if not out.empty:
        out = out.sort_values(["video_id", "frame_index"], ascending=[True, True]).reset_index(drop=True)
    return out
