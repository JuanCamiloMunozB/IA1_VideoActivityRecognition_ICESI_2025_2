"""
Script orquestador de preprocesamiento para Entrega 2.

Lee datos desde Supabase (o CSV), explota frames, filtra por calidad, normaliza,
ventaniza por tiempo y exporta un CSV de features listo para modelado.

Uso:
    # Con Supabase (lee .env: SUPABASE_URL, SUPABASE_ANON_KEY)
    python -m Entrega2.src.prepare_dataset --source supabase

    # Con CSVs
    python -m Entrega2.src.prepare_dataset --source csv \
        --videos_csv path/a/videos.csv \
        --landmarks_csv path/a/landmarks.csv

Parámetros ajustables por entorno (.env):
    VISIBILITY_MIN=0.8
    MIN_FRAMES_PER_VIDEO=10
    WINDOW_SIZE_SEC=1.0
    WINDOW_STEP_SEC=0.5
    SMOOTH_WINDOW=5
    OUTPUT_DIR=Entrega2/experiments/results
    OUTPUT_CSV=features.csv
"""

from __future__ import annotations
import os
import argparse
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# imports locales
from Entrega2.src.utils.data_utils import (
    load_from_supabase,
    load_from_csv,
    explode_frames_table,
)
from Entrega2.src.features.feature_engineering import (
    prepare_video_windows,
    DEFAULT_VISIBILITY_MIN,
    DEFAULT_MIN_FRAMES_PER_VIDEO,
    DEFAULT_WINDOW_SIZE_SEC,
    DEFAULT_WINDOW_STEP_SEC,
    DEFAULT_SMOOTH_WINDOW,
)


def get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["supabase", "csv"], required=True)
    parser.add_argument("--videos_csv", type=str, default=None)
    parser.add_argument("--landmarks_csv", type=str, default=None)
    args = parser.parse_args()

    visibility_min = get_env_float("VISIBILITY_MIN", DEFAULT_VISIBILITY_MIN)
    min_frames_per_video = get_env_int("MIN_FRAMES_PER_VIDEO", DEFAULT_MIN_FRAMES_PER_VIDEO)
    win_sec = get_env_float("WINDOW_SIZE_SEC", DEFAULT_WINDOW_SIZE_SEC)
    step_sec = get_env_float("WINDOW_STEP_SEC", DEFAULT_WINDOW_STEP_SEC)
    smooth_window = get_env_int("SMOOTH_WINDOW", DEFAULT_SMOOTH_WINDOW)

    out_dir = os.getenv("OUTPUT_DIR", "Entrega2/experiments/results")
    out_csv = os.getenv("OUTPUT_CSV", "features.csv")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_csv)

    # 1) cargar datos
    if args.source == "supabase":
        print("🔌 Cargando desde Supabase…")
        df = load_from_supabase()
    else:
        if not args.videos_csv or not args.landmarks_csv:
            raise SystemExit("Para --source csv debes pasar --videos_csv y --landmarks_csv")
        print("📄 Cargando desde CSVs…")
        df = load_from_csv(args.videos_csv, args.landmarks_csv)

    if df.empty:
        raise SystemExit("No se obtuvieron registros desde la fuente seleccionada.")

    # 2) explotar frames
    frames_df = explode_frames_table(df)
    if frames_df.empty:
        raise SystemExit("No hay frames en landmarks_json.")

    # 3) preparar features por video + ventana
    print("🧪 Preprocesando (filtros, normalización, ventanas)…")
    feat_df = prepare_video_windows(
        frames_df,
        visibility_min=visibility_min,
        min_frames_per_video=min_frames_per_video,
        win_sec=win_sec,
        step_sec=step_sec,
        smooth_window=smooth_window,
    )

    if feat_df.empty:
        raise SystemExit("No se generaron features (revisa umbrales y datos).")

    # 4) guardar
    feat_df.to_csv(out_path, index=False)
    print(f"✅ Features exportadas a: {out_path}")

    # 5) metadatos del procesamiento
    meta = {
        "visibility_min": visibility_min,
        "min_frames_per_video": min_frames_per_video,
        "window_size_sec": win_sec,
        "window_step_sec": step_sec,
        "smooth_window": smooth_window,
        "n_rows": int(len(feat_df)),
        "n_features": int(feat_df.shape[1] - 3),  # aprox (quitando video_id,label,frames)
        "label_counts": feat_df["label"].value_counts(dropna=False).to_dict(),
    }
    with open(os.path.join(out_dir, "features_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("📝 Metadatos guardados en features_meta.json")


if __name__ == "__main__":
    main()
