from __future__ import annotations

import importlib.util
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from Entrega3.src.utils.config import (
    ENTREGA2_DIR,
    WINDOW_SIZE_SEC,
    WINDOW_STEP_SEC,
    CAMERA_FPS_FALLBACK,
)


# ================== Carga dinámica de aggregate_window_features (Entrega2) ==================


def _load_feature_engineering_module():
    """
    Carga dinámicamente Entrega2/src/features/feature_engineering.py
    para reutilizar aggregate_window_features sin tocar la Entrega2.
    """
    fe_path = ENTREGA2_DIR / "src" / "features" / "feature_engineering.py"
    if not fe_path.exists():
        raise FileNotFoundError(f"No se encontró archivo: {fe_path}")

    spec = importlib.util.spec_from_file_location("feature_engineering_e2", fe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_feature_engineering_module = None


def get_aggregate_window_features():
    global _feature_engineering_module
    if _feature_engineering_module is None:
        _feature_engineering_module = _load_feature_engineering_module()

    if not hasattr(_feature_engineering_module, "aggregate_window_features"):
        raise AttributeError(
            "El módulo feature_engineering de Entrega2 no define aggregate_window_features."
        )

    return _feature_engineering_module.aggregate_window_features


# ================== Columnas reales de entrenamiento (SVM Entrega2) ==================

_TRAIN_FEATURE_COLUMNS: Optional[List[str]] = None


def get_training_feature_columns() -> List[str]:
    """
    Devuelve las columnas EXACTAS usadas para entrenar el modelo SVM reducido.

    Confirmado con tu features.csv:
    - Total columnas: 144
    - Columnas meta que NO van al modelo: video_id, label, frame_start, frame_end
    - → 144 - 4 = 140 features de entrada para el pipeline (antes de SelectKBest).
    """
    global _TRAIN_FEATURE_COLUMNS
    if _TRAIN_FEATURE_COLUMNS is not None:
        return _TRAIN_FEATURE_COLUMNS

    csv_path = ENTREGA2_DIR / "experiments" / "results" / "features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró features.csv en {csv_path}")

    df_head = pd.read_csv(csv_path, nrows=1)

    # MUY IMPORTANTE: estas 4 columnas no se usaron para entrenar
    meta_cols = {"video_id", "label", "frame_start", "frame_end"}

    cols = [c for c in df_head.columns if c not in meta_cols]

    if len(cols) != 140:
        raise ValueError(
            f"Se esperaban 140 columnas de características, pero se encontraron {len(cols)}.\n"
            f"Columnas detectadas: {cols}"
        )

    _TRAIN_FEATURE_COLUMNS = cols
    return cols


# ================== MediaPipe landmarks → diccionario ==================


def build_landmarks_dict_from_mediapipe(pose_landmarks):
    """
    Convierte los landmarks de MediaPipe Pose en el diccionario que usa todo el pipeline.
    """
    if pose_landmarks is None:
        return None

    # Compatibilidad con mediapipe 0.10.x (algunas distros usan mediapipe.python.solutions)
    try:
        from mediapipe.solutions.pose import PoseLandmark
    except Exception:
        from mediapipe.python.solutions.pose import PoseLandmark

    lm = pose_landmarks.landmark

    def p(idx):
        pt = lm[idx]
        return {
            "x": float(pt.x),
            "y": float(pt.y),
            "z": float(pt.z),
            "visibility": float(pt.visibility),
        }

    try:
        data = {
            "left_shoulder": p(PoseLandmark.LEFT_SHOULDER),
            "right_shoulder": p(PoseLandmark.RIGHT_SHOULDER),
            "left_hip": p(PoseLandmark.LEFT_HIP),
            "right_hip": p(PoseLandmark.RIGHT_HIP),
            "left_knee": p(PoseLandmark.LEFT_KNEE),
            "right_knee": p(PoseLandmark.RIGHT_KNEE),
            "left_ankle": p(PoseLandmark.LEFT_ANKLE),
            "right_ankle": p(PoseLandmark.RIGHT_ANKLE),
            "left_wrist": p(PoseLandmark.LEFT_WRIST),
            "right_wrist": p(PoseLandmark.RIGHT_WRIST),
            "left_ear": p(PoseLandmark.LEFT_EAR),
            "right_ear": p(PoseLandmark.RIGHT_EAR),
        }
    except Exception:
        return None

    # head = promedio de orejas
    le = data["left_ear"]
    re = data["right_ear"]
    data["head"] = {
        "x": (le["x"] + re["x"]) / 2.0,
        "y": (le["y"] + re["y"]) / 2.0,
        "z": (le["z"] + re["z"]) / 2.0,
        "visibility": (le["visibility"] + re["visibility"]) / 2.0,
    }

    return data


# ================== Normalización de frames para aggregate_window_features ==================


def _normalize_frames(frames: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    if fps is None or fps <= 0:
        fps = CAMERA_FPS_FALLBACK

    out: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        fr = dict(fr)

        fr.setdefault("video_id", "live_session")
        fr.setdefault("frame_index", i)
        fr.setdefault("timestamp", i / fps)
        fr.setdefault("t_start", i / fps)
        fr.setdefault("t_end", (i + 1) / fps)

        landmarks = fr.get("landmarks", {}) or {}
        fr["landmarks"] = landmarks

        fr["visible_landmarks"] = [
            k for k, v in landmarks.items()
            if isinstance(v, dict) and v.get("visibility", 0.0) >= 0.5
        ]

        out.append(fr)

    return out


# ================== Frames → vector de características para el modelo ==================


def frames_to_feature_vector(
    frames: List[Dict[str, Any]],
    fps: Optional[float],
    selected_features: Optional[List[str]] = None,  # se ignora: el SelectKBest interno se encarga
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Toma una lista de frames y genera un vector con EXACTAMENTE las 140 features
    que el pipeline reducido espera como entrada (antes de SelectKBest).

    Retorna:
        X: np.ndarray con shape (1, 140)
        feature_names: lista de columnas en el mismo orden
    """
    if not frames:
        return None

    if fps is None or fps <= 0:
        fps = CAMERA_FPS_FALLBACK

    frames_norm = _normalize_frames(frames, fps)

    agg_fn = get_aggregate_window_features()
    df = agg_fn(
        frames=frames_norm,
        fps=float(fps),
        win_sec=WINDOW_SIZE_SEC,
        step_sec=WINDOW_STEP_SEC,
    )

    if df is None or df.empty:
        return None

    last_row = df.iloc[-1]

    train_cols = get_training_feature_columns()

    # reindex para tener las 140 columnas exactas, rellenando faltantes con 0
    row_for_model = last_row.reindex(train_cols, fill_value=0.0)

    X = row_for_model.to_numpy(dtype=float).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, train_cols
