from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Importamos directamente las funciones y constantes de la Entrega 2
from Entrega2.src.features.feature_engineering import (
    aggregate_window_features,
    normalize_landmarks_frame,
    frame_visibility_ok,
    DEFAULT_WINDOW_SIZE_SEC,
    DEFAULT_WINDOW_STEP_SEC,
)

# ============================================================
# Constantes
# ============================================================

# Lista de nombres estándar de los 33 landmarks de MediaPipe Pose
POSE_LANDMARK_NAMES: List[str] = [
    "nose",                # 0
    "left_eye_inner",      # 1
    "left_eye",            # 2
    "left_eye_outer",      # 3
    "right_eye_inner",     # 4
    "right_eye",           # 5
    "right_eye_outer",     # 6
    "left_ear",            # 7
    "right_ear",           # 8
    "mouth_left",          # 9
    "mouth_right",         # 10
    "left_shoulder",       # 11
    "right_shoulder",      # 12
    "left_elbow",          # 13
    "right_elbow",         # 14
    "left_wrist",          # 15
    "right_wrist",         # 16
    "left_pinky",          # 17
    "right_pinky",         # 18
    "left_index",          # 19
    "right_index",         # 20
    "left_thumb",          # 21
    "right_thumb",         # 22
    "left_hip",            # 23
    "right_hip",           # 24
    "left_knee",           # 25
    "right_knee",          # 26
    "left_ankle",          # 27
    "right_ankle",         # 28
    "left_heel",           # 29
    "right_heel",          # 30
    "left_foot_index",     # 31
    "right_foot_index",    # 32
]

CAMERA_FPS_FALLBACK: float = 30.0

# ============================================================
# 1. MediaPipe → diccionario de landmarks
# ============================================================


def _landmark_proto_to_dict(lm: Any) -> Dict[str, float]:
    """Convierte un landmark de MediaPipe a un dict simple."""
    return {
        "x": float(getattr(lm, "x", 0.0)),
        "y": float(getattr(lm, "y", 0.0)),
        "z": float(getattr(lm, "z", 0.0)),
        "visibility": float(getattr(lm, "visibility", 1.0)),
    }


def build_landmarks_dict_from_mediapipe(results: Any) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Convierte la salida de MediaPipe Pose a un diccionario:

    {
        "nose": {"x":..., "y":..., "z":..., "visibility":...},
        "left_shoulder": {...},
        ...
        "head": {...}  # promedio de left_ear y right_ear
    }

    Soporta dos casos:
    - results es el objeto completo devuelto por MediaPipe (tiene .pose_landmarks)
    - results es directamente un NormalizedLandmarkList (tiene .landmark)
    """
    if results is None:
        return None

    # 1) Si nos pasan el objeto completo de MediaPipe
    if hasattr(results, "pose_landmarks"):
        mp_landmarks = results.pose_landmarks
    else:
        mp_landmarks = results

    if mp_landmarks is None:
        return None

    # 2) Obtenemos la lista de landmarks
    if hasattr(mp_landmarks, "landmark"):
        lm_list = mp_landmarks.landmark
    else:
        # Si no tiene .landmark, intentamos iterar directamente
        try:
            lm_list = list(mp_landmarks)
        except TypeError:
            # No es iterable, no podemos hacer nada
            return None

    data: Dict[str, Dict[str, float]] = {}

    for idx, lm in enumerate(lm_list):
        if idx < len(POSE_LANDMARK_NAMES):
            name = POSE_LANDMARK_NAMES[idx]
        else:
            name = f"landmark_{idx}"

        data[name] = _landmark_proto_to_dict(lm)

    # Landmark sintético "head" (promedio de las orejas)
    if "left_ear" in data and "right_ear" in data:
        le = data["left_ear"]
        re = data["right_ear"]
        data["head"] = {
            "x": (le["x"] + re["x"]) / 2.0,
            "y": (le["y"] + re["y"]) / 2.0,
            "z": (le["z"] + re["z"]) / 2.0,
            "visibility": (le["visibility"] + re["visibility"]) / 2.0,
        }

    return data


# ============================================================
# 2. Normalización de frames (misma lógica que Entrega 2)
# ============================================================


def _normalize_frames(
    frames: List[Dict[str, Any]],
    fps: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Añade 'landmarks_norm' a cada frame usando normalize_landmarks_frame de la Entrega 2
    y filtra los frames con visibilidad muy baja mediante frame_visibility_ok.

    Cada frame:
    {
        "video_id": str,
        "frame_index": int,
        "timestamp": float,
        "landmarks": { ... }
    }
    """
    if fps is None or fps <= 0:
        fps = CAMERA_FPS_FALLBACK

    normalized_frames: List[Dict[str, Any]] = []

    for fr in frames:
        landmarks = fr.get("landmarks")
        if not isinstance(landmarks, dict) or not landmarks:
            continue

        # Misma verificación de visibilidad que en Entrega 2
        if not frame_visibility_ok(landmarks):
            continue

        try:
            landmarks_norm = normalize_landmarks_frame(landmarks)
        except Exception:
            # Si algo falla, usamos los landmarks crudos para no romper
            landmarks_norm = landmarks

        new_fr = dict(fr)
        new_fr["landmarks_norm"] = landmarks_norm

        if "frame_index" not in new_fr:
            new_fr["frame_index"] = len(normalized_frames)

        if "timestamp" not in new_fr:
            new_fr["timestamp"] = new_fr["frame_index"] / fps

        normalized_frames.append(new_fr)

    return normalized_frames


# ============================================================
# 3. Frames → vector de características (para el SVM)
# ============================================================

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Asegúrate de tener estos imports arriba en el archivo:
# from Entrega2.src.features.feature_engineering import aggregate_window_features
# from Entrega3.src.config import (
#     CAMERA_FPS_FALLBACK,
#     DEFAULT_WINDOW_SIZE_SEC,
#     DEFAULT_WINDOW_STEP_SEC,
# )

def frames_to_feature_vector(
    frames: List[Dict[str, Any]],
    fps: Optional[float],
    selected_features: Optional[List[str]] = None,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Convierte una lista de frames con landmarks a un vector de características
    listo para pasar al pipeline clásico (con SelectKBest, StandardScaler, SVC).

    Devuelve:
      - None si no se pudo construir nada útil.
      - (X, feature_names) si todo salió bien:
          X: np.ndarray de shape (1, n_features)
          feature_names: lista de nombres de columnas en el mismo orden que X.
    """
    if fps is None or fps <= 0:
        fps = CAMERA_FPS_FALLBACK

    # Normalizar frames (usa tu función ya existente)
    norm_frames = _normalize_frames(frames, fps=fps)

    if not norm_frames:
        return None

    # Agregar características por ventana usando la lógica de Entrega 2
    feats_df: pd.DataFrame = aggregate_window_features(
        norm_frames,
        fps,
        DEFAULT_WINDOW_SIZE_SEC,
        DEFAULT_WINDOW_STEP_SEC,
    )

    if feats_df is None or feats_df.empty:
        return None

    # Tomamos la última ventana (la más reciente)
    last_window = feats_df.iloc[[-1]].copy()

    # MUY IMPORTANTE:
    # Si nos pasan selected_features, el pipeline espera EXACTAMENTE
    # len(selected_features) columnas, en ese orden.
    # Reindexamos para:
    #   - Crear las columnas faltantes con 0.0
    #   - Reordenar columnas al orden esperado por el modelo.
    if selected_features is not None:
        last_window = last_window.reindex(columns=selected_features, fill_value=0.0)

    # Evitar NaNs e infinitos
    last_window = last_window.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Guardamos los nombres de features en el orden correcto
    feature_names: List[str] = list(last_window.columns)

    # Convertimos a ndarray
    X = last_window.to_numpy(dtype=np.float32)  # shape (1, n_features)

    return X, feature_names
