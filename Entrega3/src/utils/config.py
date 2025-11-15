from __future__ import annotations

import os
from pathlib import Path


def _find_project_root(start: Path, max_levels: int = 6) -> Path:
    """
    Sube directorios hasta encontrar uno que tenga 'Entrega2' y 'Entrega3'.
    Si no lo encuentra, devuelve `start`.
    """
    current = start
    for _ in range(max_levels):
        e2 = current / "Entrega2"
        e3 = current / "Entrega3"
        if e2.exists() and e3.exists():
            return current
        if current.parent == current:
            # Llegamos a la raíz
            break
        current = current.parent
    return start


# Detectar raíz del proyecto (carpeta que contiene Entrega2 y Entrega3)
if "PROJECT_ROOT" in os.environ:
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    PROJECT_ROOT = _find_project_root(Path(os.getcwd()).resolve())

ENTREGA2_DIR = PROJECT_ROOT / "Entrega2"
ENTREGA3_DIR = PROJECT_ROOT / "Entrega3"

# Directorios de artefactos para la entrega 3
EXPERIMENTS_DIR = ENTREGA3_DIR / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Rutas por defecto de artefactos
SVM_FULL_PATH = MODELS_DIR / "svm_full.joblib"
SVM_REDUCED_PATH = MODELS_DIR / "svm_reduced.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
SELECTED_FEATURES_PATH = RESULTS_DIR / "selected_features.json"

# Parámetros de ventana (coherentes con preprocesamiento de Entrega2)
WINDOW_SIZE_SEC = float(os.getenv("WINDOW_SIZE_SEC", 2.0))
WINDOW_STEP_SEC = float(os.getenv("WINDOW_STEP_SEC", 0.5))

# FPS para cálculo de ventanas (fallback si la cámara no reporta FPS)
CAMERA_FPS_FALLBACK = float(os.getenv("CAMERA_FPS_FALLBACK", 30.0))

# ⚠️ Bajamos el umbral de visibilidad para no descartar tantos frames
# Antes: 0.8 (demasiado alto en la práctica en cámara en vivo)
VISIBILITY_MIN = float(os.getenv("VISIBILITY_MIN", 0.5))

# Nombre "falso" de video para sesión en vivo
LIVE_VIDEO_ID = os.getenv("LIVE_VIDEO_ID", "live_session_001")

# Cache interna de columnas de entrenamiento
_TRAIN_FEATURE_COLUMNS = None


def get_train_feature_columns():
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
        raise FileNotFoundError(
            f"No se encontró features.csv en {csv_path}. "
            "Copia el archivo desde Entrega2/experiments/results."
        )

    import pandas as pd  # import local para no forzar dependencia global

    df = pd.read_csv(csv_path, nrows=5)  # sólo necesitamos el header
    all_cols = list(df.columns)

    meta_cols = {"video_id", "label", "frame_start", "frame_end"}
    feature_cols = [c for c in all_cols if c not in meta_cols]

    if len(feature_cols) != 140:
        print(
            f"[config.get_train_feature_columns] Advertencia: "
            f"se esperaban 140 features y se encontraron {len(feature_cols)}."
        )

    _TRAIN_FEATURE_COLUMNS = feature_cols
    return _TRAIN_FEATURE_COLUMNS
