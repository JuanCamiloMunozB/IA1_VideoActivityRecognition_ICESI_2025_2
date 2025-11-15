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
        current = current.parent
    return start


# Detectar raíz del repo (donde están Entrega1, Entrega2, Entrega3)
try:
    # Caso script .py
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = _find_project_root(THIS_FILE)
except NameError:
    # Caso notebook (por si se importa desde Jupyter)
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
# En Entrega 2 se usó WINDOW_SIZE_SEC = 1.0 s (ver features_meta.json),
# así que aquí usamos 1.0 como valor por defecto para que despliegue y entrenamiento
# estén alineados. Se puede sobreescribir vía .env si se desea experimentar.
WINDOW_SIZE_SEC = float(os.getenv("WINDOW_SIZE_SEC", 1.0))
WINDOW_STEP_SEC = float(os.getenv("WINDOW_STEP_SEC", 0.5))

# FPS para cálculo de ventanas (fallback si la cámara no reporta FPS)
CAMERA_FPS_FALLBACK = float(os.getenv("CAMERA_FPS_FALLBACK", 30.0))

# Umbral mínimo de confianza de landmarks (se usa SOLO para métricas/UI; el modelo
# igualmente recibe los frames para evitar quedarse eternamente en "calentando ventana").
VISIBILITY_MIN = float(os.getenv("VISIBILITY_MIN", 0.8))

# Nombre "falso" de video para sesión en vivo
LIVE_VIDEO_ID = os.getenv("LIVE_VIDEO_ID", "live_session_001")
