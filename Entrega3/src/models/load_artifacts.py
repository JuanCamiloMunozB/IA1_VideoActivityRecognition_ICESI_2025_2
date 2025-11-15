from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import joblib

from Entrega3.src.utils.config import (
    MODELS_DIR,
    RESULTS_DIR,
    SVM_REDUCED_PATH,
    SVM_FULL_PATH,
    LABEL_ENCODER_PATH,
    SELECTED_FEATURES_PATH,
)


@dataclass
class ModelArtifacts:
    """
    Contenedor de artefactos necesarios para inferencia:
    - modelo SVM (reducido o full)
    - encoder de labels
    - lista de features esperadas por el modelo
    """
    model: object
    label_encoder: object
    selected_features: Optional[List[str]]
    variant: str  # "reduced" o "full"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo JSON en: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_artifacts(prefer_reduced: bool = True) -> ModelArtifacts:
    """
    Intenta cargar primero el modelo reducido (svm_reduced.joblib + selected_features.json).
    Si no existe, cae al modelo full (svm_full.joblib) sin selección explícita de features.

    Retorna un ModelArtifacts con:
    - model: pipeline de sklearn listo para .predict / .predict_proba
    - label_encoder: encoder de etiquetas
    - selected_features: lista de nombres de columnas (o None si full)
    - variant: "reduced" o "full"
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar encoder de labels (común)
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró label_encoder.joblib en {LABEL_ENCODER_PATH}. "
            "Recuerda copiarlo desde Entrega2/experiments/models."
        )
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # Intentar modelo reducido
    if prefer_reduced and SVM_REDUCED_PATH.exists() and SELECTED_FEATURES_PATH.exists():
        model = joblib.load(SVM_REDUCED_PATH)
        data = _load_json(SELECTED_FEATURES_PATH)
        selected_features = data.get("selected_features", None)
        if not selected_features:
            raise ValueError(
                f"selected_features.json no contiene 'selected_features' en {SELECTED_FEATURES_PATH}"
            )
        return ModelArtifacts(
            model=model,
            label_encoder=label_encoder,
            selected_features=list(selected_features),
            variant="reduced",
        )

    # Fallback: modelo full
    if not SVM_FULL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró ni svm_reduced.joblib ni svm_full.joblib en {MODELS_DIR}.\n"
            "Asegúrate de haber corrido 01_svm_feature_reduction.ipynb y/o copiado svm_best.joblib "
            "desde Entrega2 como svm_full.joblib."
        )
    model = joblib.load(SVM_FULL_PATH)

    # Para el modelo full no tenemos lista explícita de features;
    # asumiremos que se tomarán todas las columnas numéricas de features_df.
    return ModelArtifacts(
        model=model,
        label_encoder=label_encoder,
        selected_features=None,
        variant="full",
    )
