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
    get_train_feature_columns, 
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
    Intenta cargar primero el modelo reducido (svm_reduced.joblib).
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

    # ------------------------------------------------------------
    # Intentar modelo reducido
    # ------------------------------------------------------------
    if prefer_reduced and SVM_REDUCED_PATH.exists():
        model = joblib.load(SVM_REDUCED_PATH)

        # Validamos que exista selected_features.json (documentación de K=70),
        # pero NO usamos su lista de 70 columnas para recortar X, porque el
        # SelectKBest ya está dentro del pipeline y espera 140 inputs.
        if SELECTED_FEATURES_PATH.exists():
            try:
                _ = _load_json(SELECTED_FEATURES_PATH)
            except Exception as e:
                print(
                    f"[load_model_artifacts] Advertencia: no se pudo leer "
                    f"{SELECTED_FEATURES_PATH}: {e}. Esto NO afecta la inferencia."
                )
        else:
            print(
                f"[load_model_artifacts] Advertencia: no se encontró {SELECTED_FEATURES_PATH}. "
                "La inferencia sigue funcionando, pero no se puede reportar el resumen de reducción."
            )

        # 👉 Aquí está la clave: usamos las 140 columnas de entrenamiento
        try:
            train_feature_cols: Sequence[str] = get_train_feature_columns()
        except FileNotFoundError as e:
            # Hacemos el error más explícito para el usuario
            raise FileNotFoundError(
                f"{e}\n"
                "Estas columnas son las que el pipeline reducido espera como entrada. "
                "Copia features.csv desde Entrega2/experiments/results antes de ejecutar la UI."
            )

        return ModelArtifacts(
            model=model,
            label_encoder=label_encoder,
            # PASAMOS LAS 140 FEATURES ORIGINALES, NO LAS 70 SELECCIONADAS
            selected_features=list(train_feature_cols),
            variant="reduced",
        )

    # ------------------------------------------------------------
    # Fallback: modelo full
    # ------------------------------------------------------------
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
