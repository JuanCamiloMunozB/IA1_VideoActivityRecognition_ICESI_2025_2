from __future__ import annotations

from collections import deque
from typing import Dict, Any, Optional, Tuple

import numpy as np
import warnings

from Entrega3.src.models.load_artifacts import load_model_artifacts
from Entrega3.src.utils.config import WINDOW_SIZE_SEC, LIVE_VIDEO_ID, MIN_PREDICTION_PROB
from Entrega3.src.utils.preprocessing import frames_to_feature_vector

# ================== Filtros de warnings heredados de Entrega2 ==================
# No queremos tocar Entrega2/src/features/feature_engineering.py, pero sus
# funciones usan np.nanmean / np.nanquantile sobre slices que a veces quedan
# vacíos en tiempo real → generan RuntimeWarning que no afectan el flujo.
warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="All-NaN slice encountered",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but SelectKBest was fitted with feature names",
    category=UserWarning,
)


class RealtimeHARPredictor:
    def __init__(self, fps: float, max_seconds_buffer: float = 3.0, prefer_reduced: bool = True):
        """
        Predictor en tiempo real que acumula frames y dispara predicciones
        cuando hay suficiente ventana temporal.
        """
        self.fps = fps if fps and fps > 0 else 30.0
        self.frames: deque[Dict[str, Any]] = deque(maxlen=int(self.fps * max_seconds_buffer))
        self.counter: int = 0

        self.artifacts = load_model_artifacts(prefer_reduced=prefer_reduced)
        print(f"[RealtimeHARPredictor] Modelo cargado: {self.artifacts.variant}")
        self._last_label: Optional[str] = None
        self._last_prob: float = 0.0

    def add_frame(self, landmarks: Dict[str, Dict[str, float]]) -> None:
        """
        Agrega un frame al buffer interno.
        """
        self.counter += 1
        idx = self.counter
        fps = self.fps

        self.frames.append(
            {
                "video_id": LIVE_VIDEO_ID,
                "frame_index": idx,
                "timestamp": idx / fps,
                "t_start": idx / fps,
                "t_end": (idx + 1) / fps,
                "landmarks": landmarks,
            }
        )

    def maybe_predict(self) -> Optional[Tuple[str, float]]:
        """
        Si hay suficientes frames para al menos una ventana, genera una predicción.
        Devuelve (label, prob) o None si aún no hay suficientes datos o la
        predicción es de muy baja confianza.
        """
        min_frames = int(self.fps * WINDOW_SIZE_SEC)
        n_frames = len(self.frames)

        if n_frames < min_frames:
            # Evitar spamear el log cuando aún no ha entrado ningún frame al buffer
            if n_frames > 0 and n_frames % 30 == 0:
                print(f"[RealtimeHARPredictor] Calentando ventana: {n_frames}/{min_frames} frames")
            return None

        out = frames_to_feature_vector(list(self.frames), self.fps)
        if out is None:
            # Esto significa que aggregate_window_features devolvió vacío
            print("[RealtimeHARPredictor] frames_to_feature_vector devolvió None (sin ventana válida)")
            return None

        X, _ = out
        model = self.artifacts.model

        # 1. Predicción cruda del modelo
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            pred_raw = model.classes_[idx]
            prob = float(proba[idx])
        else:
            pred_raw = model.predict(X)[0]
            prob = 0.0

        # 2. Intentar decodificar con el LabelEncoder
        encoder = self.artifacts.label_encoder
        label: str

        try:
            # Caso típico: y original se codificó con LabelEncoder → enteros
            if np.issubdtype(type(pred_raw), np.integer):
                label = encoder.inverse_transform([pred_raw])[0]
            else:
                # Caso alternativo: pipeline con strings directos
                if hasattr(encoder, "classes_") and pred_raw in encoder.classes_:
                    pos = int(np.where(encoder.classes_ == pred_raw)[0][0])
                    label = encoder.inverse_transform([pos])[0]
                else:
                    label = str(pred_raw)
        except Exception:
            # Fallback defensivo
            label = str(pred_raw)

        # 3. Aplicar umbral de confianza
        if prob < MIN_PREDICTION_PROB:
            # Guardamos prob pero indicamos que la predicción es poco confiable.
            print(
                f"[RealtimeHARPredictor] Predicción de baja confianza ignorada: "
                f"{label} (p={prob:.3f} < {MIN_PREDICTION_PROB:.2f})"
            )
            self._last_label = None
            self._last_prob = prob
            return None

        self._last_label = label
        self._last_prob = prob

        print(f"[RealtimeHARPredictor] Predicción en vivo: {label} (p={prob:.3f})")

        return label, prob
