from __future__ import annotations

from collections import deque
from typing import Dict, Any, Optional, Tuple, Deque, List

import numpy as np

from Entrega3.src.models.load_artifacts import load_model_artifacts, ModelArtifacts
from Entrega3.src.utils.config import WINDOW_SIZE_SEC, LIVE_VIDEO_ID
from Entrega3.src.utils.preprocessing import frames_to_feature_vector


class RealtimeHARPredictor:
    """
    Administra un buffer de frames y ejecuta el pipeline SVM cuando hay
    suficiente información para una ventana deslizante.
    """

    def __init__(self, fps: float) -> None:
        if fps is None or fps <= 0:
            fps = 30.0
        self.fps: float = float(fps)

        # Guardamos hasta 3 ventanas completas por si aggregate_window_features
        # usa stride / solapamiento.
        max_len = int(self.fps * WINDOW_SIZE_SEC * 3)
        self.frames: Deque[Dict[str, Any]] = deque(maxlen=max_len)

        # Carga de modelo + encoder + selected_features
        self.artifacts: ModelArtifacts = load_model_artifacts(prefer_reduced=True)
        print(
            f"[RealtimeHARPredictor] Modelo cargado: {self.artifacts.variant} "
            f"(fps={self.fps}, buffer_max={max_len})"
        )

        # Contador para asignar frame_index y timestamp incremental
        self._frame_counter: int = 0
        self._last_timestamp: float = 0.0

    # ------------ Gestión de frames ------------

    def add_frame(
        self,
        landmarks: Dict[str, Dict[str, float]],
        *,
        timestamp: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> None:
        """
        Añade un frame al buffer interno. Si no se dan timestamp/frame_index,
        se generan de forma incremental.
        """
        if landmarks is None:
            return

        if frame_index is None:
            frame_index = self._frame_counter
            self._frame_counter += 1

        if timestamp is None:
            # timestamp aprox en segundos
            if len(self.frames) == 0:
                timestamp = 0.0
            else:
                timestamp = self._last_timestamp + 1.0 / self.fps

        self._last_timestamp = float(timestamp)

        frame = {
            "video_id": LIVE_VIDEO_ID,
            "frame_index": int(frame_index),
            "timestamp": float(timestamp),
            "landmarks": landmarks,
        }
        self.frames.append(frame)

    # ------------ Predicción ------------

    def maybe_predict(self) -> Optional[Tuple[str, float]]:
        """
        Intenta generar una predicción si hay suficientes frames.

        Devuelve:
            (label, prob_max) o None si NO se puede predecir todavía.
        """
        min_frames = int(self.fps * WINDOW_SIZE_SEC)
        n_frames = len(self.frames)

        if n_frames < max(min_frames, 10):
            # Mensaje de calentamiento cada 30 frames para no llenar la consola
            if n_frames % 30 == 0:
                print(
                    f"[RealtimeHARPredictor] Calentando ventana: "
                    f"{n_frames}/{min_frames} frames"
                )
            return None

        # Usamos TODOS los frames del buffer (últimos N serán usados por la ventana)
        frames_list: List[Dict[str, Any]] = list(self.frames)

        out = frames_to_feature_vector(
            frames_list,
            fps=self.fps,
            selected_features=self.artifacts.selected_features,
        )
        if out is None:
            # Ya hay bastantes frames, pero la ventana aún no generó features válidas
            print(
                "[RealtimeHARPredictor] frames_to_feature_vector devolvió None "
                "(probablemente ventana sin datos suficientes)."
            )
            return None

        X, feature_names = out

        if X.shape[1] != len(feature_names):
            print(
                f"[RealtimeHARPredictor] Inconsistencia: X tiene {X.shape[1]} "
                f"features pero feature_names tiene {len(feature_names)}."
            )
            return None

        model = self.artifacts.model

        # Obtener probabilidades o scores
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        else:
            # Fallback: usamos decision_function y le aplicamos softmax
            scores = model.decision_function(X)[0]
            scores = np.array(scores, dtype=np.float32)
            if scores.ndim == 0:
                scores = np.array([scores, -scores], dtype=np.float32)
            scores = scores - scores.max()
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum()

        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])

        # Mapear índice → label original
        label = self.artifacts.label_encoder.inverse_transform([best_idx])[0]
        return label, best_prob
