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

    def __init__(self, fps: float, frame_sample_every: int = 6) -> None:
        if fps is None or fps <= 0:
            fps = 30.0
        self.fps: float = float(fps)
        self.frame_sample_every: int = int(frame_sample_every)
        
        # FPS efectivo después del muestreo (si tomamos cada 6 frames, el FPS efectivo es fps/6)
        self.effective_fps: float = self.fps / self.frame_sample_every

        # Guardamos hasta 3 ventanas completas por si aggregate_window_features
        # usa stride / solapamiento. Usamos el FPS efectivo para calcular el tamaño del buffer.
        max_len = int(self.effective_fps * WINDOW_SIZE_SEC * 3)
        self.frames: Deque[Dict[str, Any]] = deque(maxlen=max_len)

        # Carga de modelo + encoder + selected_features
        self.artifacts: ModelArtifacts = load_model_artifacts(prefer_reduced=True)
        print(
            f"[RealtimeHARPredictor] Modelo cargado: {self.artifacts.variant} "
            f"(fps={self.fps}, sample_every={self.frame_sample_every}, "
            f"effective_fps={self.effective_fps:.2f}, buffer_max={max_len})"
        )

        # Contador para frames capturados de la cámara (antes del muestreo)
        self._raw_frame_counter: int = 0
        # Contador para frames muestreados (después del muestreo)
        self._sampled_frame_counter: int = 0
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
        Añade un frame al buffer interno solo si corresponde según el muestreo.
        Si frame_sample_every=6, solo se añaden los frames 0, 6, 12, 18, etc.
        
        Si no se dan timestamp/frame_index, se generan de forma incremental.
        """
        if landmarks is None:
            return

        # Incrementar contador de frames capturados
        self._raw_frame_counter += 1
        
        # Solo procesar si es un frame que debemos muestrear (cada frame_sample_every frames)
        if (self._raw_frame_counter - 1) % self.frame_sample_every != 0:
            return

        # Este frame sí se procesa
        if frame_index is None:
            frame_index = self._sampled_frame_counter
            self._sampled_frame_counter += 1

        if timestamp is None:
            # timestamp aprox en segundos usando el FPS efectivo
            if len(self.frames) == 0:
                timestamp = 0.0
            else:
                # El tiempo entre frames muestreados es frame_sample_every / fps original
                timestamp = self._last_timestamp + self.frame_sample_every / self.fps

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
        # Usamos el FPS efectivo (después del muestreo) para calcular min_frames
        min_frames = int(self.effective_fps * WINDOW_SIZE_SEC)
        n_frames = len(self.frames)

        if n_frames < max(min_frames, 10):
            # Mensaje de calentamiento cada 30 frames para no llenar la consola
            if n_frames % 30 == 0:
                print(
                    f"[RealtimeHARPredictor] Calentando ventana: "
                    f"{n_frames}/{min_frames} frames (effective_fps={self.effective_fps:.2f})"
                )
            return None

        # Usamos TODOS los frames del buffer (últimos N serán usados por la ventana)
        frames_list: List[Dict[str, Any]] = list(self.frames)

        # IMPORTANTE: Pasamos el FPS efectivo (después del muestreo) a frames_to_feature_vector
        # para que calcule correctamente las ventanas temporales
        out = frames_to_feature_vector(
            frames_list,
            fps=self.effective_fps,
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
