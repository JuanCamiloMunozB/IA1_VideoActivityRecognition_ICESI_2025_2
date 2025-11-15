from __future__ import annotations

import cv2
import numpy as np

from mediapipe import solutions as mp_solutions

from Entrega3.src.utils.config import CAMERA_FPS_FALLBACK, VISIBILITY_MIN
from Entrega3.src.utils.preprocessing import build_landmarks_dict_from_mediapipe
from Entrega3.src.online.realtime_inference import RealtimeHARPredictor
from Entrega3.src.online.posture_metrics import compute_posture_metrics


def run_realtime_app(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[UI] No se pudo abrir la cámara con índice {camera_index}")
        return

    # Intentar leer FPS de la cámara
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = CAMERA_FPS_FALLBACK

    print(f"[UI] FPS estimado de cámara: {fps}")

    predictor = RealtimeHARPredictor(fps=fps)

    mp_pose = mp_solutions.pose

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[UI] No se pudo leer frame de la cámara. Saliendo...")
                break

            # MediaPipe trabaja en RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            landmarks_dict = None
            mean_vis = 0.0

            if result.pose_landmarks is not None:
                landmarks_dict = build_landmarks_dict_from_mediapipe(
                    result.pose_landmarks
                )

            if landmarks_dict is not None:
                # Calcular visibilidad media (ignorando NaNs)
                vis_vals = [
                    v["visibility"]
                    for v in landmarks_dict.values()
                    if not np.isnan(v["visibility"])
                ]
                if vis_vals:
                    mean_vis = float(np.mean(vis_vals))

                # Antes descartábamos frames con VISIBILITY_MIN muy alto (0.8).
                # Ahora, mientras haya landmarks, SIEMPRE agregamos el frame
                # al predictor, y usamos el umbral solo como info de calidad.
                predictor.add_frame(landmarks_dict)

                # Métricas de postura (si fallan, simplemente no mostramos nada extra)
                try:
                    metrics = compute_posture_metrics(landmarks_dict)
                except Exception as e:
                    metrics = {}
                    print(f"[UI] Error en compute_posture_metrics: {e}")
            else:
                metrics = {}

            # Intentar predecir
            pred = predictor.maybe_predict()

            # ---------- Overlay en la imagen ----------
            overlay_text_lines = []

            if pred is None:
                overlay_text_lines.append(
                    "Actividad: --- (sin actividad clara / calentando ventana)"
                )
            else:
                label, prob = pred
                overlay_text_lines.append(
                    f"Actividad: {label} ({prob * 100:.1f}%)"
                )

            # Info de visibilidad
            overlay_text_lines.append(f"Visibilidad media: {mean_vis:.2f}")
            if mean_vis < VISIBILITY_MIN:
                overlay_text_lines.append("⚠ Pose con visibilidad baja")

            # Algunas métricas de postura, si existen
            for k, v in metrics.items():
                try:
                    overlay_text_lines.append(f"{k}: {float(v):.1f}")
                except Exception:
                    overlay_text_lines.append(f"{k}: {v}")

            # Dibujar texto en la parte superior izquierda
            y0 = 20
            dy = 20
            for i, line in enumerate(overlay_text_lines):
                y = y0 + i * dy
                cv2.putText(
                    frame,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("HAR Realtime - Entrega 3", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[UI] Tecla 'q' presionada. Saliendo...")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_app()
