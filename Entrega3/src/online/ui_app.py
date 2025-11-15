from __future__ import annotations

import cv2
import numpy as np

from mediapipe import solutions as mp_solutions

from Entrega3.src.utils.config import CAMERA_FPS_FALLBACK, VISIBILITY_MIN
from Entrega3.src.utils.preprocessing import build_landmarks_dict_from_mediapipe
from Entrega3.src.online.realtime_inference import RealtimeHARPredictor
from Entrega3.src.online.posture_metrics import compute_posture_metrics


def _default_metrics() -> dict:
    return {
        "trunk_inclination_deg": 0.0,
        "knee_angle_l_deg": 0.0,
        "knee_angle_r_deg": 0.0,
    }


def run_realtime_app(camera_index: int = 0) -> None:
    mp_pose = mp_solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara en el índice {camera_index}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = CAMERA_FPS_FALLBACK
    print(f"[UI] FPS estimado de cámara: {fps:.1f}")

    predictor = RealtimeHARPredictor(fps=fps, max_seconds_buffer=3.0, prefer_reduced=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[UI] No se pudo leer frame de la cámara. Saliendo.")
                break

            # BGR → RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            landmarks_dict = None
            if result.pose_landmarks is not None:
                landmarks_dict = build_landmarks_dict_from_mediapipe(result.pose_landmarks)

            # Por defecto, métricas en cero
            metrics = _default_metrics()

            if landmarks_dict is not None:
                # Filtrar por visibilidad mínima promedio de puntos clave
                vis_vals = [v.get("visibility", 0.0) for v in landmarks_dict.values()]
                mean_vis = float(np.mean(vis_vals)) if vis_vals else 0.0

                # Siempre alimentamos el predictor para que la ventana se "llene".
                predictor.add_frame(landmarks_dict)

                # Las métricas solo se consideran confiables si la visibilidad promedio
                # supera el umbral (para no mostrar valores raros en la UI).
                if mean_vis >= VISIBILITY_MIN:
                    metrics = compute_posture_metrics(landmarks_dict)
                else:
                    # Métricas en cero si la pose es de baja calidad
                    metrics = _default_metrics()

            pred = predictor.maybe_predict()
            if pred is not None:
                label, prob = pred
                text = f"Actividad: {label}  (conf: {prob:0.2f})"
            else:
                text = "Actividad: --- (calentando ventana...)"

            # Dibujar texto y métricas en el frame
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), thickness=-1)
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Inclinacion tronco: {metrics['trunk_inclination_deg']:.1f}°",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Rodilla L: {metrics['knee_angle_l_deg']:.1f}° | R: {metrics['knee_angle_r_deg']:.1f}°",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("HAR – Entrega 3 (IA1)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_app()
