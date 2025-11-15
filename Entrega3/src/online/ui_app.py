from __future__ import annotations

import cv2
import numpy as np

from mediapipe import solutions as mp_solutions

from Entrega3.src.utils.config import CAMERA_FPS_FALLBACK, VISIBILITY_MIN
from Entrega3.src.utils.preprocessing import build_landmarks_dict_from_mediapipe
from Entrega3.src.online.realtime_inference import RealtimeHARPredictor
from Entrega3.src.online.posture_metrics import compute_posture_metrics


def _draw_info_panel(frame, lines, *, x=10, y=20, line_height=22, padding=10):
    """
    Dibuja un panel con fondo oscuro y varias lineas de texto.
    """
    if not lines:
        return

    panel_height = padding * 2 + line_height * len(lines)
    max_len = max(len(l) for l in lines)
    panel_width = padding * 2 + int(max_len * 7)

    x0, y0 = x, y - line_height
    x1, y1 = x0 + panel_width, y0 + panel_height

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x1, y1),
        (0, 0, 0),
        -1,
    )
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(
            frame,
            line,
            (x + padding, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 255, 200),
            1,
            cv2.LINE_AA,
        )


def _activity_color(prob: float) -> tuple[int, int, int]:
    if prob < 0.4:
        return (0, 0, 255)
    if prob < 0.7:
        return (0, 255, 255)
    return (0, 255, 0)


def run_realtime_app(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[UI] No se pudo abrir la camara con indice", camera_index)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = CAMERA_FPS_FALLBACK

    print("[UI] FPS estimado camara:", fps)

    predictor = RealtimeHARPredictor(fps=fps)
    print("[UI] Predictor inicializado con modelo", predictor.artifacts.variant)

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
                print("[UI] No se pudo leer frame. Saliendo...")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            landmarks_dict = None
            mean_vis = 0.0

            if result.pose_landmarks is not None:
                landmarks_dict = build_landmarks_dict_from_mediapipe(
                    result.pose_landmarks
                )

            if landmarks_dict is not None:
                vis_vals = [
                    v["visibility"]
                    for v in landmarks_dict.values()
                    if not np.isnan(v["visibility"])
                ]
                if vis_vals:
                    mean_vis = float(np.mean(vis_vals))

                predictor.add_frame(landmarks_dict)

                try:
                    metrics = compute_posture_metrics(landmarks_dict)
                except Exception as e:
                    metrics = {}
                    print("[UI] Error posture metrics:", e)
            else:
                metrics = {}

            pred = predictor.maybe_predict()

            # Panel principal
            overlay_lines = [
                f"Modelo: SVM {predictor.artifacts.variant} | FPS: {fps:.1f}"
            ]

            overlay_lines.append(f"Visibilidad media: {mean_vis:.2f}")
            if mean_vis < VISIBILITY_MIN:
                overlay_lines.append("Advertencia: baja visibilidad")

            current_prob = None

            if pred is None:
                activity_text = "Actividad: --- (calentando ventana)"
                overlay_lines.insert(1, activity_text)
            else:
                label, prob = pred
                activity_text = f"Actividad: {label} ({prob*100:.1f}%)"
                current_prob = prob

            # Panel de metrica
            metric_lines = []
            if metrics:
                metric_lines.append("Metricas postura:")
                for k in ["trunk_inclination_deg", "knee_angle_l_deg", "knee_angle_r_deg"]:
                    if k in metrics:
                        try:
                            metric_lines.append(f"  {k}: {float(metrics[k]):.1f}")
                        except:
                            metric_lines.append(f"  {k}: {metrics[k]}")

            _draw_info_panel(frame, overlay_lines, x=10, y=30)

            if metric_lines:
                _draw_info_panel(frame, metric_lines, x=10, y=120)

            # Texto de actividad coloreado
            if pred is not None and current_prob is not None:
                color = _activity_color(current_prob)
                cv2.putText(
                    frame,
                    activity_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            h, w, _ = frame.shape
            cv2.putText(
                frame,
                "Pulsa 'q' para salir",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("HAR Tiempo Real - SVM", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[UI] Tecla q. Saliendo...")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_app()
