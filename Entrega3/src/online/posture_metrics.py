from __future__ import annotations

import math
from typing import Dict


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _angle_3pts(a, b, c) -> float:
    """
    Ángulo en radianes en el punto b, entre segmentos ba y bc.
    """
    v1 = _vec(b, a)
    v2 = _vec(b, c)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(v1[0], v1[1]) or 1.0
    n2 = math.hypot(v2[0], v2[1]) or 1.0
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def compute_posture_metrics(landmarks: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calcula algunas métricas simples para mostrar en la UI:
    - inclinación del tronco
    - ángulos de rodilla izquierda/derecha
    """
    out = {
        "trunk_inclination_deg": 0.0,
        "knee_angle_l_deg": 0.0,
        "knee_angle_r_deg": 0.0,
    }

    if not landmarks:
        return out

    def pt(name):
        if name not in landmarks:
            return None
        return (landmarks[name]["x"], landmarks[name]["y"])

    # Inclinación tronco: vector hombros->caderas vs vertical
    if all(k in landmarks for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
        sx = (landmarks["left_shoulder"]["x"] + landmarks["right_shoulder"]["x"]) / 2.0
        sy = (landmarks["left_shoulder"]["y"] + landmarks["right_shoulder"]["y"]) / 2.0
        hx = (landmarks["left_hip"]["x"] + landmarks["right_hip"]["x"]) / 2.0
        hy = (landmarks["left_hip"]["y"] + landmarks["right_hip"]["y"]) / 2.0

        vx, vy = sx - hx, sy - hy
        vertical = (0.0, -1.0)
        dot = vx * vertical[0] + vy * vertical[1]
        n1 = math.hypot(vx, vy) or 1.0
        n2 = 1.0
        cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
        out["trunk_inclination_deg"] = math.degrees(math.acos(cosang))

    # Rodilla izquierda
    if all(k in landmarks for k in ["left_hip", "left_knee", "left_ankle"]):
        a = pt("left_hip")
        b = pt("left_knee")
        c = pt("left_ankle")
        if a and b and c:
            out["knee_angle_l_deg"] = _angle_3pts(a, b, c)

    # Rodilla derecha
    if all(k in landmarks for k in ["right_hip", "right_knee", "right_ankle"]):
        a = pt("right_hip")
        b = pt("right_knee")
        c = pt("right_ankle")
        if a and b and c:
            out["knee_angle_r_deg"] = _angle_3pts(a, b, c)

    return out
