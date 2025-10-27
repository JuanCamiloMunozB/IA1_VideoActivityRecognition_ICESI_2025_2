"""
Funciones de preprocesamiento y feature engineering para HAR con landmarks de MediaPipe.

Notas importantes
- Los landmarks disponibles incluyen: shoulders, hips, knees, ankles, wrists, head.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# ---------- Parámetros por defecto (sobreescribir desde prepare_dataset.py) ----------
DEFAULT_VISIBILITY_MIN = 0.80
DEFAULT_MIN_FRAMES_PER_VIDEO = 10

# Ventana temporal (en segundos)
DEFAULT_WINDOW_SIZE_SEC = 1.0
DEFAULT_WINDOW_STEP_SEC = 0.5

# Suavizado temporal (media móvil) en Nº de frames
DEFAULT_SMOOTH_WINDOW = 5


# ---------- Utilidades geométricas ----------
def _angle_3pts(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Ángulo ABC (radianes) entre vectores BA y BC."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc)) or 1.0
    cosang = float(np.dot(ba, bc) / den)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ---------- Normalización ----------
def normalize_landmarks_frame(
    landmarks: Dict[str, Dict[str, float]],
    origin: str = "pelvis_mid",
    scale_ref: Tuple[str, str] = ("left_shoulder", "right_shoulder"),
    scale_z: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Normaliza landmarks por frame:
    - Traslada al origen (p.ej., pelvis_mid = promedio de left_hip y right_hip).
    - Escala por distancia entre puntos de referencia (ej. ancho de hombros).

    Devuelve un dict con mismas keys normalizado en x,y (z se deja igual).
    """
    if not isinstance(landmarks, dict) or not landmarks:
        return landmarks

    L = landmarks.copy()

    # origen
    if origin == "pelvis_mid":
        req = ["left_hip", "right_hip"]
        if all(k in L for k in req):
            ox = (L["left_hip"]["x"] + L["right_hip"]["x"]) / 2.0
            oy = (L["left_hip"]["y"] + L["right_hip"]["y"]) / 2.0
        else:
            # fallback: usa left_hip
            ref = "left_hip" if "left_hip" in L else list(L.keys())[0]
            ox, oy = L[ref]["x"], L[ref]["y"]
    else:
        # origen como landmark específico
        if origin not in L:
            origin = list(L.keys())[0]
        ox, oy = L[origin]["x"], L[origin]["y"]

    # escala
    s1, s2 = scale_ref
    if s1 in L and s2 in L:
        dx = L[s1]["x"] - L[s2]["x"]
        dy = L[s1]["y"] - L[s2]["y"]
        scale = float(np.hypot(dx, dy)) or 1.0
    else:
        scale = 1.0

    out = {}
    for k, v in L.items():
        out[k] = {
            "x": (v.get("x", 0.0) - ox) / scale,
            "y": (v.get("y", 0.0) - oy) / scale,
            "z": (v.get("z", 0.0) / scale) if scale_z else v.get("z", 0.0),
            "visibility": v.get("visibility", 1.0),
        }
    return out


# ---------- Calidad / visibilidad ----------
def frame_visibility_ok(
    landmarks: Dict[str, Dict[str, float]],
    min_visibility: float = DEFAULT_VISIBILITY_MIN,
    required_keys: Tuple[str, ...] = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "head",
    ),
) -> bool:
    """Verifica que los landmarks esenciales superen el umbral de visibilidad."""
    if not isinstance(landmarks, dict):
        return False
    for k in required_keys:
        if k not in landmarks:
            return False
        if landmarks[k].get("visibility", 0.0) < min_visibility:
            return False
    return True


def filter_frames_by_quality(frames_df: pd.DataFrame, min_visibility: float = DEFAULT_VISIBILITY_MIN) -> pd.DataFrame:
    """
    Filtra frames por visibilidad mínima en landmarks esenciales.
    `frames_df` debe tener columnas: ['video_id','frame_index','landmarks', ...]
    """
    if frames_df.empty:
        return frames_df.copy()

    mask = frames_df["landmarks"].apply(lambda d: frame_visibility_ok(d, min_visibility=min_visibility))
    out = frames_df.loc[mask].copy()
    return out.reset_index(drop=True)


def filter_videos_min_frames(frames_df: pd.DataFrame, min_frames_per_video: int = DEFAULT_MIN_FRAMES_PER_VIDEO) -> pd.DataFrame:
    """Descarta videos con menos de `min_frames_per_video` frames válidos."""
    if frames_df.empty:
        return frames_df.copy()
    counts = frames_df.groupby("video_id")["frame_index"].count()
    keep_ids = counts[counts >= min_frames_per_video].index
    return frames_df[frames_df["video_id"].isin(keep_ids)].reset_index(drop=True)


# ---------- Features núcleo ----------
def compute_joint_angles(landmarks: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calcula ángulos clave (en radianes) por frame:
    - Rodillas: (hip - knee - ankle)
    - Caderas: (shoulder - hip - knee)
    """
    if not isinstance(landmarks, dict):
        return {}

    def pt(name: str):
        if name not in landmarks:
            return None
        return (landmarks[name]["x"], landmarks[name]["y"])

    out: Dict[str, float] = {}

    defs = {
        "knee_left": ("left_hip", "left_knee", "left_ankle"),
        "knee_right": ("right_hip", "right_knee", "right_ankle"),
        "hip_left": ("left_shoulder", "left_hip", "left_knee"),
        "hip_right": ("right_shoulder", "right_hip", "right_knee"),
    }
    for name, (a, b, c) in defs.items():
        pa, pb, pc = pt(a), pt(b), pt(c)
        if None in (pa, pb, pc):
            continue
        out[name] = _angle_3pts(pa, pb, pc)

    return out


def compute_trunk_inclination(landmarks: Dict[str, Dict[str, float]]) -> float:
    """
    Inclinación del tronco respecto a la vertical (radianes).
    Usa centros de hombros y caderas.
    """
    req = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if any(k not in landmarks for k in req):
        return np.nan

    sx = (landmarks["left_shoulder"]["x"] + landmarks["right_shoulder"]["x"]) / 2.0
    sy = (landmarks["left_shoulder"]["y"] + landmarks["right_shoulder"]["y"]) / 2.0
    hx = (landmarks["left_hip"]["x"] + landmarks["right_hip"]["x"]) / 2.0
    hy = (landmarks["left_hip"]["y"] + landmarks["right_hip"]["y"]) / 2.0

    v = np.array([sx - hx, sy - hy])
    vertical = np.array([0.0, -1.0])  # hacia arriba
    den = float(np.linalg.norm(v) * np.linalg.norm(vertical)) or 1.0
    cosang = float(np.dot(v, vertical) / den)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


def compute_torso_yaw_proxy(landmarks: Dict[str, Dict[str, float]]) -> float:
    """
    Proxy simple de yaw del torso usando los hombros en el plano X-Z.
    Definimos vector hombros: SR - SL, y calculamos atan2(dz, dx).
    Si no hay z, retorna NaN.
    """
    ls, rs = landmarks.get("left_shoulder"), landmarks.get("right_shoulder")
    if not ls or not rs:
        return np.nan
    dx = float(rs.get("x", np.nan) - ls.get("x", np.nan))
    dz = float(rs.get("z", np.nan) - ls.get("z", np.nan))
    if not np.isfinite(dx) or not np.isfinite(dz):
        return np.nan
    return float(np.arctan2(dz, dx))


def compute_velocities_sequence(
    frames: List[Dict[str, Any]],
    fps: float,
    keys: Tuple[str, ...] = (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_wrist",
        "right_wrist",
        "head",
    ),
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> Dict[str, np.ndarray]:
    """
    Velocidad (magnitud) por landmark a través de una secuencia de frames normalizados.
    Devuelve dict con arrays del mismo largo que `frames`.
    """
    if fps is None or fps <= 0:
        fps = 30.0  # fallback razonable

    default_dt = 1.0 / float(fps)
    n = len(frames)
    out: Dict[str, np.ndarray] = {}

    # Construye series x,y para cada key
    series_xy: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # timestamps (si existen) para derivar velocidades con espaciamiento real
    times: List[float] = []
    t_acc = 0.0
    for k in keys:
        xs, ys = [], []
        times.clear()
        t_acc = 0.0
        for i, fr in enumerate(frames):
            lmk = fr.get("landmarks_norm")
            if not lmk or k not in lmk:
                xs.append(np.nan)
                ys.append(np.nan)
            else:
                xs.append(lmk[k]["x"])
                ys.append(lmk[k]["y"])
            # manejar timestamp real si viene, sino acumular con dt fijo
            t = fr.get("timestamp")
            if isinstance(t, (int, float)) and t is not None:
                times.append(float(t))
            else:
                times.append(t_acc)
                t_acc += default_dt
        series_xy[k] = (np.array(xs, dtype=float), np.array(ys, dtype=float))
    times_arr = np.array(times, dtype=float) if n > 0 else np.array([], dtype=float)
    if n >= 2:
        # asegurar monotonicidad mínima en tiempos sintéticos
        if not np.all(np.diff(times_arr) > 0):
            # si no son estrictamente crecientes, usar malla regular por fps
            times_arr = np.arange(n, dtype=float) * default_dt

    # Derivada finita y magnitud
    for k, (xs, ys) in series_xy.items():
        # suavizado previo opcional
        if smooth_window and smooth_window > 1:
            xs = _moving_average(xs, smooth_window)
            ys = _moving_average(ys, smooth_window)

        # usar timestamps si están disponibles
        if n >= 2 and times_arr.size == n:
            vx = np.gradient(xs, times_arr)
            vy = np.gradient(ys, times_arr)
        else:
            vx = np.gradient(xs, default_dt)
            vy = np.gradient(ys, default_dt)
        vmag = np.sqrt(vx * vx + vy * vy)
        out[f"vel_{k}"] = vmag

    return out


# ---------- Ensamblado por ventana ----------
def window_indices(n_frames: int, fps: float, win_sec: float, step_sec: float) -> List[Tuple[int, int]]:
    """Genera índices (start, end) en frames para una ventana deslizante temporal."""
    if fps is None or fps <= 0:
        fps = 30.0
    win = max(1, int(round(win_sec * fps)))
    step = max(1, int(round(step_sec * fps)))
    idxs: List[Tuple[int, int]] = []
    start = 0
    while start + win <= n_frames:
        idxs.append((start, start + win))
        start += step
    if not idxs and n_frames > 0:  # si es muy corto, devolver 1 ventana
        idxs.append((0, n_frames))
    return idxs


def aggregate_window_features(
    frames: List[Dict[str, Any]],
    fps: float,
    win_sec: float = DEFAULT_WINDOW_SIZE_SEC,
    step_sec: float = DEFAULT_WINDOW_STEP_SEC,
) -> pd.DataFrame:
    """
    Crea features agregadas por ventana:
    - Promedios/DE de ángulos (rodillas, caderas)
    - Media de inclinación del tronco
    - Promedios/percentiles de velocidades por landmark
    """
    n = len(frames)
    if n == 0:
        return pd.DataFrame()

    idxs = window_indices(n, fps, win_sec, step_sec)

    # Timestamps del video: usar reales si existen; fallback a malla regular por fps
    default_dt = 1.0 / float(fps if (fps and fps > 0) else 30.0)
    times = []
    use_real = False
    for fr in frames:
        t = fr.get("timestamp")
        if isinstance(t, (int, float)) and t is not None:
            use_real = True
            times.append(float(t))
        else:
            times.append(np.nan)
    if not use_real:
        times = list(np.arange(n, dtype=float) * default_dt)
    else:
        # rellenar NaN si hubiese huecos y asegurar monotonicidad simple
        t_arr = np.array(times, dtype=float)
        # fill forward/backward
        mask = np.isnan(t_arr)
        if mask.any():
            # ffill
            idxs_valid = np.where(~mask)[0]
            if idxs_valid.size:
                for i in range(1, n):
                    if np.isnan(t_arr[i]):
                        t_arr[i] = t_arr[i-1]
                # bfill
                for i in range(n-2, -1, -1):
                    if np.isnan(t_arr[i]):
                        t_arr[i] = t_arr[i+1]
        # si no es estrictamente creciente, forzar malla regular
        if not np.all(np.diff(t_arr) > 0):
            t_arr = np.arange(n, dtype=float) * default_dt
        times = t_arr.tolist()

    # Pre-calcular por frame: ángulos e inclinación
    angles_per_frame: List[Dict[str, float]] = []
    incl_per_frame: List[float] = []

    for fr in frames:
        lmk = fr.get("landmarks_norm") or fr.get("landmarks")
        ang = compute_joint_angles(lmk)
        inc = compute_trunk_inclination(lmk)
        angles_per_frame.append(ang)
        incl_per_frame.append(inc)

    # Yaw proxy por frame (usar landmarks originales para z más estable)
    yaw_per_frame: List[float] = []
    shoulder_width: List[float] = []
    for fr in frames:
        L = fr.get("landmarks") or {}
        yaw_per_frame.append(compute_torso_yaw_proxy(L))
        if "left_shoulder" in L and "right_shoulder" in L:
            dx = float(L["right_shoulder"].get("x", np.nan) - L["left_shoulder"].get("x", np.nan))
            dy = float(L["right_shoulder"].get("y", np.nan) - L["left_shoulder"].get("y", np.nan))
            shoulder_width.append(float(np.hypot(dx, dy)))
        else:
            shoulder_width.append(np.nan)

    # Pre-calcular velocidades
    vel_dict = compute_velocities_sequence(frames, fps=fps)
    vel_keys = list(vel_dict.keys())

    # Convertir ángulos a DataFrame (alineado con frames)
    ang_df = pd.DataFrame(angles_per_frame).fillna(method="ffill").fillna(method="bfill")
    inc_arr = np.array(incl_per_frame, dtype=float)
    yaw_arr = np.array(yaw_per_frame, dtype=float)
    shoulder_arr = np.array(shoulder_width, dtype=float)

    rows = []
    for (a, b) in idxs:
        # ángulos
        ang_win = ang_df.iloc[a:b]
        feat = {}
        for col in ang_win.columns:
            feat[f"{col}_mean"] = float(ang_win[col].mean())
            feat[f"{col}_std"] = float(ang_win[col].std(ddof=0))

        # inclinación
        inc_win = inc_arr[a:b]
        feat["inclination_mean"] = float(np.nanmean(inc_win))
        feat["inclination_std"] = float(np.nanstd(inc_win))

        # velocidades por landmark
        for vk in vel_keys:
            v = vel_dict[vk][a:b]
            feat[f"{vk}_mean"] = float(np.nanmean(v))
            feat[f"{vk}_p90"] = float(np.nanpercentile(v, 90))

        # movimiento por región (suma de medias por región)
        region_map = {
            "hips": ["vel_left_hip", "vel_right_hip"],
            "knees": ["vel_left_knee", "vel_right_knee"],
            "ankles": ["vel_left_ankle", "vel_right_ankle"],
            "wrists": ["vel_left_wrist", "vel_right_wrist"],
            "head": ["vel_head"],
        }
        region_values = {}
        for rname, rk in region_map.items():
            vals = []
            for kname in rk:
                if kname in vel_dict:
                    vals.append(float(np.nanmean(vel_dict[kname][a:b])))
            region_values[rname] = float(np.nansum(vals)) if vals else np.nan
            feat[f"region_move_{rname}"] = region_values[rname]

        # visibilidad: medias y mínimos por región en la ventana
        req_keys = {
            "shoulders": ["left_shoulder", "right_shoulder"],
            "hips": ["left_hip", "right_hip"],
            "knees": ["left_knee", "right_knee"],
            "ankles": ["left_ankle", "right_ankle"],
            "wrists": ["left_wrist", "right_wrist"],
            "head": ["head"],
        }
        # recopilar visibilidades
        vis_by_region = {r: [] for r in req_keys}
        for i in range(a, b):
            lmk = frames[i].get("landmarks") or {}
            for r, names in req_keys.items():
                vals = [lmk[n].get("visibility", np.nan) for n in names if n in lmk]
                if vals:
                    vis_by_region[r].append(np.nanmean(vals))
        for r, arr in vis_by_region.items():
            if arr:
                arr_np = np.array(arr, dtype=float)
                feat[f"vis_{r}_mean"] = float(np.nanmean(arr_np))
                feat[f"vis_{r}_min"] = float(np.nanmin(arr_np))
            else:
                feat[f"vis_{r}_mean"] = np.nan
                feat[f"vis_{r}_min"] = np.nan

        # yaw del torso y su velocidad
        yaw_win = yaw_arr[a:b]
        feat["yaw_mean"] = float(np.nanmean(yaw_win))
        feat["yaw_std"] = float(np.nanstd(yaw_win))
        # derivada de yaw con timestamps
        t_win = np.array(times[a:b], dtype=float)
        if len(t_win) >= 2 and np.all(np.isfinite(yaw_win)):
            try:
                yaw_vel = np.gradient(yaw_win, t_win)
            except Exception:
                yaw_vel = np.gradient(yaw_win)
        else:
            yaw_vel = np.gradient(yaw_win)
        feat["yaw_vel_mean"] = float(np.nanmean(yaw_vel))
        feat["yaw_vel_p90"] = float(np.nanpercentile(yaw_vel, 90))

        # ancho de hombros (indicador de rotación respecto a la cámara)
        sw = shoulder_arr[a:b]
        feat["shoulder_width_mean"] = float(np.nanmean(sw))
        feat["shoulder_width_std"] = float(np.nanstd(sw))
        feat["shoulder_width_min"] = float(np.nanmin(sw))
        # relación mínimo/medio para detectar encogimiento relativo
        mean_sw = np.nanmean(sw)
        feat["shoulder_width_min_ratio"] = float(np.nanmin(sw) / mean_sw) if mean_sw and np.isfinite(mean_sw) else np.nan

        # tendencias por landmark (pendiente x/y vs tiempo dentro de la ventana)
        def _slope(t_arr: np.ndarray, v_arr: np.ndarray) -> float:
            mask = np.isfinite(t_arr) & np.isfinite(v_arr)
            if mask.sum() >= 2:
                try:
                    return float(np.polyfit(t_arr[mask], v_arr[mask], 1)[0])
                except Exception:
                    return float("nan")
            return float("nan")

        landmark_keys = (
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "left_wrist", "right_wrist",
            "head",
        )
        t_win = np.array(times[a:b], dtype=float)
        for lk in landmark_keys:
            xs, ys = [], []
            for i in range(a, b):
                lmk = frames[i].get("landmarks_norm") or frames[i].get("landmarks") or {}
                if lk in lmk:
                    xs.append(float(lmk[lk].get("x", np.nan)))
                    ys.append(float(lmk[lk].get("y", np.nan)))
                else:
                    xs.append(np.nan)
                    ys.append(np.nan)
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            feat[f"trend_{lk}_vx"] = _slope(t_win, xs)
            feat[f"trend_{lk}_vy"] = _slope(t_win, ys)

        # desplazamiento neto por landmark (último - primero de la ventana)
        l0 = frames[a].get("landmarks_norm") or frames[a].get("landmarks") or {}
        l1 = frames[b - 1].get("landmarks_norm") or frames[b - 1].get("landmarks") or {}
        for lk in landmark_keys:
            if lk in l0 and lk in l1:
                dx = float(l1[lk].get("x", np.nan) - l0[lk].get("x", np.nan))
                dy = float(l1[lk].get("y", np.nan) - l0[lk].get("y", np.nan))
            else:
                dx = np.nan
                dy = np.nan
            feat[f"disp_{lk}_dx"] = dx
            feat[f"disp_{lk}_dy"] = dy
            feat[f"disp_{lk}_mag"] = float(np.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else np.nan

        # información temporal de la ventana (basada en timestamps)
        t_start = float(times[a])
        t_end = float(times[b - 1])
        t_dur = float(max(t_end - t_start, 0.0))
        t0 = float(times[0])
        t_last = float(times[-1])
        denom = max(t_last - t0, 1e-6)
        t_mid = 0.5 * (t_start + t_end)
        feat["t_start"] = t_start
        feat["t_end"] = t_end
        feat["t_duration"] = t_dur
        feat["video_progress"] = float((t_mid - t0) / denom)

        feat["frame_start"] = int(frames[a]["frame_index"])
        feat["frame_end"] = int(frames[b - 1]["frame_index"])
        rows.append(feat)

    return pd.DataFrame(rows)


# ---------- Pipeline por video ----------
def prepare_video_windows(
    frames_df: pd.DataFrame,
    visibility_min: float = DEFAULT_VISIBILITY_MIN,
    min_frames_per_video: int = DEFAULT_MIN_FRAMES_PER_VIDEO,
    win_sec: float = DEFAULT_WINDOW_SIZE_SEC,
    step_sec: float = DEFAULT_WINDOW_STEP_SEC,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> pd.DataFrame:
    """
    Prepara features por video:
      1) filtra frames por visibilidad
      2) normaliza landmarks por frame
      3) agrega por ventana temporal
    Devuelve DF con columnas de features y columns ['video_id','label'].
    """
    if frames_df.empty:
        return pd.DataFrame()

    # 1) filtrar por calidad
    f = filter_frames_by_quality(frames_df, min_visibility=visibility_min)
    f = filter_videos_min_frames(f, min_frames_per_video=min_frames_per_video)
    if f.empty:
        return pd.DataFrame()

    out_rows = []
    for vid, sub in f.groupby("video_id"):
        label = sub["label"].iloc[0]
        fps = sub["fps"].iloc[0] if "fps" in sub.columns else None

        # construir lista de frames (normalizados)
        frames_list = []
        for _, r in sub.sort_values("frame_index").iterrows():
            lmk = r["landmarks"]
            lmk_norm = normalize_landmarks_frame(lmk)
            frames_list.append(
                {
                    "frame_index": int(r["frame_index"]),
                    "timestamp": float(r.get("timestamp", 0.0) or 0.0),
                    "landmarks": lmk,        # original
                    "landmarks_norm": lmk_norm,  # normalizado
                }
            )

        # suavizado implícito lo aplicamos en velocidades (y puedes extender a landmarks si lo necesitas)

        # agregación por ventana
        feat_df = aggregate_window_features(frames_list, fps=fps, win_sec=win_sec, step_sec=step_sec)
        if feat_df.empty:
            continue
        feat_df["video_id"] = vid
        feat_df["label"] = label
        out_rows.append(feat_df)

    if not out_rows:
        return pd.DataFrame()

    return pd.concat(out_rows, axis=0, ignore_index=True)
