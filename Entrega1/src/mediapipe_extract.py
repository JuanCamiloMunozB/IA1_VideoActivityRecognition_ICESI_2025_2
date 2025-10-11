import cv2
import mediapipe as mp

def extract_landmarks(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0

    # índices de los landmarks que nos interesan
    selected_landmarks = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_wrist": 15,
        "right_wrist": 16,
        "head": 0   # puedes usar 0 (nariz) o el promedio de orejas (7,8)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            filtered = {
                name: {
                    "x": lm[idx].x,
                    "y": lm[idx].y,
                    "z": lm[idx].z,
                    "visibility": lm[idx].visibility
                } for name, idx in selected_landmarks.items()
            }

            # opcional: calcular cabeza promediando orejas
            left_ear, right_ear = lm[7], lm[8]
            filtered["head"] = {
                "x": (left_ear.x + right_ear.x) / 2,
                "y": (left_ear.y + right_ear.y) / 2,
                "z": (left_ear.z + right_ear.z) / 2,
                "visibility": (left_ear.visibility + right_ear.visibility) / 2
            }

            frames.append({
                "frame_index": frame_idx,
                "timestamp": frame_idx / fps,
                "landmarks": filtered
            })

        frame_idx += 1

    cap.release()
    pose.close()
    return frames, fps
