from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import ROOT_DIR

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


@dataclass
class LandmarkMetrics:
    face_detected: bool
    ear: float
    mar: float
    brow_ratio: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    head_point: Optional[Tuple[float, float]] = None


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def compute_ear(pts: np.ndarray, indices: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in indices]
    horiz = _euclidean(p1, p4)
    if horiz < 1e-6:
        return 0.28
    vert = _euclidean(p2, p6) + _euclidean(p3, p5)
    return vert / (2.0 * horiz)


class FaceLandmarkAnalyzer:
    """Menghitung EAR, MAR, dan jarak alis menggunakan MediaPipe FaceLandmarker Tasks API (v1.1+).

    Didukung fallback Haar Cascade jika model task belum siap.
    """

    def __init__(self) -> None:
        self._landmarker = None
        task_path = ROOT_DIR / "models" / "face_landmarker.task"
        if HAS_MEDIAPIPE and task_path.exists():
            try:
                base_options = python.BaseOptions(model_asset_path=str(task_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    num_faces=1,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[FaceLandmarkAnalyzer] MediaPipe init error: {e}")
                self._landmarker = None

        cascade_dir = ROOT_DIR / "models" / "haarcascades"
        face_xml = cascade_dir / "haarcascade_frontalface_default.xml"
        eye_xml = cascade_dir / "haarcascade_eye.xml"
        self._face_cascade = cv2.CascadeClassifier(str(face_xml)) if face_xml.exists() else None
        self._eye_cascade = cv2.CascadeClassifier(str(eye_xml)) if eye_xml.exists() else None

    def analyze(self, frame_bgr: np.ndarray) -> LandmarkMetrics:
        h, w = frame_bgr.shape[:2]

        if self._landmarker is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect(mp_image)

            if res.face_landmarks and len(res.face_landmarks) > 0:
                landmarks = res.face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

                left_ear = compute_ear(pts, LEFT_EYE_IDX)
                right_ear = compute_ear(pts, RIGHT_EYE_IDX)
                ear = (left_ear + right_ear) / 2.0

                mouth_horiz = _euclidean(pts[78], pts[308])
                if mouth_horiz > 1e-6:
                    mouth_vert = (
                        _euclidean(pts[13], pts[14])
                        + _euclidean(pts[81], pts[178])
                        + _euclidean(pts[311], pts[402])
                    ) / 3.0
                    mar = mouth_vert / mouth_horiz
                else:
                    mar = 0.15

                eye_span = _euclidean(pts[33], pts[263])
                if eye_span > 1e-6:
                    inner_brow = _euclidean(pts[107], pts[336]) / eye_span
                    left_brow_eye = _euclidean(pts[65], pts[159]) / eye_span
                    right_brow_eye = _euclidean(pts[295], pts[386]) / eye_span
                    brow_ratio = 0.5 * inner_brow + 0.25 * (left_brow_eye + right_brow_eye)
                else:
                    brow_ratio = 0.24

                # Tentukan bounding box wajah nyata dari landmarks
                x_min = max(0, int(np.min(pts[:, 0])))
                y_min = max(0, int(np.min(pts[:, 1])))
                x_max = min(w - 1, int(np.max(pts[:, 0])))
                y_max = min(h - 1, int(np.max(pts[:, 1])))

                # Tambahkan margin sedikit agar pas di seluruh kepala
                margin_x = int((x_max - x_min) * 0.1)
                margin_y = int((y_max - y_min) * 0.15)
                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(w - 1, x_max + margin_x)
                y_max = min(h - 1, y_max + margin_y)

                nose_pt = (float(pts[1][0] / max(w, 1)), float(pts[1][1] / max(h, 1)))

                return LandmarkMetrics(
                    face_detected=True,
                    ear=float(ear),
                    mar=float(mar),
                    brow_ratio=float(brow_ratio),
                    bbox=(x_min, y_min, x_max, y_max),
                    head_point=nose_pt,
                )

        # Fallback Haar Cascade jika MediaPipe tidak mendeteksi wajah
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._face_cascade is not None and not self._face_cascade.empty():
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4)
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                roi_gray = gray[y : y + fh // 2, x : x + fw]
                eyes = (
                    self._eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
                    if (self._eye_cascade is not None and not self._eye_cascade.empty())
                    else [1]
                )
                ear_est = 0.28 if len(eyes) >= 1 else 0.18
                head_pt = ((x + fw * 0.5) / max(w, 1), (y + fh * 0.5) / max(h, 1))
                return LandmarkMetrics(
                    face_detected=True,
                    ear=ear_est,
                    mar=0.16,
                    brow_ratio=0.24,
                    bbox=(int(x), int(y), int(x + fw), int(y + fh)),
                    head_point=head_pt,
                )

        return LandmarkMetrics(
            face_detected=False,
            ear=0.28,
            mar=0.15,
            brow_ratio=0.24,
            bbox=None,
            head_point=None,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
