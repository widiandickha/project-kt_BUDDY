from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
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
    """Menghitung EAR (mata), MAR (menguap), dan jarak alis dari MediaPipe FaceMesh.

    Jika MediaPipe FaceMesh tidak tersedia pada sistem, kelas ini menyediakan
    estimasi cadangan berbasis OpenCV Haar Cascade agar pipeline tetap berjalan.
    """

    def __init__(self) -> None:
        self._face_mesh = None
        if HAS_MEDIAPIPE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        face_xml = Path(getattr(cv2, "data", None).haarcascades if hasattr(cv2, "data") else "") / "haarcascade_frontalface_default.xml"
        eye_xml = Path(getattr(cv2, "data", None).haarcascades if hasattr(cv2, "data") else "") / "haarcascade_eye.xml"
        self._face_cascade = cv2.CascadeClassifier(str(face_xml)) if face_xml.exists() else None
        self._eye_cascade = cv2.CascadeClassifier(str(eye_xml)) if eye_xml.exists() else None

    def analyze(self, frame_bgr: np.ndarray) -> LandmarkMetrics:
        h, w = frame_bgr.shape[:2]
        if self._face_mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return LandmarkMetrics(
                    face_detected=False,
                    ear=0.28,
                    mar=0.15,
                    brow_ratio=0.24,
                )

            landmarks = results.multi_face_landmarks[0].landmark
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

            # Normalize brow-to-eye and inner-brow distance by outer eye span
            # so camera distance changes do not trigger false brow furrow events.
            eye_span = _euclidean(pts[33], pts[263])
            if eye_span > 1e-6:
                inner_brow = _euclidean(pts[107], pts[336]) / eye_span
                left_brow_eye = _euclidean(pts[65], pts[159]) / eye_span
                right_brow_eye = _euclidean(pts[295], pts[386]) / eye_span
                brow_ratio = 0.5 * inner_brow + 0.25 * (left_brow_eye + right_brow_eye)
            else:
                brow_ratio = 0.24

            x_min = max(0, int(np.min(pts[:, 0])))
            y_min = max(0, int(np.min(pts[:, 1])))
            x_max = min(w - 1, int(np.max(pts[:, 0])))
            y_max = min(h - 1, int(np.max(pts[:, 1])))
            nose_pt = (float(pts[1][0] / max(w, 1)), float(pts[1][1] / max(h, 1)))

            return LandmarkMetrics(
                face_detected=True,
                ear=float(ear),
                mar=float(mar),
                brow_ratio=float(brow_ratio),
                bbox=(x_min, y_min, x_max, y_max),
                head_point=nose_pt,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._face_cascade is None or self._face_cascade.empty():
            # Fallback when OpenCV 5 wheel omits bundled haarcascade XML files:
            # estimate eye openness from upper-central facial contrast variance.
            roi = gray[int(h * 0.22) : int(h * 0.78), int(w * 0.25) : int(w * 0.75)]
            if roi.size == 0:
                return LandmarkMetrics(face_detected=False, ear=0.28, mar=0.15, brow_ratio=0.24)
            eye_band = roi[int(roi.shape[0] * 0.20) : int(roi.shape[0] * 0.48), :]
            mouth_band = roi[int(roi.shape[0] * 0.65) : int(roi.shape[0] * 0.92), :]
            ear_est = 0.28 if float(np.std(eye_band)) > 22.0 else 0.17
            mar_est = 0.58 if float(np.std(mouth_band)) > 38.0 else 0.16
            return LandmarkMetrics(
                face_detected=True,
                ear=ear_est,
                mar=mar_est,
                brow_ratio=0.24,
                bbox=(int(w * 0.25), int(h * 0.20), int(w * 0.75), int(h * 0.80)),
                head_point=(0.5, 0.5),
            )

        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
        if len(faces) == 0:
            return LandmarkMetrics(
                face_detected=False,
                ear=0.28,
                mar=0.15,
                brow_ratio=0.24,
            )

        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        roi_gray = gray[y : y + fh // 2, x : x + fw]
        eyes = (
            self._eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
            if (self._eye_cascade is not None and not self._eye_cascade.empty())
            else [1]
        )
        ear_est = 0.29 if len(eyes) >= 1 else 0.16
        head_pt = ((x + fw * 0.5) / max(w, 1), (y + fh * 0.5) / max(h, 1))
        return LandmarkMetrics(
            face_detected=True,
            ear=ear_est,
            mar=0.16,
            brow_ratio=0.24,
            bbox=(int(x), int(y), int(x + fw), int(y + fh)),
            head_point=head_pt,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
