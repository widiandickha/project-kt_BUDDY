from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import numpy as np

from src.config import ROOT_DIR

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    HAS_ULTRALYTICS = False


CLASS_ALIASES = {
    "mata_tertutup": "mata_tertutup",
    "closed_eye": "mata_tertutup",
    "closed": "mata_tertutup",
    "drowsy": "mata_tertutup",
    "menguap": "menguap",
    "yawn": "menguap",
    "yawning": "menguap",
    "kerutan_dahi": "kerutan_dahi",
    "brow_furrow": "kerutan_dahi",
}


@dataclass
class FaceExpressionResult:
    model_loaded: bool
    mata_tertutup: bool = False
    menguap: bool = False
    kerutan_dahi: bool = False
    confidences: Dict[str, float] = field(default_factory=dict)
    boxes: List[Dict[str, object]] = field(default_factory=list)


class CustomFaceYOLODetector:
    """pembungkus YOLOv8n kustom untuk deteksi mata_tertutup, menguap, dan kerutan_dahi.

    Mendukung bobot .pt maupun hasil ekspor .onnx. Jika bobot hasil training belum tersedia,
    detektor memberi tahu pipeline agar mengandalkan indikator landmark MediaPipe.
    """

    def __init__(
        self,
        model_rel_path: str = "models/face_expression_yolov8n.pt",
        conf_threshold: float = 0.45,
        use_two_class_fallback: bool = False,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.use_two_class_fallback = use_two_class_fallback
        self.model_path = self._resolve_path(model_rel_path)
        self._model = None
        self._load_model()

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        onnx_candidate = candidate.with_suffix(".onnx")
        if onnx_candidate.exists():
            return onnx_candidate
        return candidate

    def _load_model(self) -> None:
        if not HAS_ULTRALYTICS or not self.model_path.exists():
            self._model = None
            return
        try:
            self._model = YOLO(str(self.model_path))
        except Exception:
            self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray) -> FaceExpressionResult:
        if self._model is None:
            return FaceExpressionResult(model_loaded=False)

        result = FaceExpressionResult(model_loaded=True)
        preds = self._model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            verbose=False,
        )
        if not preds:
            return result

        pred = preds[0]
        names = pred.names or {}
        if pred.boxes is None:
            return result

        for box in pred.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_name = str(names.get(cls_id, cls_id)).lower().strip()
            canonical = CLASS_ALIASES.get(raw_name, raw_name)

            if canonical == "kerutan_dahi" and self.use_two_class_fallback:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            result.boxes.append(
                {
                    "class": canonical,
                    "confidence": round(conf, 3),
                    "bbox": xyxy,
                }
            )
            prev_conf = result.confidences.get(canonical, 0.0)
            if conf > prev_conf:
                result.confidences[canonical] = conf

            if canonical == "mata_tertutup":
                result.mata_tertutup = True
            elif canonical == "menguap":
                result.menguap = True
            elif canonical == "kerutan_dahi":
                result.kerutan_dahi = True

        return result
