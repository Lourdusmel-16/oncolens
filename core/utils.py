"""Shared helpers: image preparation used by every stage."""
import cv2
import numpy as np
from typing import Tuple

MAX_SIDE = 1280  # keeps uploads small, fast and cheap on credits


def load_resized(image_path: str, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, bytes]:
    """Load an image (EXIF-rotated), shrink so the long side <= max_side.
    Returns (bgr_array, jpeg_bytes). Send the SAME bytes to SAM and to Muse Spark
    so box coordinates, masks and overlays all live in one coordinate system."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image from '{image_path}'")
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bgr, buf.tobytes()
