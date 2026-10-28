import cv2
import numpy as np
from typing import Dict, Any

from core.utils import load_resized


class ImageQualityGate:
    """Pre-flight quality checks for intraoral photos (blur, glare) + optional enhancement.

    NOTE: the models always receive the ORIGINAL (resized) photo. Gray-World white balance
    assumes an average-gray scene; oral mucosa is mostly red/pink, so it can wash out the
    redness that matters clinically. It is therefore OFF by default (white_balance=False).
    """

    def __init__(self, blur_threshold: float = 100.0, glare_threshold: float = 0.08):
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold

    def calculate_blur_score(self, gray_image: np.ndarray) -> float:
        return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    def calculate_glare_ratio(self, bgr_image: np.ndarray) -> float:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        glare_mask = (v > 220) & (s < 45)
        return float(np.count_nonzero(glare_mask) / (bgr_image.shape[0] * bgr_image.shape[1]))

    def enhance_contrast_clahe(self, bgr_image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def gray_world_white_balance(self, bgr_image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(bgr_image.astype(np.float32))
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        k = (b_avg + g_avg + r_avg) / 3.0
        b = np.clip(b * (k / (b_avg + 1e-5)), 0, 255)
        g = np.clip(g * (k / (g_avg + 1e-5)), 0, 255)
        r = np.clip(r * (k / (r_avg + 1e-5)), 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    def evaluate(self, image_path: str, white_balance: bool = False) -> Dict[str, Any]:
        # Fixed working size so blur scores are comparable across phone cameras.
        bgr, _ = load_resized(image_path, max_side=1024)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        blur_score = self.calculate_blur_score(gray)
        glare_ratio = self.calculate_glare_ratio(bgr)

        issues = []
        is_blurry = blur_score < self.blur_threshold
        too_much_glare = glare_ratio > self.glare_threshold
        if is_blurry:
            issues.append(f"Image is out of focus (sharpness {blur_score:.1f} < {self.blur_threshold}). "
                          "Hold the camera steady and retake.")
        if too_much_glare:
            issues.append(f"Excessive flash/saliva glare ({glare_ratio*100:.1f}% of image). "
                          "Tilt the camera slightly and retake.")

        work = self.gray_world_white_balance(bgr) if white_balance else bgr
        return {
            "is_acceptable": (not is_blurry) and (not too_much_glare),
            "blur_score": round(blur_score, 2),
            "glare_ratio": round(glare_ratio, 4),
            "issues": issues,
            "enhanced_image": self.enhance_contrast_clahe(work),
        }
