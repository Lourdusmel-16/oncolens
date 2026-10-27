import cv2
import numpy as np
from typing import Dict, Any

class ImageQualityGate:
    """
    Automated pre-flight image quality checker for intraoral photography.
    Evaluates blur via Laplacian variance and specular glare via HSV thresholding.
    Applies Gray-World White Balance and CLAHE for contrast enhancement.
    """
    def __init__(self, blur_threshold: float = 100.0, glare_threshold: float = 0.08):
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold

    def calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """Computes focus sharpness using the variance of the Laplacian operator."""
        return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    def calculate_glare_ratio(self, bgr_image: np.ndarray) -> float:
        """
        Detects specular glare (saliva reflection & direct camera flash).
        Specular glare in intraoral photography exhibits high Value (brightness) 
        and low Saturation in HSV color space.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        
        glare_mask = (v > 220) & (s < 45)
        glare_pixels = np.count_nonzero(glare_mask)
        total_pixels = bgr_image.shape[0] * bgr_image.shape[1]
        
        return float(glare_pixels / total_pixels)

    def enhance_contrast_clahe(self, bgr_image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization on the Luminance channel."""
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def gray_world_white_balance(self, bgr_image: np.ndarray) -> np.ndarray:
        """Corrects yellowish flashlight cast using Gray-World assumption."""
        b, g, r = cv2.split(bgr_image.astype(np.float32))
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        k = (b_avg + g_avg + r_avg) / 3.0
        
        b = np.clip(b * (k / (b_avg + 1e-5)), 0, 255)
        g = np.clip(g * (k / (g_avg + 1e-5)), 0, 255)
        r = np.clip(r * (k / (r_avg + 1e-5)), 0, 255)
        
        return cv2.merge([b, g, r]).astype(np.uint8)

    def evaluate(self, image_path: str) -> Dict[str, Any]:
        """Runs quality gate on an image file."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image from '{image_path}'")
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        blur_score = self.calculate_blur_score(gray)
        glare_ratio = self.calculate_glare_ratio(bgr)
        
        issues = []
        is_blurry = blur_score < self.blur_threshold
        has_excessive_glare = glare_ratio > self.glare_threshold
        
        if is_blurry:
            issues.append(f"Image is out of focus (sharpness score {blur_score:.1f} < {self.blur_threshold}). Please hold camera steady.")
        if has_excessive_glare:
            issues.append(f"Excessive flash/saliva glare ({glare_ratio*100:.1f}% area). Please tilt camera slightly to avoid direct reflection.")
            
        is_acceptable = (not is_blurry) and (not has_excessive_glare)
        
        balanced = self.gray_world_white_balance(bgr)
        enhanced = self.enhance_contrast_clahe(balanced)
        
        return {
            "is_acceptable": is_acceptable,
            "blur_score": round(blur_score, 2),
            "glare_ratio": round(glare_ratio, 4),
            "issues": issues,
            "enhanced_image": enhanced
        }