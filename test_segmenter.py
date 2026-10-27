import os
import cv2
from core.segment import LesionSegmenter

segmenter = LesionSegmenter()

test_img = "data/test_images/sample1.jpeg"
if not os.path.exists(test_img):
    test_img = "data/test_images/sample1.jpg"

print(f"[*] Testing LesionSegmenter on: {test_img}")
res = segmenter.segment(test_img, primary_concept="white patch")

print(f" -> Lesion Detected: {res['has_lesion_detected']}")
print(f" -> Matched Concept: {res['matched_concept']}")
print(f" -> Region Count: {res['region_count']}")
print(f" -> Relative Lesion Area: {res['area_ratio'] * 100:.2f}% of image")

if res['has_lesion_detected']:
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/sample1_segmenter_overlay.jpg"
    cv2.imwrite(out_path, res['overlay_image'])
    print(f" -> Saved reviewer overlay to: {out_path}")
    