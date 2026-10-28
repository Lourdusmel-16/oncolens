"""Run from project root:  python -m tests.test_segmenter"""
import os
import cv2
from core.segment import LesionSegmenter

img = "data/test_images/sample1.jpeg"
if not os.path.exists(img):
    img = "data/test_images/sample1.jpg"

seg = LesionSegmenter().segment(img)          # 3 SAM calls (cached after the first run)
regions = seg["regions"]
print(f"Regions kept: {len(regions)}")
for r in regions:
    print(f"  #{r['id']} {r['concept']:12s} conf={r['confidence']} area_px={r['area_px']} box={r['box']}")
print(f"Union area: {LesionSegmenter.area_ratio(regions) * 100:.2f}% of image")

os.makedirs("outputs", exist_ok=True)
out = LesionSegmenter.render_overlay(seg["image_bgr"], regions)
cv2.imwrite("outputs/sample1_masks.jpg", out)
print("Saved outputs/sample1_masks.jpg  <- open it and check the masks sit on the right tissue")
