import os
import cv2
from core.quality import ImageQualityGate

gate = ImageQualityGate()

# Test on your sample image
test_img = "data/test_images/sample1.jpeg"
if not os.path.exists(test_img):
    test_img = "data/test_images/sample1.jpg"

print(f"[*] Running Quality Gate on: {test_img}")
result = gate.evaluate(test_img)

print(f" -> Is Acceptable: {result['is_acceptable']}")
print(f" -> Sharpness Score (Blur): {result['blur_score']} (Threshold: 100)")
print(f" -> Glare Ratio: {result['glare_ratio']*100:.2f}% (Threshold: 8.00%)")

if not result['is_acceptable']:
    print("\n[!] Rejection Reasons:")
    for issue in result['issues']:
        print(f"   - {issue}")
else:
    print("\n[PASS] Image cleared the quality gate!")
    # Save the enhanced version
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/sample1_enhanced.jpg", result['enhanced_image'])
    print(" -> Saved color/contrast-enhanced copy to 'outputs/sample1_enhanced.jpg'")