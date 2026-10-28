"""Run from project root:  python -m tests.calibrate_quality
Prints blur/glare for every test image so you can pick sensible thresholds."""
import glob
from core.quality import ImageQualityGate

gate = ImageQualityGate()
for p in sorted(glob.glob("data/test_images/*.jp*g") + glob.glob("data/test_images/*.png")):
    r = gate.evaluate(p)
    print(f"{p:45s} blur={r['blur_score']:8.1f}  glare={r['glare_ratio']*100:5.2f}%  ok={r['is_acceptable']}")
