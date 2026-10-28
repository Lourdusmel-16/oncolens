"""End-to-end service + CLI.  python pipeline.py [image] [answers.json]"""
import os
import sys
import json
from typing import Any, Dict, List

import cv2
import numpy as np

from core.quality import ImageQualityGate
from core.segment import LesionSegmenter
from core.reason import ClinicalReasoner
from core.safety import apply_safety


def process_case(case_id: str, image_path: str, questionnaire: Dict[str, Any],
                 out_root: str = "outputs/cases", verbose: bool = True) -> Dict[str, Any]:
    folder = os.path.join(out_root, case_id)
    os.makedirs(folder, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    log("[1/3] Quality gate...")
    quality = ImageQualityGate().evaluate(image_path)

    log("[2/3] SAM 3.1 segmentation...")
    site = questionnaire.get("anatomical_site")
    seg = LesionSegmenter().segment(image_path, site=site)
    regions: List[Dict[str, Any]] = seg["regions"]
    meta = []
    for r in regions:  # masks go to disk as PNGs so the reviewer UI can toggle them later
        cv2.imwrite(os.path.join(folder, f"region_{r['id']}.png"), r["mask"] * 255)
        meta.append({k: r[k] for k in ("id", "concept", "box", "confidence", "area_px")})
    cv2.imwrite(os.path.join(folder, "resized.jpg"), seg["image_bgr"])
    area_all = LesionSegmenter.area_ratio(regions)
    cv2.imwrite(os.path.join(folder, "overlay.jpg"),
                LesionSegmenter.render_overlay(seg["image_bgr"], regions))

    log("[3/3] Clinical reasoning (Muse Spark)...")
    segment_info = {"has_lesion_detected": bool(regions), "region_count": len(regions), "area_ratio": area_all}
    referral = ClinicalReasoner().reason(image_path, questionnaire, segment_info)
    safe = apply_safety(referral.model_dump(), questionnaire, len(regions))

    result = {
        "case_id": case_id,
        "quality": {k: quality[k] for k in ("is_acceptable", "blur_score", "glare_ratio", "issues")},
        "image_is_intraoral": referral.image_is_intraoral,
        "segmentation": {"regions": meta, "area_ratio_all": area_all},
        "referral": safe["referral"],
        "urgency_llm": safe["urgency_llm"],
        "urgency_final": safe["urgency_final"],
        "safety_flags": safe["flags"],
    }
    with open(os.path.join(folder, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def load_regions(case_id: str, result: Dict[str, Any], out_root: str = "outputs/cases") -> List[Dict[str, Any]]:
    """Rebuild region dicts (with masks) from disk for the reviewer tab."""
    folder = os.path.join(out_root, case_id)
    regions = []
    for m in result["segmentation"]["regions"]:
        mask = cv2.imread(os.path.join(folder, f"region_{m['id']}.png"), cv2.IMREAD_GRAYSCALE)
        regions.append({**m, "mask": (mask > 127).astype(np.uint8)})
    return regions


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/sample1.jpeg"
    ans = sys.argv[2] if len(sys.argv) > 2 else "data/test_images/answers_1.json"
    with open(ans, "r", encoding="utf-8") as f:
        q = json.load(f)
    res = process_case("cli-" + os.path.splitext(os.path.basename(img))[0], img, q)
    print(json.dumps({k: res[k] for k in ("urgency_llm", "urgency_final", "safety_flags")}, indent=2))
