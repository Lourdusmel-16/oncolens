import os
import sys
import json
import cv2
from typing import Dict, Any

from core.quality import ImageQualityGate
from core.segment import LesionSegmenter
from core.reason import ClinicalReasoner

def run_pipeline(image_path: str, answers_path: str) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("          ONCOLENS END-TO-END CLINICAL PIPELINE         ")
    print("="*60)

    # 1. Load Questionnaire
    if not os.path.exists(answers_path):
        raise FileNotFoundError(f"Questionnaire file not found: {answers_path}")
    with open(answers_path, "r", encoding="utf-8") as f:
        questionnaire = json.load(f)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs("outputs", exist_ok=True)

    # 2. Step 1: Quality Gate
    print("\n[STAGE 1/3] Running Image Quality Gate (OpenCV)...")
    gate = ImageQualityGate()
    quality = gate.evaluate(image_path)
    print(f" -> Sharpness Score : {quality['blur_score']} (Floor: 100)")
    print(f" -> Specular Glare  : {quality['glare_ratio']*100:.2f}% (Ceiling: 8.00%)")
    print(f" -> Gate Status     : {'PASSED' if quality['is_acceptable'] else 'REJECTED'}")

    if not quality['is_acceptable']:
        print("\n[!] Image failed quality gate. Action required:")
        for issue in quality['issues']:
            print(f"   • {issue}")
        # In field deployment, quality gate halts or requests retake
        print("[!] Note: Continuing execution in 'Override Mode' for demonstration...")

    # 3. Step 2: SAM 3.1 Lesion Segmentation
    print("\n[STAGE 2/3] Calling SAM 3.1 Segmentation (Meta Model API)...")
    segmenter = LesionSegmenter()
    segment_info = segmenter.segment(image_path, primary_concept="white patch")
    
    if segment_info['has_lesion_detected']:
        print(f" -> Lesion Detected : YES ({segment_info['matched_concept']})")
        print(f" -> Regions Found   : {segment_info['region_count']}")
        print(f" -> Affected Area   : {segment_info['area_ratio']*100:.2f}% of image")
        overlay_path = f"outputs/{base_name}_overlay.jpg"
        cv2.imwrite(overlay_path, segment_info['overlay_image'])
        print(f" -> Saved Overlay   : {overlay_path}")
    else:
        print(" -> Lesion Detected : NO (No focal lesion boundary identified)")

    # 4. Step 3: Multimodal Clinical Reasoning (Muse Spark)
    print("\n[STAGE 3/3] Multimodal Referral Synthesis (Meta Model API)...")
    reasoner = ClinicalReasoner()
    referral = reasoner.reason(image_path, questionnaire, segment_info)

    # 5. Compile Unified Case Output
    case_result = {
        "case_id": base_name,
        "quality_gate": {
            "passed": quality['is_acceptable'],
            "blur_score": quality['blur_score'],
            "glare_ratio": quality['glare_ratio'],
            "issues": quality['issues'],
        },
        "segmentation": {
            "detected": segment_info['has_lesion_detected'],
            "concept": segment_info['matched_concept'],
            "area_ratio": segment_info['area_ratio'],
            "region_count": segment_info['region_count'],
        },
        "referral": referral.model_dump(),
    }

    result_json_path = f"outputs/{base_name}_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(case_result, f, indent=2)

    # 6. Display Clinical Referral Summary
    print("\n" + "="*60)
    print(f"CASE RESULT: {base_name.upper()} | URGENCY: {referral.urgency.upper()}")
    print("="*60)
    print(f"NEXT STEP      : {referral.suggested_next_step}")
    print("\nFINDINGS       :")
    for f_item in referral.visible_findings:
        print(f"  • {f_item}")
    print("\nRISK PROFILE   :")
    for r_item in referral.risk_factors:
        print(f"  • {r_item}")
    print(f"\nPATIENT MEMO (EN):\n{referral.patient_summary_en}")
    print(f"\nSaved full case JSON to: {result_json_path}")
    print("="*60 + "\n")

    return case_result

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/sample1.jpeg"
    ans_arg = sys.argv[2] if len(sys.argv) > 2 else "data/test_images/answers_1.json"

    if not os.path.exists(img_arg) and os.path.exists("data/test_images/sample1.jpg"):
        img_arg = "data/test_images/sample1.jpg"

    run_pipeline(img_arg, ans_arg)