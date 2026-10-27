import os
import json
from core.reason import ClinicalReasoner

reasoner = ClinicalReasoner()

image_path = "data/test_images/sample1.jpeg"
if not os.path.exists(image_path):
    image_path = "data/test_images/sample1.jpg"

with open("data/test_images/answers_1.json", "r") as f:
    questionnaire = json.load(f)

# Mock SAM 3.1 segment info for testing
segment_info = {
    "has_lesion_detected": True,
    "region_count": 1,
    "area_ratio": 0.148,
}

referral = reasoner.reason(image_path, questionnaire, segment_info)

print("\n" + "="*50)
print("         ONCOLENS CLINICAL REFERRAL MEMO          ")
print("="*50)
print(f"TRIAGE URGENCY      : {referral.urgency.upper()}")
print(f"RECOMMENDED ACTION  : {referral.suggested_next_step}")
print(f"CONFIDENCE / LIMITS : {referral.confidence_note}")

print("\nOBJECTIVE VISUAL FINDINGS:")
for finding in referral.visible_findings:
    print(f"  • {finding}")

print("\nIDENTIFIED RISK FACTORS:")
for risk in referral.risk_factors:
    print(f"  • {risk}")

print("\nPATIENT SUMMARY (ENGLISH):")
print(f"{referral.patient_summary_en}")
print("="*50 + "\n")