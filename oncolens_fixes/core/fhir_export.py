"""Build a FHIR R4 Bundle (collection) from a processed case. No diagnosis codes on purpose:
OncoLens documents and triages; it does not diagnose."""
import base64
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

GENDER = {"Male": "male", "Female": "female", "Other": "other"}


def build_bundle(case: Dict[str, Any], overlay_jpeg: Optional[bytes] = None) -> Dict[str, Any]:
    q, res, review = case["questionnaire"], case["result"], case.get("review")
    ref = res["referral"]
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    pid, rid = f"patient-{case['id']}", f"report-{case['id']}"

    def obs(oid: str, text: str, **value) -> Dict[str, Any]:
        return {"resourceType": "Observation", "id": f"{oid}-{case['id']}", "status": "preliminary",
                "code": {"text": text}, "subject": {"reference": f"Patient/{pid}"},
                "effectiveDateTime": now, **value}

    wk = q.get("lesion_duration_weeks")
    observations = [
        obs("age", "Patient age", valueQuantity={"value": q.get("age"), "unit": "years",
                                                  "system": "http://unitsofmeasure.org", "code": "a"}),
        obs("site", "Anatomical site of finding", valueString=q.get("anatomical_site", "Oral cavity")),
        obs("habits", "Reported habits", valueString=", ".join(q.get("habits") or []) or "None reported"),
        obs("duration", "Duration of change",
            **({"valueQuantity": {"value": wk, "unit": "weeks", "system": "http://unitsofmeasure.org", "code": "wk"}}
               if wk is not None else {"valueString": "Not sure"})),
        obs("redflags", "Warning signs reported",
            valueString=", ".join(k for k, v in (q.get("red_flags") or {}).items() if v) or "None"),
        obs("footprint", "Candidate lesion area (relative, % of image)",
            valueQuantity={"value": round(res["segmentation"]["area_ratio_all"] * 100, 2), "unit": "%",
                           "system": "http://unitsofmeasure.org", "code": "%"}),
        obs("triage", "Triage urgency (after safety rules)", valueString=res["urgency_final"]),
    ]

    conclusion = ref["suggested_next_step"]
    if review:
        conclusion += f" | Clinician decision: {review['decision']} ({review['final_urgency']}). {review.get('notes', '')}"

    report: Dict[str, Any] = {
        "resourceType": "DiagnosticReport", "id": rid,
        "status": "final" if review else "preliminary",
        "code": {"text": "Oral lesion documentation and referral triage (screening aid, not a diagnosis)"},
        "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": now, "issued": now,
        "result": [{"reference": f"Observation/{o['id']}"} for o in observations],
        "conclusion": conclusion[:1000],
    }
    if overlay_jpeg:
        report["presentedForm"] = [{"contentType": "image/jpeg", "title": "Photo with candidate regions",
                                    "data": base64.b64encode(overlay_jpeg).decode()}]

    patient = {"resourceType": "Patient", "id": pid, "gender": GENDER.get(q.get("sex"), "unknown")}
    entries = [patient, *observations, report]
    return {"resourceType": "Bundle", "id": str(uuid.uuid4()), "type": "collection", "timestamp": now,
            "entry": [{"fullUrl": f"urn:uuid:{e['id']}", "resource": e} for e in entries]}
