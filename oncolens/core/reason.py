import os
import json
import base64
import hashlib
from typing import List, Literal, Dict, Any, Optional

from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from core.utils import load_resized

load_dotenv()


class ReferralOutput(BaseModel):
    image_is_intraoral: bool = Field(default=True, description="False if not an intraoral photo.")
    urgency: Literal["routine", "priority", "urgent"]
    visible_findings: List[str]
    risk_factors: List[str]
    suggested_next_step: str
    image_quality_notes: str
    confidence_note: str
    patient_summary_en: str
    patient_summary_hi: str


SYSTEM_PROMPT = (
    "You are a clinical decision-support copilot for frontline community health workers "
    "and district dentists in India. "
    "You document oral mucosal findings and assist referral triage.\n"
    "CRITICAL RULES:\n"
    "1. You do NOT diagnose. Never write the words cancer, carcinoma, malignant, benign, tumour or precancer.\n"
    "2. Objectively describe visible features only: color, margins, ulceration, texture, anatomical location.\n"
    "3. Assess risk factors: khaini, gutka, supari/areca nut, bidi/smoking, alcohol, lesion duration.\n"
    "4. Any persistent mucosal change (>2 weeks) in a chronic tobacco/areca user MUST be 'priority' or 'urgent'.\n"
    "5. If evidence is ambiguous, escalate urgency (safe-by-design). Never answer 'routine' for a visible lesion.\n"
    "6. Always state that a licensed clinician must examine the patient in person.\n"
    "7. If the photo is not an intraoral photo, set image_is_intraoral to false.\n"
    "8. Reply STRICTLY with one JSON object, no markdown, matching:\n"
    "{\n"
    '  "image_is_intraoral": true | false,\n'
    '  "urgency": "routine" | "priority" | "urgent",\n'
    '  "visible_findings": ["..."],\n'
    '  "risk_factors": ["..."],\n'
    '  "suggested_next_step": "...",\n'
    '  "image_quality_notes": "...",\n'
    '  "confidence_note": "A human clinician must confirm all observations in person.",\n'
    '  "patient_summary_en": "Simple, respectful, non-alarming English.",\n'
    '  "patient_summary_hi": "Simple, respectful Hindi in Devanagari."\n'
    "}"
)


def _extract_json(text: str) -> str:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model reply")
    return text[start:end + 1]


class ClinicalReasoner:
    def __init__(self, model_name: Optional[str] = None, cache_dir: str = "outputs/cache"):
        self.api_key = os.getenv("MODEL_API_KEY")
        if not self.api_key:
            raise ValueError("MODEL_API_KEY is not set.")
        self.model_name = model_name or os.getenv("REASONER_MODEL", "muse-spark-1.1")
        if "contributor" in self.model_name:
            # Contributor-mode models use inputs/outputs for training: never send patient photos there.
            raise ValueError("Use a Standard-mode model (not '-contributor') for clinical images.")
        self.client = OpenAI(base_url="https://api.meta.ai/v1", api_key=self.api_key)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, jpeg: bytes, questionnaire: Dict[str, Any], seg: Optional[Dict[str, Any]]) -> str:
        blob = (jpeg + json.dumps(questionnaire, sort_keys=True).encode()
                + json.dumps(seg or {}, sort_keys=True).encode())
        return hashlib.sha256(blob + self.model_name.encode()).hexdigest()

    def reason(self, image_path: str, questionnaire: Dict[str, Any],
               segment_info: Optional[Dict[str, Any]] = None,
               use_cache: bool = True) -> ReferralOutput:
        _, jpeg = load_resized(image_path)
        seg_summary = None
        if segment_info:
            seg_summary = {"region_count": segment_info.get("region_count", 0),
                           "area_ratio": segment_info.get("area_ratio", 0.0)}

        cache_path = os.path.join(self.cache_dir, f"{self._cache_key(jpeg, questionnaire, seg_summary)}.json")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return ReferralOutput.model_validate_json(f.read())

        area_note = ""
        if seg_summary and seg_summary["region_count"] > 0:
            area_note = (f"\nSAM 3.1 proposed {seg_summary['region_count']} candidate region(s) covering about "
                         f"{seg_summary['area_ratio']*100:.1f}% of the image (a reviewer will confirm them).")
        elif seg_summary is not None:
            area_note = "\nSAM 3.1 found no candidate regions (this does NOT mean the mouth is normal)."

        user_text = (
            "Patient questionnaire:\n"
            f"- Age: {questionnaire.get('age', 'Unknown')}\n"
            f"- Sex: {questionnaire.get('sex', 'Unknown')}\n"
            f"- Site: {questionnaire.get('anatomical_site', 'Oral cavity')}\n"
            f"- Habits: {', '.join(questionnaire.get('habits') or ['None reported'])}\n"
            f"- Lesion duration: {questionnaire.get('lesion_duration_weeks', 'Unknown')} weeks\n"
            f"- Symptoms: {', '.join(questionnaire.get('symptoms') or ['None'])}\n"
            f"{area_note}\n\nAnalyze the image with the questionnaire and return the referral JSON."
        )
        b64 = base64.b64encode(jpeg).decode("utf-8")

        last_err = None
        for _ in range(2):  # one retry on malformed JSON
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]},
                ],
                temperature=0.1,
            )
            try:
                referral = ReferralOutput.model_validate_json(_extract_json(resp.choices[0].message.content))
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(referral.model_dump_json(indent=2))
                return referral
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise RuntimeError(f"Model returned invalid JSON twice: {last_err}")
