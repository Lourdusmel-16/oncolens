import os
import base64
import json
import hashlib
from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class ReferralOutput(BaseModel):
    urgency: Literal["routine", "priority", "urgent"] = Field(
        description="Triage urgency based on habit risk, duration (>2 weeks), and visual features."
    )
    visible_findings: List[str] = Field(
        description="Objective visual characteristics (color, margins, texture, site). No formal histology diagnosis."
    )
    risk_factors: List[str] = Field(
        description="Identified carcinogen exposures (e.g. smokeless tobacco, bidi, areca nut duration)."
    )
    suggested_next_step: str = Field(
        description="Actionable clinical recommendation (e.g. 'Specialist dental examination and biopsy within 2 weeks')."
    )
    image_quality_notes: str = Field(
        description="Lighting, focus, or positioning limitations noted by the model."
    )
    confidence_note: str = Field(
        description="Explicit clinical limitation statement reminding that a human clinician must confirm."
    )
    patient_summary_en: str = Field(
        description="Clear, non-alarming explanation in simple English for the patient."
    )
    patient_summary_hi: str = Field(
        description="Clear, non-alarming explanation in conversational Hindi for the patient."
    )

class ClinicalReasoner:
    def __init__(self, model_name: str = "muse-spark-1.1", cache_dir: str = "outputs/cache"):
        self.api_key = os.getenv("MODEL_API_KEY")
        if not self.api_key:
            raise ValueError("MODEL_API_KEY is not set.")
        
        self.client = OpenAI(
            base_url="https://api.meta.ai/v1",
            api_key=self.api_key,
        )
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_cache_key(self, image_bytes: bytes, questionnaire: Dict[str, Any]) -> str:
        content = image_bytes + json.dumps(questionnaire, sort_keys=True).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def reason(
        self,
        image_path: str,
        questionnaire: Dict[str, Any],
        segment_info: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> ReferralOutput:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: '{image_path}'")

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

        cache_key = self._generate_cache_key(image_bytes, questionnaire)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if use_cache and os.path.exists(cache_path):
            print(f"[*] Loaded cached referral reasoning from: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return ReferralOutput.model_validate_json(f.read())

        print(f"[*] Calling Meta Model API ({self.model_name}) for clinical referral synthesis...")

        area_note = ""
        if segment_info and segment_info.get("has_lesion_detected"):
            area_note = (
                f"\nSAM 3.1 lesion segmentation identified {segment_info.get('region_count', 0)} region(s) "
                f"occupying approximately {segment_info.get('area_ratio', 0)*100:.1f}% of the visible image."
            )

        system_prompt = (
            "You are a clinical decision-support copilot for frontline community health workers and district dentists in India. "
            "You document oral mucosal findings and assist in referral triage. "
            "CRITICAL RULES:\n"
            "1. You do NOT diagnose (e.g. do not say 'the patient has Oral Cancer' or 'this is benign').\n"
            "2. Objectively describe visible features: color, margins, ulceration, plaque texture, and anatomical location.\n"
            "3. Assess risk factors: chewing khaini, gutka, supari, bidi smoking, alcohol, and lesion duration.\n"
            "4. Triage rule: any persistent mucosal alteration (>2 weeks) in a chronic tobacco/areca nut user MUST be triaged as 'priority' or 'urgent'.\n"
            "5. If visual evidence is ambiguous, escalate urgency (safe-by-design).\n"
            "6. Always state a licensed human clinician must examine the patient in person.\n"
            "7. Reply STRICTLY with a JSON object matching this schema:\n"
            "{\n"
            '  "urgency": "routine" | "priority" | "urgent",\n'
            '  "visible_findings": ["finding 1", "finding 2"],\n'
            '  "risk_factors": ["habit 1", "duration factor"],\n'
            '  "suggested_next_step": "e.g. District hospital dental review within 10 days",\n'
            '  "image_quality_notes": "notes on glare, focus, or field of view",\n'
            '  "confidence_note": "A human clinician must confirm all observations in person.",\n'
            '  "patient_summary_en": "Simple, respectful, non-alarming explanation in English.",\n'
            '  "patient_summary_hi": "Simple, respectful explanation in Hindi (Devanagari script)."\n'
            "}"
        )

        user_content = (
            f"Patient Clinical Questionnaire:\n"
            f"- Age: {questionnaire.get('age', 'Unknown')}\n"
            f"- Sex: {questionnaire.get('sex', 'Unknown')}\n"
            f"- Anatomical Site: {questionnaire.get('anatomical_site', 'Oral cavity')}\n"
            f"- Habits: {', '.join(questionnaire.get('habits', ['None reported']))}\n"
            f"- Lesion Duration: {questionnaire.get('lesion_duration_weeks', 'Unknown')} weeks\n"
            f"- Reported Symptoms: {', '.join(questionnaire.get('symptoms', ['None']))}\n"
            f"{area_note}\n\n"
            f"Analyze this intraoral image alongside the clinical questionnaire and produce the structured referral JSON."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    ],
                },
            ],
            temperature=0.1,
        )

        raw_text = response.choices[0].message.content.strip()

        # Clean Markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("\n", 1)[0]
        raw_text = raw_text.strip()

        # Validate with Pydantic
        referral = ReferralOutput.model_validate_json(raw_text)

        # Cache response
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(referral.model_dump_json(indent=2))

        return referral