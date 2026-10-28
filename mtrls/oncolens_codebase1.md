# OncoLens: Complete Working Codebase (Days 1 & 2)

This document contains the final, tested, and verified versions of every code file created for OncoLens. All intermediate debugging attempts, typos, and broken snippets have been omitted—only the working final code is included.

---

## Directory Structure

```text
oncolens/
├── .env.example
├── .gitignore
├── pipeline.py                 # End-to-end runner (Quality -> SAM 3.1 -> Reasoner)
├── core/
│   ├── __init__.py
│   ├── quality.py              # OpenCV quality gate (Laplacian blur, HSV glare, CLAHE)
│   ├── segment.py              # SAM 3.1 wrapper (Token parser, box extraction, area ratio)
│   └── reason.py               # Multimodal clinical reasoner with Pydantic schema & cache
├── data/
│   └── test_images/
│       ├── sample1.jpeg        # Clinical oral photo
│       └── answers_1.json      # Structured patient questionnaire
├── outputs/                    # Output directory for results, masks, overlays, and cache
└── tests/
    ├── smoke_test.py           # Multimodal vision smoke test
    ├── test_sam.py             # SAM 3.1 box detection test
    ├── test_quality.py         # Quality gate verification
    ├── test_segmenter.py       # Segmenter module verification
    └── test_reason.py          # Reasoner module verification
```

---

## 1. Environment & Configuration

### `.env.example`
```env
MODEL_API_KEY=your_meta_model_api_key_here
```

### `.gitignore`
```gitignore
# 1. CRITICAL: Never commit your API key or secrets
.env
*.env
.env.local

# 2. Python Virtual Environments
.venv/
venv/
env/
ENV/

# 3. Python Bytecode and Cache
__pycache__/
*.py[cod]
*$py.class
*.pyd

# 4. Large Dataset Downloads & Temporary Outputs
*.zip
*.tar.gz
outputs/*.jpg
outputs/*.png
outputs/*.json
!outputs/.gitkeep

# 5. OS and Editor Artifacts
.DS_Store
Thumbs.db
.vscode/
.idea/
```

---

## 2. Core Modules

### `core/__init__.py`
*(Empty file to designate `core` as a Python package)*

---

### `core/quality.py`
```python
import cv2
import numpy as np
from typing import Dict, Any

class ImageQualityGate:
    """
    Automated pre-flight image quality checker for intraoral photography.
    Evaluates blur via Laplacian variance and specular glare via HSV thresholding.
    Applies Gray-World White Balance and CLAHE for contrast enhancement.
    """
    def __init__(self, blur_threshold: float = 100.0, glare_threshold: float = 0.08):
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold

    def calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """Computes focus sharpness using the variance of the Laplacian operator."""
        return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    def calculate_glare_ratio(self, bgr_image: np.ndarray) -> float:
        """
        Detects specular glare (saliva reflection & direct camera flash).
        Specular glare in intraoral photography exhibits high Value (brightness) 
        and low Saturation in HSV color space.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        
        # Pixels with brightness > 220 and saturation < 45 are specular reflections
        glare_mask = (v > 220) & (s < 45)
        glare_pixels = np.count_nonzero(glare_mask)
        total_pixels = bgr_image.shape[0] * bgr_image.shape[1]
        
        return float(glare_pixels / total_pixels)

    def enhance_contrast_clahe(self, bgr_image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization on the Luminance channel."""
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def gray_world_white_balance(self, bgr_image: np.ndarray) -> np.ndarray:
        """Corrects yellowish flashlight cast using Gray-World assumption."""
        b, g, r = cv2.split(bgr_image.astype(np.float32))
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        k = (b_avg + g_avg + r_avg) / 3.0
        
        b = np.clip(b * (k / (b_avg + 1e-5)), 0, 255)
        g = np.clip(g * (k / (g_avg + 1e-5)), 0, 255)
        r = np.clip(r * (k / (r_avg + 1e-5)), 0, 255)
        
        return cv2.merge([b, g, r]).astype(np.uint8)

    def evaluate(self, image_path: str) -> Dict[str, Any]:
        """
        Runs quality gate on an image file.
        Returns whether the image passed or failed, along with concrete feedback.
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image from '{image_path}'")
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        blur_score = self.calculate_blur_score(gray)
        glare_ratio = self.calculate_glare_ratio(bgr)
        
        issues = []
        is_blurry = blur_score < self.blur_threshold
        has_excessive_glare = glare_ratio > self.glare_threshold
        
        if is_blurry:
            issues.append(f"Image is out of focus (sharpness score {blur_score:.1f} < {self.blur_threshold}). Please hold camera steady.")
        if has_excessive_glare:
            issues.append(f"Excessive flash/saliva glare ({glare_ratio*100:.1f}% area). Please tilt camera slightly to avoid direct reflection.")
            
        is_acceptable = (not is_blurry) and (not has_excessive_glare)
        
        # Color & contrast enhancement for downstream models
        balanced = self.gray_world_white_balance(bgr)
        enhanced = self.enhance_contrast_clahe(balanced)
        
        return {
            "is_acceptable": is_acceptable,
            "blur_score": round(blur_score, 2),
            "glare_ratio": round(glare_ratio, 4),
            "issues": issues,
            "enhanced_image": enhanced
        }
```

---

### `core/segment.py`
```python
import os
import base64
import json
import re
import cv2
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load .env automatically whenever this module is imported
load_dotenv()

class LesionSegmenter:
    """
    Production wrapper for Meta SAM 3.1 over the Responses API.
    Streams token grammar, extracts bounding boxes, draws visual overlays,
    and calculates relative lesion area ratio.
    """
    def __init__(self, api_key: str = None, base_url: str = "https://api.meta.ai/v1"):
        self.api_key = api_key or os.getenv("MODEL_API_KEY")
        if not self.api_key:
            raise ValueError("MODEL_API_KEY is not set.")
        self.base_url = base_url

    def segment(self, image_path: str, primary_concept: str = "white patch") -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at '{image_path}'")

        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        total_pixels = img_h * img_w

        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # Attempt primary concept, fallback to 'oral lesion' if no match
        concepts_to_try = [primary_concept, "oral lesion"] if primary_concept != "oral lesion" else ["oral lesion"]
        box_matches = []
        matched_concept = None

        for concept in concepts_to_try:
            payload = {
                "model": "sam-3.1",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": concept},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_image}"},
                        ],
                    }
                ],
                "stream": True,
            }

            resp = requests.post(url, headers=headers, json=payload, stream=True)
            if resp.status_code != 200:
                continue

            raw_tokens = []
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    if event.get("type") == "response.output_text.delta":
                        raw_tokens.append(event.get("delta", ""))
                except json.JSONDecodeError:
                    continue

            stream_text = "".join(raw_tokens)
            matches = re.findall(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", stream_text)
            if matches:
                box_matches = matches
                matched_concept = concept
                break

        overlay = img.copy()
        regions: List[Dict[str, int]] = []
        total_lesion_pixels = 0

        for i, (x1, y1, x2, y2, w, h) in enumerate(box_matches):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            total_lesion_pixels += box_area

            regions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area_px": box_area})

            # Draw visual highlight for clinician reviewer view
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                f"SAM 3.1: {matched_concept}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        area_ratio = round(float(total_lesion_pixels / total_pixels), 4)

        return {
            "has_lesion_detected": len(regions) > 0,
            "matched_concept": matched_concept,
            "region_count": len(regions),
            "regions": regions,
            "area_ratio": area_ratio,
            "overlay_image": overlay,
        }
```

---

### `core/reason.py`
```python
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
```

---

## 3. End-to-End Pipeline

### `pipeline.py`
```python
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
```

---

## 4. Test Data

### `data/test_images/answers_1.json`
```json
{
  "age": 48,
  "sex": "Male",
  "anatomical_site": "Left buccal mucosa",
  "habits": [
    "Smokeless tobacco (Khaini) 5 times daily for 18 years",
    "Occasional bidi smoking (2-3/day)"
  ],
  "lesion_duration_weeks": 6,
  "symptoms": [
    "Burning sensation with spicy food",
    "Rough, non-scrapable surface texture",
    "No spontaneous bleeding"
  ]
}
```

---

## 5. Test Scripts

### `smoke_test.py`
```python
import os
import base64
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("MODEL_API_KEY")
if not api_key:
    raise ValueError("MODEL_API_KEY is missing from your .env file!")

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=api_key,
)

image_path = "data/test_images/sample1.jpeg"
if not os.path.exists(image_path):
    image_path = "data/test_images/sample1.jpg"

with open(image_path, "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

print(f"[*] Loaded image: {image_path}")
print("[*] Sending multimodal request to Meta Model API (muse-spark-1.1)...")

MODEL_NAME = "muse-spark-1.1"

system_prompt = (
    "You are a clinical lesion documentation copilot for frontline health workers. "
    "You do NOT diagnose pathology. You objectively describe visual features (color, margins, texture, anatomical site) "
    "and output strictly valid JSON matching this schema:\n"
    "{\n"
    '  "visible_findings": ["feature 1", "feature 2"],\n'
    '  "image_quality": "good | acceptable | poor",\n'
    '  "quality_issues": ["glare", "blur"] or [],\n'
    '  "clinical_note": "objective description"\n'
    "}"
)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the visible lesion characteristics in this intraoral photo. Do not diagnose."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            },
        ],
        temperature=0.1,
    )

    raw_content = response.choices[0].message.content.strip()
    if raw_content.startswith("```"):
        raw_content = raw_content.split("\n", 1)[1]
    if raw_content.endswith("```"):
        raw_content = raw_content.rsplit("\n", 1)[0]
    raw_content = raw_content.strip()

    parsed = json.loads(raw_content)
    print("\n--- Parsed JSON Output ---")
    print(json.dumps(parsed, indent=2))
    print("\n[SUCCESS] Day 1 Multimodal Smoke Test Passed!")

except Exception as e:
    print(f"\n[ERROR] API call failed: {e}")
```

### `test_sam.py`
```python
import os
import base64
import json
import re
import cv2
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MODEL_API_KEY")
if not api_key:
    raise ValueError("MODEL_API_KEY is missing from .env!")

image_path = "data/test_images/sample1.jpeg"
if not os.path.exists(image_path):
    image_path = "data/test_images/sample1.jpg"

print(f"[*] Reading image: {image_path}")
with open(image_path, "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

CONCEPT = "white patch"
print(f"[*] Calling SAM 3.1 with concept: '{CONCEPT}'...")

url = "https://api.meta.ai/v1/responses"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
}

payload = {
    "model": "sam-3.1",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": CONCEPT},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_image}"},
            ],
        }
    ],
    "stream": True,
}

response = requests.post(url, headers=headers, json=payload, stream=True)
response.raise_for_status()

raw_tokens = []
for raw_line in response.iter_lines(decode_unicode=False):
    if not raw_line: continue
    line = raw_line.decode("utf-8", errors="replace")
    if not line.startswith("data: "): continue
    data = line.removeprefix("data: ")
    if data == "[DONE]": break
    try:
        event = json.loads(data)
        if event.get("type") == "response.output_text.delta":
            raw_tokens.append(event.get("delta", ""))
    except json.JSONDecodeError:
        continue

full_stream = "".join(raw_tokens)
box_matches = re.findall(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", full_stream)

if not box_matches:
    print(f"\n[!] No match for '{CONCEPT}'. Retrying with 'oral lesion'...")
    payload["input"][0]["content"][0]["text"] = "oral lesion"
    resp2 = requests.post(url, headers=headers, json=payload, stream=True)
    tokens2 = []
    for raw_line in resp2.iter_lines(decode_unicode=False):
        if not raw_line: continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "): continue
        data = line.removeprefix("data: ")
        if data == "[DONE]": break
        try:
            event = json.loads(data)
            if event.get("type") == "response.output_text.delta":
                tokens2.append(event.get("delta", ""))
        except: continue
    full_stream = "".join(tokens2)
    box_matches = re.findall(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", full_stream)

if not box_matches:
    print("[!] Still no box detected. Make sure the photo shows clear lesion tissue.")
    exit()

print(f"\n[+] SUCCESS! SAM 3.1 identified {len(box_matches)} lesion region(s).")
img = cv2.imread(image_path)
overlay = img.copy()

for i, (x1, y1, x2, y2, w, h) in enumerate(box_matches):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(f" -> Region #{i+1}: Top-Left=({x1}, {y1}), Bottom-Right=({x2}, {y2})")
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        overlay,
        f"Lesion #{i+1}",
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/sample1_detected.jpg"
cv2.imwrite(output_path, overlay)
print(f"\n[DONE] Saved visual detection result to: {output_path}")
```

### `test_quality.py`
```python
import os
import cv2
from core.quality import ImageQualityGate

gate = ImageQualityGate()

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
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/sample1_enhanced.jpg", result['enhanced_image'])
    print(" -> Saved color/contrast-enhanced copy to 'outputs/sample1_enhanced.jpg'")
```

### `test_segmenter.py`
```python
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
```

### `test_reason.py`
```python
import os
import json
from core.reason import ClinicalReasoner

reasoner = ClinicalReasoner()

image_path = "data/test_images/sample1.jpeg"
if not os.path.exists(image_path):
    image_path = "data/test_images/sample1.jpg"

with open("data/test_images/answers_1.json", "r") as f:
    questionnaire = json.load(f)

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
```
