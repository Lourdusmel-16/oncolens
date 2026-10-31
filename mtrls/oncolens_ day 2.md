# OncoLens: Comprehensive Project Architecture, Implementation Journey & Codebase Reference

**Project Title:** OncoLens — Oral Lesion Documentation & Clinical Referral Triage Copilot  
**Author / Team:** OncoLens Team  
**Date:** October 29, 2026  
**Target Deployment:** Frontline Community Health Workers (ASHA / ANM) and District Hospital Dentists (Rural India / Patna, Bihar Focus)  
**Core Technologies:** Meta SAM 3.1 (Segment Anything Model), Meta Model API (Muse Spark Multimodal LLM), Python, OpenCV, Streamlit, SQLite

---

## 1. Executive Summary & Clinical Mission

Oral cancer accounts for over 30% of all cancer cases in India, predominantly concentrated in rural and peri-urban populations with widespread use of chewable smokeless tobacco (Khaini, Gutka, Zarda), areca nut (Supari/Paan), and bidi smoking. Over 70% of patients present at Stage III or Stage IV, where 5-year survival drops below 20%, largely because rural primary health centres (PHCs) lack dental surgeons or oral pathologists. 

**OncoLens** addresses this critical diagnostic bottleneck through an **offline-first, human-in-the-loop screening copilot**. It bridges frontline Accredited Social Health Activists (ASHAs) and auxiliary nurse-midwives (ANMs) with district dental specialists.

### Core System Philosophy
1. **Assistive Documentation, Not Autonomous Diagnosis:** OncoLens never claims to diagnose cancer, carcinoma, or malignancy. It objectively documents visible mucosal features, localizes lesions, quantifies pixel area, and triages urgency.
2. **Deterministic Safety Floors:** High-risk screening decisions cannot be left to probabilistic AI hallucinations. If a patient exhibits persistent mucosal alterations (>2 weeks) alongside chronic tobacco use or clinical red flags, hardcoded clinical safety rules force an urgent or priority referral floor, regardless of model confidence.
3. **Anatomical Negative Constraints:** Foundation vision models (like SAM 3.1) are trained on natural open-world images and frequently mistake teeth enamel and dental tartar for leukoplakia. OncoLens engineers SAM against itself by segmenting teeth as an anatomical negative mask, eliminating dental false positives.
4. **Point-of-Care Quality Assurance:** Prevents unusable, blurry, or overexposed photos from entering the clinical queue through a pre-flight image quality gate.
5. **Culturally Grounded Communication:** Automatically generates patient counseling summaries in respectful, non-alarming Devanagari Hindi alongside formal English clinical referral notes.

---

## 2. Chronological Engineering Journey (Days 1 to 4)

### Day 1 & Day 2: Foundational Architecture & Decoupling
* Established clean separation of concerns across 5 modular components:
  * `core/quality.py`: Image pre-flight checks.
  * `core/segment.py`: SAM 3.1 API wrapper and token parsing.
  * `core/reason.py`: Multimodal reasoner with Pydantic validation.
  * `core/safety.py`: Deterministic urgency floors and anti-diagnosis wording filters.
  * `pipeline.py`: Unified end-to-end execution service.

### Day 3: Execution, Critical Failure Discovery & The Pivot
* **Task A Setup & Token Parsing:** Configured streaming Server-Sent Events (SSE) parsing for SAM 3.1 bounding box and one-bit raster mask tokens using `meta-sam-parser`.
* **The "Zero Region" Bottleneck:** When running `test_segmenter.py` on `sample1.jpeg`, specific prompt concepts (`"white patch"`, `"red patch"`, `"ulcer"`) returned zero bounding boxes.
* **Diagnostics via `debug_sam.py`:** Tested multiple prompts against the live Meta API:
  * Control prompt `"tooth"`: Returned 12 bounding boxes with 12 parsed masks (confirming the API connection, streaming decoder, and mask reconstruction were 100% functional).
  * Prompt `"oral lesion"`: Returned 1 bounding box with 1 parsed mask.
* **The Critical Failure Mode (Teeth False Positive):**
  * Visual inspection of `outputs/sample1_masks.jpg` revealed that the single mask produced by `"oral lesion"` was positioned directly over the patient's lower molars and premolars.
  * *Root Cause Analysis:* SAM 3.1 is not a pathology-specific model; when forced with a generic term, it latched onto the high-contrast textured enamel and staining of the teeth rather than the subtle mucosal pigmentation on the cheek.
* **The Engineering Breakthrough (Anatomical Negative Masking):**
  * Rather than relying on manual unticking, we leveraged SAM's high accuracy at recognizing teeth (12/12 teeth detected).
  * Implemented an anatomical negative filter in `core/segment.py`: SAM generates a composite `tooth_mask`. Any candidate lesion mask overlapping >35% with the tooth mask is automatically rejected as a dental false positive.
  * *Result:* The false green mask over healthy teeth was completely eliminated (`Selected area: 0.0% of image`), and the clean original photo was preserved for clinical evaluation.
* **Field & Reviewer Web Application (`app.py`):**
  * Built a two-tab Streamlit application featuring an offline SQLite queue (`core/store.py`), real-time quality gate feedback, side-by-side mask review, and interactive clinician sign-off.

### Day 4: Clinical UX Polish, Queue Management & Bilingual Delivery
* Integrated structured clinical referral cards distinguishing objective visible findings from epidemiological risk profiles.
* Added side-by-side English and Devanagari Hindi patient guidance cards.
* Added empty-mask informative banners explaining that zero segmented boundaries do not imply normal tissue, highlighting safety floor activations.
* Built a one-click downloadable printable clinical referral slip (`.txt`).
* Implemented queue management operations (`delete_case` and `clear_all_cases`) while preserving the `outputs/cache/` directory to protect expensive API tokens.

---

## 3. Detailed Component Architecture

| Component | File Path | Primary Function | Key Algorithms / Models |
| :--- | :--- | :--- | :--- |
| **Image Quality Gate** | `core/quality.py` | Point-of-care blur and saliva/flash glare screening | Laplacian variance, HSV specular thresholding, CLAHE contrast |
| **Lesion Segmenter** | `core/segment.py` | Localizes mucosal alterations; filters dental structures | Meta SAM 3.1 streaming API, `meta-sam-parser`, Tooth Negative Filter |
| **Clinical Reasoner** | `core/reason.py` | Synthesizes image tokens + risk history into triage memo | Meta Model API (`muse-spark-1.1`), Pydantic schema enforcement |
| **Safety Guardrails** | `core/safety.py` | Deterministic urgency floors & banned terminology filter | Regex vocabulary sanitizer, Protocol-driven triage escalation |
| **Offline Case Store** | `core/store.py` | Local SQLite queue for asynchronous field sync | SQLite3 relational store, JSON case payload serialization |
| **Service Pipeline** | `pipeline.py` | Orchestrates end-to-end evaluation & artifact export | OpenCV overlay generation, file system artifact persistence |
| **Clinical Web App** | `app.py` | Frontline field capture & district dentist review portal | Streamlit reactive web framework, custom CSS cards |

---

## 4. Complete Codebase Reference

### 4.1 Image Preparation Helper (`core/utils.py`)
Standardizes all input photos to a maximum dimension of 1280px. Sending identical JPEG bytes to both SAM 3.1 and Muse Spark ensures coordinate spaces match across masks, bounding boxes, and overlays.

```python
"""Shared helpers: image preparation used by every stage."""
import cv2
import numpy as np
from typing import Tuple

MAX_SIDE = 1280  # Keeps uploads small, fast, and economical on API quota

def load_resized(image_path: str, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, bytes]:
    """Load an image (EXIF-rotated), shrink so long side <= max_side.
    Returns (bgr_array, jpeg_bytes). Sending identical bytes to SAM and Muse Spark
    ensures coordinate spaces match across masks, boxes, and overlays."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image from '{image_path}'")
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bgr, buf.tobytes()
```

---

### 4.2 Image Quality Gate (`core/quality.py`)
Evaluates intraoral photos for motion blur (Laplacian variance) and specular reflections from camera flash or saliva (HSV thresholding).

```python
"""Pre-flight quality checks for intraoral photos (blur, glare) + enhancement."""
import cv2
import numpy as np
from typing import Dict, Any
from core.utils import load_resized

class ImageQualityGate:
    def __init__(self, blur_threshold: float = 100.0, glare_threshold: float = 0.08):
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold

    def calculate_blur_score(self, gray_image: np.ndarray) -> float:
        return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    def calculate_glare_ratio(self, bgr_image: np.ndarray) -> float:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        glare_mask = (v > 220) & (s < 45)
        return float(np.count_nonzero(glare_mask) / (bgr_image.shape[0] * bgr_image.shape[1]))

    def enhance_contrast_clahe(self, bgr_image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def evaluate(self, image_path: str, white_balance: bool = False) -> Dict[str, Any]:
        bgr, _ = load_resized(image_path, max_side=1024)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur_score = self.calculate_blur_score(gray)
        glare_ratio = self.calculate_glare_ratio(bgr)

        issues = []
        is_blurry = blur_score < self.blur_threshold
        too_much_glare = glare_ratio > self.glare_threshold

        if is_blurry:
            issues.append(f"Image is out of focus (sharpness {blur_score:.1f} < {self.blur_threshold}). Hold camera steady and retake.")
        if too_much_glare:
            issues.append(f"Excessive flash/saliva glare ({glare_ratio*100:.1f}% of image). Tilt camera slightly and retake.")

        return {
            "is_acceptable": (not is_blurry) and (not too_much_glare),
            "blur_score": round(blur_score, 2),
            "glare_ratio": round(glare_ratio, 4),
            "issues": issues,
            "enhanced_image": self.enhance_contrast_clahe(bgr),
        }
```

---

### 4.3 Lesion Segmenter with Anatomical Negative Masking (`core/segment.py`)
Implements real pixel mask decoding via `meta_sam_parser` and incorporates the **Tooth Filter** to eliminate dental false positives.

```python
import os
import re
import json
import time
import base64
import hashlib
from typing import Dict, Any, List, Optional, Sequence

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from core.utils import load_resized

load_dotenv()

RECORD_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>"
    r"<\|mask;x=\d+;y=\d+;(?:c=[\d.]+;)?data=(\d+),(\d+),([^|<>;,"\]+)\|>"
)

DEFAULT_CONCEPTS = ("white patch", "red patch", "ulcer")
FALLBACK_CONCEPT = "oral lesion"
ANATOMICAL_NEGATIVE = "tooth"

class LesionSegmenter:
    """SAM 3.1 wrapper with anatomical negative filtering (teeth suppression)."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.meta.ai/v1",
                 cache_dir: str = "outputs/cache"):
        self.api_key = api_key or os.getenv("MODEL_API_KEY")
        if not self.api_key:
            raise ValueError("MODEL_API_KEY is not set.")
        self.base_url = base_url
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _call_sam(self, jpeg_bytes: bytes, concept: str, use_cache: bool = True) -> str:
        key = hashlib.sha256(jpeg_bytes + concept.encode()).hexdigest()[:24]
        cache_path = os.path.join(self.cache_dir, f"sam_{key}.txt")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        payload = {
            "model": "sam-3.1",
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ]}],
            "stream": True,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "Accept": "text/event-stream"}

        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(f"{self.base_url}/responses", headers=headers,
                                     json=payload, stream=True, timeout=120)
                if resp.status_code == 200:
                    break
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"SAM request failed ({last_err})")
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"SAM request failed after retries ({last_err})")

        parts: List[str] = []
        for raw in resp.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "response.output_text.delta":
                parts.append(ev.get("delta", ""))
            elif ev.get("type") in ("error", "response.failed", "response.incomplete"):
                raise RuntimeError(f"SAM stream error: {data[:300]}")

        text = "".join(parts)
        if text:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @staticmethod
    def parse_regions(stream_text: str, img_w: int, img_h: int, concept: str) -> List[Dict[str, Any]]:
        regions = []
        for m in RECORD_RE.finditer(stream_text):
            x1, y1, x2, y2, w, h, conf, mh, mw, payload = m.groups()
            x1, y1, x2, y2, w, h, mh, mw = map(int, (x1, y1, x2, y2, w, h, mh, mw))
            enc = "lossless" if payload[0] == "~" else "one_bit"
            small = np.frombuffer(
                decode_mask_to_raster(SegmentationMask(encoding=enc, payload=payload,
                                                       width=mw, height=mh)),
                dtype=np.uint8).reshape(mh, mw)

            sx, sy = img_w / w, img_h / h
            full = np.zeros((img_h, img_w), dtype=np.uint8)
            if abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6:
                bx1, by1 = x1, y1
            else:
                bx1, by1 = int(x1 * sx), int(y1 * sy)
                small = cv2.resize(small, (max(1, int(mw * sx)), max(1, int(mh * sy))),
                                   interpolation=cv2.INTER_NEAREST)
            hh, ww = small.shape
            ye, xe = min(img_h, by1 + hh), min(img_w, bx1 + ww)
            full[by1:ye, bx1:xe] = small[: ye - by1, : xe - bx1]
            regions.append({
                "concept": concept,
                "box": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                "confidence": float(conf) if conf is not None else None,
                "mask": full,
                "area_px": int(full.sum()),
            })
        return regions

    def segment(self, image_path: str,
                concepts: Sequence[str] = DEFAULT_CONCEPTS,
                site: Optional[str] = None,
                min_confidence: float = 0.3,
                use_cache: bool = True,
                filter_teeth: bool = True) -> Dict[str, Any]:
        bgr, jpeg = load_resized(image_path)
        img_h, img_w = bgr.shape[:2]

        # 1. Negative Anatomical Masking: Detect teeth structures
        tooth_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if filter_teeth:
            tooth_text = self._call_sam(jpeg, ANATOMICAL_NEGATIVE, use_cache=use_cache)
            teeth_regions = self.parse_regions(tooth_text, img_w, img_h, concept=ANATOMICAL_NEGATIVE)
            for t in teeth_regions:
                tooth_mask |= t["mask"]

        # 2. Context-Guided Site Concept Expansion
        target_concepts = list(concepts)
        if site:
            clean_site = site.lower().replace("left ", "").replace("right ", "").replace("mucosa", "").strip()
            if clean_site and clean_site not in ("other / multiple", "oral cavity"):
                site_concept = f"{clean_site} lesion"
                if site_concept not in target_concepts:
                    target_concepts.insert(0, site_concept)

        # 3. Query candidate concepts with fallback
        candidates: List[Dict[str, Any]] = []
        all_prompts = list(target_concepts) + [FALLBACK_CONCEPT]

        for concept in all_prompts:
            if concept == FALLBACK_CONCEPT and candidates:
                break
            text = self._call_sam(jpeg, concept, use_cache=use_cache)
            for r in self.parse_regions(text, img_w, img_h, concept):
                if r["confidence"] is None or r["confidence"] >= min_confidence:
                    candidates.append(r)

        # 4. Filter dental false positives
        valid_regions: List[Dict[str, Any]] = []
        has_teeth = tooth_mask.sum() > 0

        for r in candidates:
            if has_teeth:
                overlap_px = int(np.logical_and(r["mask"], tooth_mask).sum())
                overlap_ratio = overlap_px / max(1, r["area_px"])
                # If >35% of candidate mask is on teeth, reject as dental artifact
                if overlap_ratio > 0.35:
                    continue
                # If it slightly touches teeth, subtract dental pixels
                if overlap_px > 0:
                    cleaned_mask = r["mask"] & ~tooth_mask
                    if cleaned_mask.sum() < 50:
                        continue
                    r["mask"] = cleaned_mask
                    r["area_px"] = int(cleaned_mask.sum())
            valid_regions.append(r)

        for i, r in enumerate(valid_regions):
            r["id"] = i

        return {"image_bgr": bgr, "regions": valid_regions, "img_w": img_w, "img_h": img_h, "teeth_detected": bool(has_teeth)}

    @staticmethod
    def area_ratio(regions: List[Dict[str, Any]], selected_ids: Optional[Sequence[int]] = None) -> float:
        chosen = [r for r in regions if selected_ids is None or r["id"] in selected_ids]
        if not chosen:
            return 0.0
        union = np.zeros_like(chosen[0]["mask"], dtype=np.uint8)
        for r in chosen:
            union |= r["mask"]
        return round(float(union.sum()) / union.size, 4)

    @staticmethod
    def render_overlay(bgr: np.ndarray, regions: List[Dict[str, Any]],
                       selected_ids: Optional[Sequence[int]] = None) -> np.ndarray:
        out = bgr.copy()
        palette = {"white patch": (0, 215, 255), "red patch": (0, 0, 255), "ulcer": (255, 0, 255),
                   "oral lesion": (0, 255, 0), "lesion": (0, 255, 0)}
        for r in regions:
            if selected_ids is not None and r["id"] not in selected_ids:
                continue
            color = palette.get(r["concept"], (0, 255, 0))
            m = r["mask"].astype(bool)
            tint = out.copy()
            tint[m] = color
            out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
            cnts, _ = cv2.findContours(r["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2)
            x1, y1, _, _ = r["box"]
            conf = f" {r['confidence']:.2f}" if r["confidence"] is not None else ""
            cv2.putText(out, f"#{r['id']} {r['concept']}{conf}", (x1, max(y1 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return out
```

---

### 4.4 Clinical Reasoner (`core/reason.py`)
Queries the Meta Model API (`muse-spark-1.1`), enforces strict Pydantic JSON schemas, prevents diagnostic claims, and generates bilingual summaries.

```python
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
    "and district dentists in India. You document oral mucosal findings and assist referral triage.\n"
    "CRITICAL RULES:\n"
    "1. You do NOT diagnose. Never write the words cancer, carcinoma, malignant, benign, tumour or precancer.\n"
    "2. Objectively describe visible features only: color, margins, ulceration, texture, anatomical location.\n"
    "3. Assess risk factors: khaini, gutka, supari/areca nut, bidi/smoking, alcohol, lesion duration.\n"
    "4. Any persistent mucosal change (>2 weeks) in a chronic tobacco/areca user MUST be 'priority' or 'urgent'.\n"
    "5. If evidence is ambiguous, escalate urgency (safe-by-design). Never answer 'routine' for a visible lesion.\n"
    "6. Always state that a licensed clinician must examine the patient in person.\n"
    "7. If the photo is not an intraoral photo, set image_is_intraoral to false.\n"
    "8. Reply STRICTLY with one JSON object matching the required schema."
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
        for _ in range(2):
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
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Model returned invalid JSON twice: {last_err}")
```

---

### 4.5 Safety Guardrails & Urgency Escalation (`core/safety.py`)
Enforces clinical safety policies: the LLM may raise urgency, but deterministic rules establish an irreversible floor.

```python
"""Deterministic safety layer. The LLM can raise urgency but can never lower it below these floors."""
import re
from typing import Dict, Any, List

RANK = {"routine": 0, "priority": 1, "urgent": 2}
NAME = {v: k for k, v in RANK.items()}

PERSISTENT_WEEKS = 2  # Persistent mucosal change with tobacco habit -> at least priority
URGENT_WEEKS = 3      # Persistent change with clinical red flag -> urgent

HABIT_WORDS = ("tobacco", "khaini", "gutka", "gutkha", "supari", "areca", "betel", "paan",
               "bidi", "beedi", "cigarette", "smok", "alcohol")

BANNED = re.compile(r"carcinoma|cancer|malignan|benign|tumou?r|pre-?cancer", re.I)

SAFE_EN = ("A change was seen in your mouth. Please visit the district hospital dentist soon for a proper "
           "check. This is not a diagnosis; only a doctor can examine you and tell you what it is.")
SAFE_HI = (
    "\u0906\u092a\u0915\u0947 \u092e\u0941\u0901\u0939 \u092e\u0947\u0902 \u090f\u0915 "
    "\u092c\u0926\u0932\u093e\u0935 \u0926\u093f\u0916\u093e\u0908 \u0926\u093f\u092f\u093e "
    "\u0939\u0948\u0964 \u0915\u0943\u092a\u092f\u093e \u091c\u0932\u094d\u0926 \u0939\u0940 "
    "\u091c\u093f\u0932\u093e \u0905\u0938\u094d\u092a\u0924\u093e\u0932 \u0915\u0947 "
    "\u0926\u0902\u0924 \u091a\u093f\u0915\u093f\u0924\u094d\u0938\u0915 \u0938\u0947 "
    "\u091c\u093e\u0901\u091a \u0915\u0930\u0935\u093e\u090f\u0901\u0964 \u092f\u0939 "
    "\u0915\u094b\u0908 \u0928\u093f\u0926\u093e\u0928 \u0928\u0939\u0940\u0902 \u0939\u0948; "
    "\u0915\u0947\u0925\u0932 \u0921\u0949\u0915\u094d\u091f\u0930 \u0939\u0940 \u091c\u093e\u0901\u091a "
    "\u0915\u0930\u0915\u0947 \u092c\u0924\u093e \u0938\u0915\u0924\u0947 \u0939\u0948\u0902\u0964"
)

RED_FLAG_KEYS = ("non_healing_ulcer", "bleeding", "neck_lump", "hard_or_fixed",
                 "numbness", "difficulty_opening_mouth")

def has_risk_habit(habits: List[str]) -> bool:
    text = " ".join(habits or []).lower()
    return any(w in text for w in HABIT_WORDS)

def apply_safety(referral: Dict[str, Any], questionnaire: Dict[str, Any], region_count: int) -> Dict[str, Any]:
    flags: List[str] = []
    out = dict(referral)
    llm = out["urgency"]
    floor = "routine"

    weeks = questionnaire.get("lesion_duration_weeks")
    habit = has_risk_habit(questionnaire.get("habits", []))
    red = questionnaire.get("red_flags", {}) or {}
    n_red = sum(1 for k in RED_FLAG_KEYS if red.get(k))

    if habit and weeks is not None and weeks >= PERSISTENT_WEEKS:
        floor = "priority"
        flags.append(f"Floor 'priority': lesion >= {PERSISTENT_WEEKS} weeks with risk habit.")

    if weeks is not None and weeks >= URGENT_WEEKS and n_red >= 1:
        floor = "urgent"
        flags.append(f"Floor 'urgent': lesion >= {URGENT_WEEKS} weeks with red flag(s).")

    if n_red >= 2:
        floor = "urgent"
        flags.append("Floor 'urgent': two or more red flags.")

    if region_count == 0 and floor == "routine" and habit:
        floor = "priority"
        flags.append("No segmented region, but risk habit present: not treated as routine.")

    final = NAME[max(RANK[llm], RANK[floor])]
    if final != llm:
        flags.append(f"Urgency raised from '{llm}' to '{final}' by safety rules.")

    kept = [f for f in out["visible_findings"] if not BANNED.search(f)]
    if len(kept) != len(out["visible_findings"]):
        flags.append("Removed diagnostic wording from findings.")
    out["visible_findings"] = kept

    if BANNED.search(out["patient_summary_en"]) or BANNED.search(out["patient_summary_hi"]):
        flags.append("Patient summary replaced with safe template (diagnostic wording detected).")
        out["patient_summary_en"], out["patient_summary_hi"] = SAFE_EN, SAFE_HI

    out["urgency"] = final
    return {"referral": out, "urgency_llm": llm, "urgency_final": final, "flags": flags}
```

---

### 4.6 Offline Queue Store (`core/store.py`)
Provides an offline-first SQLite database supporting case ingestion, batch syncing, review state updates, and safe queue resets.

```python
"""Tiny SQLite queue for the offline-first field app."""
import os
import json
import sqlite3
import uuid
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("ONCOLENS_DB", "outputs/oncolens.db")
CASE_ROOT = "outputs/cases"

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS cases(
        id TEXT PRIMARY KEY, created TEXT, status TEXT, image_path TEXT,
        questionnaire TEXT, result TEXT, review TEXT)""")
    return conn

def new_case(image_bytes: bytes, questionnaire: Dict[str, Any]) -> str:
    case_id = datetime.now().strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    folder = os.path.join(CASE_ROOT, case_id)
    os.makedirs(folder, exist_ok=True)
    image_path = os.path.join(folder, "photo.jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    with _conn() as c:
        c.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?)",
                  (case_id, datetime.now().isoformat(timespec="seconds"), "pending",
                   image_path, json.dumps(questionnaire), None, None))
    return case_id

def list_cases(status: Optional[str] = None) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM cases" + (" WHERE status=?" if status else "") +
                         " ORDER BY created DESC", (status,) if status else ()).fetchall()
    return [_row(r) for r in rows]

def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        r = c.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return _row(r) if r else None

def save_result(case_id: str, result: Dict[str, Any]) -> None:
    with _conn() as c:
        c.execute("UPDATE cases SET status='processed', result=? WHERE id=?", (json.dumps(result), case_id))

def save_review(case_id: str, review: Dict[str, Any]) -> None:
    with _conn() as c:
        c.execute("UPDATE cases SET status='reviewed', review=? WHERE id=?", (json.dumps(review), case_id))

def delete_case(case_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM cases WHERE id=?", (case_id,))
    folder = os.path.join(CASE_ROOT, case_id)
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)

def clear_all_cases() -> None:
    with _conn() as c:
        c.execute("DELETE FROM cases")
    if os.path.exists(CASE_ROOT):
        shutil.rmtree(CASE_ROOT, ignore_errors=True)
        os.makedirs(CASE_ROOT, exist_ok=True)

def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    for k in ("questionnaire", "result", "review"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d
```

---

### 4.7 End-to-End Orchestrator (`pipeline.py`)

```python
"""End-to-end service + CLI: python pipeline.py [image] [answers.json]"""
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
    for r in regions:
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
```

---

### 4.8 Streamlit Clinical Web Application (`app.py`)

```python
import os
import json
import cv2
import streamlit as st

from core import store
from core.quality import ImageQualityGate
from core.segment import LesionSegmenter
from pipeline import process_case, load_regions

st.set_page_config(page_title="OncoLens", layout="wide", page_icon="🔬")

HABITS = ["Khaini / gutka (smokeless tobacco)", "Supari / areca nut / betel quid",
          "Bidi / cigarette smoking", "Alcohol", "None reported"]
SITES = ["Left buccal mucosa", "Right buccal mucosa", "Tongue", "Floor of mouth",
         "Palate", "Gingiva", "Lip", "Other / multiple"]
RED_FLAGS = {
    "non_healing_ulcer": "Ulcer that has not healed (>3 weeks)",
    "bleeding": "Bleeding from the area on touch",
    "neck_lump": "Lump or swelling in neck",
    "hard_or_fixed": "Area feels hard, indurated, or fixed",
    "numbness": "Numbness in tongue or lip",
    "difficulty_opening_mouth": "Difficulty opening mouth (Trismus)",
}

URG_TAG = {"routine": "🟢 [ROUTINE]", "priority": "🟠 [PRIORITY]", "urgent": "🔴 [URGENT]"}
URG_COLOR = {"routine": "green", "priority": "orange", "urgent": "red"}

st.title("🔬 OncoLens")
st.caption("Oral Mucosal Documentation & Referral Copilot for Frontline Health Workers & District Dentists.")

tab_field, tab_review = st.tabs(["📲 Field (ASHA / ANM)", "👨‍⚕️ Reviewer (District Dentist)"])

# ----------------- FIELD TAB -----------------
with tab_field:
    st.subheader("1. Intraoral Photo Capture")
    src = st.radio("Photo Source", ["Upload", "Camera"], horizontal=True)
    img_file = st.camera_input("Capture with flash on") if src == "Camera" else st.file_uploader("Choose an intraoral photo", type=["jpg", "jpeg", "png"])

    gate_ok, img_bytes = False, None
    if img_file is not None:
        img_bytes = img_file.getvalue()
        os.makedirs("outputs", exist_ok=True)
        tmp = "outputs/_capture.jpg"
        with open(tmp, "wb") as f:
            f.write(img_bytes)

        q = ImageQualityGate().evaluate(tmp)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpness Score", f"{q['blur_score']:.1f}")
        c2.metric("Glare Percentage", f"{q['glare_ratio']*100:.1f}%")
        c3.metric("Quality Gate", "PASS" if q["is_acceptable"] else "RETAKE")

        if q["is_acceptable"]:
            gate_ok = True
            st.success("✅ Image quality is sufficient for clinical review.")
        else:
            for issue in q["issues"]:
                st.warning(f"⚠️ {issue}")
            gate_ok = st.checkbox("Submit anyway (Reviewer will be alerted to low image quality)")

    st.divider()
    st.subheader("2. Patient Screening Questionnaire")
    with st.form("questionnaire_form"):
        a, b = st.columns(2)
        age = a.number_input("Patient Age", min_value=1, max_value=110, value=45)
        sex = b.selectbox("Sex", ["Male", "Female", "Other"])
        site = st.selectbox("Anatomical Location of Finding", SITES)
        habits = st.multiselect("Habits & Tobacco History", HABITS)
        weeks = st.number_input("Duration of lesion / change (weeks)", min_value=0, max_value=520, value=2)

        st.markdown("**Clinical Red Flags (Check all that apply):**")
        red = {k: st.checkbox(label, key=f"rf_{k}") for k, label in RED_FLAGS.items()}
        extra = st.text_input("Other symptoms or medical notes (optional)")
        submitted = st.form_submit_button("💾 Save Case to Offline Queue")

        if submitted:
            if img_bytes is None:
                st.error("Please provide an intraoral photo before saving.")
            elif not gate_ok:
                st.error("Please retake the photo or confirm 'Submit anyway'.")
            else:
                symptoms = [RED_FLAGS[k] for k, v in red.items() if v] + ([extra] if extra else [])
                questionnaire = {
                    "age": int(age), "sex": sex, "anatomical_site": site,
                    "habits": [h for h in habits if h != "None reported"],
                    "lesion_duration_weeks": int(weeks),
                    "symptoms": symptoms or ["None reported"],
                    "red_flags": red,
                }
                cid = store.new_case(img_bytes, questionnaire)
                st.success(f"Case #{cid} saved to offline queue. Ready for sync.")

# ----------------- REVIEWER TAB -----------------
with tab_review:
    pending = store.list_cases("pending")
    done = store.list_cases("processed") + store.list_cases("reviewed")

    col_sync1, col_sync2 = st.columns([2, 1])
    with col_sync1:
        st.subheader(f"Triage Queue ({len(pending)} pending sync, {len(done)} ready)")
    with col_sync2:
        if st.button("🔄 Sync & Process Queue", disabled=not pending, use_container_width=True):
            prog = st.progress(0.0)
            for idx, case in enumerate(pending):
                try:
                    with st.spinner(f"Processing case #{case['id']}..."):
                        res = process_case(case["id"], case["image_path"], case["questionnaire"], verbose=False)
                        store.save_result(case["id"], res)
                except Exception as e:
                    st.error(f"Failed to process case #{case['id']}: {e}")
                prog.progress((idx + 1) / len(pending))
            st.rerun()

    if not done:
        st.info("No processed cases yet. Submit a case in the Field tab and click 'Sync'.")
    else:
        labels = {c["id"]: f"{c['id']} | {URG_TAG[c['result']['urgency_final']]} | {c['questionnaire'].get('anatomical_site', 'Oral')} [{c['status'].upper()}]" for c in done}
        cid = st.selectbox("Select Case for Review", list(labels), format_func=lambda k: labels[k])
        case = store.get_case(cid)
        res = case["result"]
        q_data = case["questionnaire"]
        folder = os.path.join("outputs/cases", cid)
        regions = load_regions(cid, res)
        bgr = cv2.imread(os.path.join(folder, "resized.jpg"))

        if not res.get("image_is_intraoral", True):
            st.error("⚠️ NON-INTRAORAL IMAGE DETECTED: Model flagged this photo as non-oral tissue.")
        if not res["quality"]["is_acceptable"]:
            st.warning("⚠️ Low Quality Field Image: " + " ".join(res["quality"]["issues"]))

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Original Field Capture**")
            st.image(bgr, channels="BGR", width="stretch")

        with col_img2:
            st.markdown("**SAM 3.1 Anatomical Segmentation**")
            chosen = []
            if regions:
                st.caption("Untick regions that represent teeth, glare, or artifacts:")
                for r in regions:
                    conf_str = f"{r['confidence']:.2f}" if r["confidence"] is not None else "0.50"
                    if st.checkbox(f"#{r['id']} {r['concept']} (confidence: {conf_str})", value=True, key=f"{cid}_r{r['id']}"):
                        chosen.append(r["id"])
                st.image(LesionSegmenter.render_overlay(bgr, regions, chosen), channels="BGR", width="stretch")
                selected_pct = LesionSegmenter.area_ratio(regions, chosen) * 100
                st.caption(f"Calculated lesion footprint: **{selected_pct:.1f}%** of field area.")
            else:
                st.image(bgr, channels="BGR", width="stretch")
                st.info("ℹ️ No focal mucosal boundary segmented by SAM 3.1. Dental structures excluded; triage driven by clinical findings and risk questionnaire.")

        st.divider()

        ref = res["referral"]
        urgency = res["urgency_final"].upper()
        urg_col = URG_COLOR[res["urgency_final"]]

        st.markdown(f"## :{urg_col}[TRIAGE LEVEL: {urgency}]")
        if res["urgency_final"] != res.get("urgency_llm"):
            st.info(f"🛡️ Safety Floor Applied: Baseline AI triage was '{res.get('urgency_llm')}'; escalated to '{res['urgency_final']}' by deterministic risk protocol.")

        st.markdown(f"**Actionable Next Step:** {ref['suggested_next_step']}")

        c_memo1, c_memo2 = st.columns(2)
        with c_memo1:
            st.markdown("### 📋 Visible Findings (Objective)")
            for f in ref["visible_findings"]:
                st.markdown(f"- {f}")
        with c_memo2:
            st.markdown("### ⚠️ Clinical Risk Profile")
            st.markdown(f"- **Demographics:** {q_data.get('age', 'Unknown')} y/o {q_data.get('sex', 'Patient')}")
            st.markdown(f"- **Site:** {q_data.get('anatomical_site', 'Oral cavity')}")
            st.markdown(f"- **Duration:** {q_data.get('lesion_duration_weeks', 0)} weeks")
            st.markdown(f"- **Habits:** {', '.join(q_data.get('habits', [])) or 'None reported'}")
            for rf in ref.get("risk_factors", []):
                st.markdown(f"- {rf}")

        st.divider()
        st.subheader("🗣️ Patient Communication Summaries")
        col_en, col_hi = st.columns(2)
        with col_en:
            st.markdown("**English Summary (Community Worker / Patient):**")
            st.info(ref['patient_summary_en'])
        with col_hi:
            st.markdown("**Hindi Summary (Devanagari - मरीज़ के लिए संदेश):**")
            st.success(ref['patient_summary_hi'])

        with st.expander("🛡️ Clinical Safety Rule & Guardrail Audit"):
            st.write(res.get("safety_flags") or ["Standard clinical screening rules."])
            st.caption(ref.get("confidence_note", "Clinician must confirm in person."))

        memo_text = f"""==================================================
ONCOLENS CLINICAL REFERRAL SLIP
Case ID: {cid} | Date: {case.get('created', '')}
Triage Urgency: {urgency}
--------------------------------------------------
PATIENT DETAILS:
Age: {q_data.get('age', 'N/A')} | Sex: {q_data.get('sex', 'N/A')}
Location: {q_data.get('anatomical_site', 'Oral cavity')}
Lesion Duration: {q_data.get('lesion_duration_weeks', 'N/A')} weeks
Habits: {', '.join(q_data.get('habits', [])) or 'None'}

OBJECTIVE VISIBLE FINDINGS:
{chr(10).join(['- ' + x for x in ref['visible_findings']])}

RECOMMENDED ACTION:
{ref['suggested_next_step']}

PATIENT SUMMARY (ENGLISH):
{ref['patient_summary_en']}

PATIENT SUMMARY (HINDI):
{ref['patient_summary_hi']}
=================================================="""

        st.download_button(label="📄 Download Printable Clinical Referral Slip (.txt)",
                           data=memo_text, file_name=f"OncoLens_Referral_{cid}.txt", mime="text/plain")

        st.divider()
        st.subheader("👨‍⚕️ Clinician Review & Final Sign-Off")
        with st.form(f"review_form_{cid}"):
            decision = st.radio("Reviewer Decision", ["Confirm referral", "Override urgency"], horizontal=True)
            final_urg = st.selectbox("Final Assigned Urgency", ["routine", "priority", "urgent"],
                                     index=["routine", "priority", "urgent"].index(res["urgency_final"]))
            clinician_notes = st.text_area("Clinician Decision Notes",
                                           value=case["review"]["notes"] if case.get("review") else "Examination confirmed. Follow-up arranged at District Hospital.")
            if st.form_submit_button("✅ Save & Sign Decision"):
                store.save_review(cid, {"decision": decision, "final_urgency": final_urg,
                                        "notes": clinician_notes, "selected_region_ids": chosen})
                st.success("Review saved successfully. Case status updated to REVIEWED.")
                st.rerun()

        st.divider()
        with st.expander("🗑️ Manage / Reset Case History"):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Delete Current Case**")
                if st.button(f"🗑️ Delete Case #{cid}", key=f"del_{cid}", use_container_width=True):
                    store.delete_case(cid)
                    st.success(f"Case #{cid} deleted.")
                    st.rerun()
            with col_d2:
                st.markdown("**Reset Entire Case History**")
                confirm_wipe = st.checkbox("Confirm: Wipe all cases", key="confirm_wipe")
                if st.button("⚠️ Wipe All History", disabled=not confirm_wipe, type="primary", use_container_width=True):
                    store.clear_all_cases()
                    st.success("Queue completely cleared.")
                    st.rerun()
```

---

## 5. Verification, Execution & Testing Commands

To run and verify the complete OncoLens stack, use the following commands from your project root:

```powershell
# 1. Run Pre-flight Image Quality Gate Calibration
python -m tests.calibrate_quality

# 2. Test SAM 3.1 Segmentation & Tooth Filter
python -m tests.test_segmenter

# 3. Inspect Model Prompt Sensitivity & Controls
python -m tests.debug_sam

# 4. Execute Full End-to-End Orchestration CLI
python pipeline.py data/test_images/sample1.jpeg data/test_images/answers_1.json

# 5. Launch the Local Streamlit Web Application
streamlit run app.py

# 6. Run on Local Network for Mobile Testing (Same Wi-Fi)
streamlit run app.py --server.address 0.0.0.0
```

---

## 6. Roadmap to Final Submission (Nov 4/5 Deadline)

| Day | Date | Target Milestone | Deliverable |
| :--- | :--- | :--- | :--- |
| **Day 3** | Oct 29 | Local Web Application | Working Field + Reviewer tabs with SAM tooth negative filter |
| **Day 4** | Oct 30 | Bilingual UI & Demo Pre-Caching | Polished bilingual cards, downloadable slip, 4-6 cached demo cases |
| **Day 5** | Oct 31 | ABDM / FHIR Standard Compliance | FHIR JSON Bundle (`DiagnosticReport`, `Patient`, `Observation`) + PDF referral slip |
| **Day 6** | Nov 1 | Evaluation & Clinician Review | Good/Okay/Failed accuracy audit table + clinical limitation analysis |
| **Day 7** | Nov 2 | Cloud Deployment | Public deployment on Streamlit Cloud or Hugging Face Spaces |
| **Day 8** | Nov 3 | Video Recording & Documentation | 3-minute demonstration video + comprehensive GitHub README |
| **Day 9** | Nov 4 | Submission | Verification of links, credentials safety check, final submission |
