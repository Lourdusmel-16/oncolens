# OncoLens: Execution Guide (Oct 27 – Nov 4, 2026)

Meta Global AI Developer Hackathon · Indie Developer track
Goal: ship a **working, honest, well-demoed** prototype, not a full product.

---

## 0. Read this first

| Item | Detail |
|---|---|
| Deadline | ~**8 PM ET, Nov 4** (= Nov 5, 6:30 AM IST). Your old file said 9 PM, so **verify on the official page** |
| Official page | https://dev.meta.ai/events/global-hackathon |
| Internal freeze | **Nov 4, evening IST.** Don't wait for the last hour |
| Credits | $150 Meta Model API credits, so don't waste calls on dev loops (cache responses) |
| Help | Live workshops + Discord office hours through Nov 4. Use them for API questions |

**Positioning (important):** OncoLens is a **lesion documentation + referral copilot** for frontline workers and district dentists. It does **not** diagnose. A human clinician always decides.

---

## 1. Scope: what's in, what's out

### IN (MVP)
- [ ] Photo capture/upload (Streamlit `st.camera_input` + file upload)
- [ ] Quality gate: blur, glare, basic colour correction (OpenCV)
- [ ] Risk questionnaire (tobacco, betel/gutka, alcohol, lesion duration, pain, bleeding)
- [ ] SAM 3.1 lesion segmentation via Meta Model API
- [ ] Meta Model API multimodal synthesis → structured JSON → referral memo + patient summary (Hindi + English)
- [ ] Reviewer view: raw photo + mask side by side, confirm/override
- [ ] Minimal FHIR Bundle + one-page PDF export
- [ ] Local queue + "Sync" button (fakes the offline-first story)
- [ ] Small evaluation (Dice/IoU if annotated images exist) + honest failure cases

### OUT (say "future work" in the pitch)
- ~~Trained MobileNet edge classifier~~ (no validated data, and it's a safety risk)
- ~~Real offline PWA / ONNX in browser~~
- ~~Tier 3 epidemiology portal~~
- ~~Grad-CAM~~ (only relevant if you train your own classifier)
- ~~Running SAM 1/2 locally~~ (use SAM 3.1 through the API)

---

## 2. Simplified architecture

```text
[Field tab]  photo + questionnaire
     |
     v
Quality gate (OpenCV: blur / glare / white balance)
     |  fail -> "retake" prompt
     v
Local queue (SQLite/JSON)  --Sync button-->
     |
     v
SAM 3.1 (Meta Model API)  -> lesion mask + relative area
     |
     v
Meta Model API (multimodal) -> structured JSON
     |
     v
[Reviewer tab] photo + mask + memo -> confirm / override
     |
     v
Export: PDF referral + FHIR Bundle
```

---

## 3. Day-by-day plan

### Day 1: Oct 27 (today): Access + smoke tests
- [ ] Confirm acceptance + accept official rules
- [ ] Get Meta Model API key from dev.meta.ai, store in `.env`
- [ ] Read the API docs: auth, image input, structured output, SAM 3.1 usage, model names, rate limits
- [ ] One working SAM 3.1 call on a test image
- [ ] One working multimodal call returning JSON
- [ ] Collect **15–20 public oral-lesion images** (normal, ulcer, leukoplakia-like, suspicious lesion). Check each license. No real patient data
- [ ] Create GitHub repo (public, MIT license)

**Done when:** you have a mask and a JSON response for at least one image.

### Day 2: Oct 28: Core pipeline
- [ ] `quality.py`: Laplacian-variance blur score, HSV glare ratio, gray-world white balance, CLAHE
- [ ] `segment.py`: wrapper around SAM 3.1 call → mask → area ratio via OpenCV
- [ ] `reason.py`: multimodal call with Pydantic schema, retries, response caching
- [ ] `pipeline.py`: image + answers → everything above → one result object
- [ ] Run on all test images, save outputs to `outputs/`

**Done when:** `python pipeline.py img.jpg answers.json` prints a full result.

### Day 3: Oct 29: Streamlit app
- [ ] Field tab: capture/upload, quality feedback, questionnaire
- [ ] Reviewer tab: raw + mask overlay side by side, referral memo, confirm/override buttons
- [ ] Use `st.session_state` for case data

**Done when:** a full case flows through the UI locally.

### Day 4: Oct 30: Outputs + language + queue
- [ ] Referral memo (clinical, concise) and patient summary (Hindi + English)
- [ ] Local queue (SQLite) + Sync button that processes pending cases
- [ ] Safe-by-design logic: low confidence or poor image → "refer / retake", never "benign"
- [ ] Disclaimers in the UI and every export

### Day 5: Oct 31: Exports + clinical sanity check
- [ ] FHIR Bundle (Patient, Observation, DiagnosticReport). Validate with `fhir.resources`
- [ ] One-page PDF referral
- [ ] If possible, have a dentist or doctor look at 5–10 outputs and note feedback (even one comment is gold for the pitch)

### Day 6: Nov 1: Evaluation
- [ ] If annotated images exist: Dice/IoU of SAM masks vs ground truth
- [ ] Otherwise: qualitative table (good / okay / failed) across your test set
- [ ] Document failure cases honestly: glare, shadows, dark lesions, poor framing
- [ ] Add a "Limitations" section to the README

### Day 7: Nov 2: Polish + deploy
- [ ] Error handling (API failures, empty mask, bad uploads)
- [ ] Clean UI (consistent layout, readable fonts, mobile-friendly)
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces (put API key in secrets)
- [ ] Test the live link from your phone

### Day 8: Nov 3: Demo video + README
- [ ] Record 2–3 min video (script in section 6)
- [ ] README: problem, architecture diagram, setup, Meta tools used, limitations, safety
- [ ] Repo clean-up: remove keys, add `.env.example`, add requirements.txt

### Day 9: Nov 4: Submit
- [ ] Re-read the submission form and rules
- [ ] Verify repo is public, video link works, live demo works
- [ ] **Submit by evening IST**, then screenshot the confirmation

---

## 4. Suggested repo structure

```text
oncolens/
├── app.py                # Streamlit entry
├── pipeline.py           # end-to-end runner
├── core/
│   ├── quality.py        # blur / glare / colour
│   ├── segment.py        # SAM 3.1 wrapper
│   ├── reason.py         # Meta Model API multimodal + schema
│   ├── fhir_export.py
│   ├── pdf_export.py
│   └── queue.py          # local SQLite queue + sync
├── data/test_images/
├── outputs/
├── README.md
├── requirements.txt
└── .env.example
```

### Referral JSON schema (starting point)

```python
from pydantic import BaseModel
from typing import Literal, List

class Referral(BaseModel):
    urgency: Literal["routine", "priority", "urgent"]
    visible_findings: List[str]          # descriptive only, no diagnosis
    risk_factors: List[str]
    suggested_next_step: str             # e.g. "specialist examination within 2 weeks"
    image_quality_notes: str
    confidence_note: str                 # must state limits
    patient_summary_en: str
    patient_summary_hi: str
```

**Prompt rules:** describe, don't diagnose. If unsure, escalate urgency. Always say a clinician must examine the patient.

---

## 5. Learning checklist (just-in-time, learn on the day you need it)

| Topic | Needed on | Resource |
|---|---|---|
| Meta Model API (auth, images, JSON output) | Day 1 | dev.meta.ai docs + workshops |
| SAM 3.1 prompting | Day 1–2 | Meta Model API docs; SAM concepts: https://github.com/facebookresearch/segment-anything |
| OpenCV quality checks | Day 2 | Laplacian variance, HSV thresholding, CLAHE tutorials |
| Pydantic structured output | Day 2 | Pydantic docs |
| Streamlit (session state, tabs, camera) | Day 3 | Streamlit docs |
| Oral lesion basics (OPMDs, red flags) | Day 1 and 4 | https://pubmed.ncbi.nlm.nih.gov/33128420/ |
| FHIR Bundle / DiagnosticReport | Day 5 | https://www.hl7.org/fhir/bundle.html + `fhir.resources` |
| Segmentation metrics (Dice, IoU) | Day 6 | Any standard tutorial |
| Demo + pitch craft | Day 8 | See next section |

---

## 6. Demo video script (2–3 min)

1. **0:00–0:25: Problem.** Rural India, oral cancer caught late, no specialist nearby. One striking, **sourced** stat (find a citation first)
2. **0:25–1:30: Live flow.** Field worker captures photo → quality gate rejects a blurry one → retake → questionnaire → sync → SAM 3.1 mask → memo appears
3. **1:30–2:15: Reviewer view.** Dentist sees photo + mask + memo, confirms referral, exports PDF + FHIR
4. **2:15–2:45: Meta stack + safety.** Name SAM 3.1 and the Meta Model API explicitly, say "assistive, human-in-the-loop", show one honest limitation
5. **2:45–3:00: Impact + next steps.** Validation with clinicians, offline edge model, ABDM integration

---

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| SAM masks poor on oral images | Use box/point prompts from the UI; show it as a "reviewer-assisted" tool; report failures honestly |
| API quota or errors | Cache every response; keep sample outputs for the demo |
| Behind schedule at Day 5 | Drop FHIR or PDF (keep one), keep the core flow polished |
| Live deploy breaks | Record the video from local first, deploy second |
| No annotated data for Dice | Qualitative evaluation table is acceptable |

---

## 8. Final submission checklist

- [ ] Repo public, README complete, no secrets committed
- [ ] Live demo link tested on mobile
- [ ] Video uploaded, link opens without login
- [ ] Meta tools (SAM 3.1, Model API) named clearly in README + video
- [ ] Limitations + safety section included
- [ ] Submitted before the deadline, confirmation saved
