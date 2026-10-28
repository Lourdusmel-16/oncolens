import json
import os
import cv2
import streamlit as st

from core import store
from core.quality import ImageQualityGate
from core.segment import LesionSegmenter
from pipeline import load_regions, process_case

st.set_page_config(page_title="OncoLens", layout="wide", page_icon="🔬")

HABITS = [
    "Khaini / gutka (smokeless tobacco)",
    "Supari / areca nut / betel quid",
    "Bidi / cigarette smoking",
    "Alcohol",
    "None reported",
]

SITES = [
    "Left buccal mucosa",
    "Right buccal mucosa",
    "Tongue",
    "Floor of mouth",
    "Palate",
    "Gingiva",
    "Lip",
    "Other / multiple",
]

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

# Custom CSS for clinical card styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 12px;
    }
    .patient-card-en {
        background-color: #1a2332;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #2a3a50;
    }
    .patient-card-hi {
        background-color: #262135;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #40325a;
        font-size: 1.05rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔬 OncoLens")
st.caption(
    "Oral Mucosal Documentation & Referral Copilot for Frontline Health Workers"
    " & District Dentists. Assistive screening tool: does not diagnose."
)

tab_field, tab_review = st.tabs(["📲 Field (ASHA / ANM)", "👨‍⚕️ Reviewer (District Dentist)"])

# ------------------------------------------------------------------ TAB 1: FIELD
with tab_field:
  st.subheader("1. Intraoral Photo Capture")
  src = st.radio("Photo Source", ["Upload", "Camera"], horizontal=True)
  img_file = (
      st.camera_input("Capture with flash on")
      if src == "Camera"
      else st.file_uploader(
          "Choose an intraoral photo", type=["jpg", "jpeg", "png"]
      )
  )

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
    gate_status = "PASS" if q["is_acceptable"] else "RETAKE"
    c3.metric(
        "Quality Gate",
        gate_status,
        delta="Clear" if q["is_acceptable"] else "Blur/Glare Warning",
    )

    if q["is_acceptable"]:
      gate_ok = True
      st.success("✅ Image quality is sufficient for clinical review.")
    else:
      for issue in q["issues"]:
        st.warning(f"⚠️ {issue}")
      gate_ok = st.checkbox(
          "Submit anyway (Reviewer will be alerted to low image quality)"
      )

  st.divider()
  st.subheader("2. Patient Screening Questionnaire")
  with st.form("questionnaire_form"):
    a, b = st.columns(2)
    age = a.number_input("Patient Age", min_value=1, max_value=110, value=45)
    sex = b.selectbox("Sex", ["Male", "Female", "Other"])
    site = st.selectbox("Anatomical Location of Finding", SITES)
    habits = st.multiselect("Habits & Tobacco History", HABITS)
    weeks = st.number_input(
        "Duration of lesion / change (weeks)", min_value=0, max_value=520, value=2
    )

    st.markdown("**Clinical Red Flags (Check all that apply):**")
    red = {
        k: st.checkbox(label, key=f"rf_{k}") for k, label in RED_FLAGS.items()
    }
    extra = st.text_input("Other symptoms or medical notes (optional)")
    submitted = st.form_submit_button("💾 Save Case to Offline Queue")

    if submitted:
      if img_bytes is None:
        st.error("Please provide an intraoral photo before saving.")
      elif not gate_ok:
        st.error(
            "Please retake the photo or confirm 'Submit anyway' if retake is not"
            " possible."
        )
      else:
        symptoms = [RED_FLAGS[k] for k, v in red.items() if v] + (
            [extra] if extra else []
        )
        questionnaire = {
            "age": int(age),
            "sex": sex,
            "anatomical_site": site,
            "habits": [h for h in habits if h != "None reported"],
            "lesion_duration_weeks": int(weeks),
            "symptoms": symptoms or ["None reported"],
            "red_flags": red,
        }
        cid = store.new_case(img_bytes, questionnaire)
        st.success(
            f"Case #{cid} saved to offline queue. Ready for sync when"
            " connectivity is available."
        )

# ------------------------------------------------------------------ TAB 2: REVIEWER
with tab_review:
  pending = store.list_cases("pending")
  done = store.list_cases("processed") + store.list_cases("reviewed")

  col_sync1, col_sync2 = st.columns([2, 1])
  with col_sync1:
    st.subheader(f"Triage Queue ({len(pending)} pending sync, {len(done)} ready)")
  with col_sync2:
    if st.button(
        "🔄 Sync & Process Queue",
        disabled=not pending,
        use_container_width=True,
    ):
      prog = st.progress(0.0)
      for idx, case in enumerate(pending):
        try:
          with st.spinner(f"Processing case #{case['id']}..."):
            res = process_case(
                case["id"],
                case["image_path"],
                case["questionnaire"],
                verbose=False,
            )
            store.save_result(case["id"], res)
        except Exception as e:
          st.error(f"Failed to process case #{case['id']}: {e}")
        prog.progress((idx + 1) / len(pending))
      st.rerun()

  if not done:
    st.info(
        "No processed cases yet. Submit a case in the Field tab and click 'Sync'."
    )
  else:
    labels = {
        c["id"]: (
            f"{c['id']} | {URG_TAG[c['result']['urgency_final']]} |"
            f" {c['questionnaire'].get('anatomical_site', 'Oral')}"
            f" [{c['status'].upper()}]"
        )
        for c in done
    }
    cid = st.selectbox(
        "Select Case for Review", list(labels), format_func=lambda k: labels[k]
    )
    case = store.get_case(cid)
    res = case["result"]
    q_data = case["questionnaire"]
    folder = os.path.join("outputs/cases", cid)
    regions = load_regions(cid, res)
    bgr = cv2.imread(os.path.join(folder, "resized.jpg"))

    # Non-intraoral warning
    if not res.get("image_is_intraoral", True):
      st.error(
          "⚠️ NON-INTRAORAL IMAGE DETECTED: The vision model flagged this photo"
          " as non-oral tissue. Request an immediate in-person retake."
      )

    # Low quality warning
    if not res["quality"]["is_acceptable"]:
      st.warning(
          "⚠️ Low Quality Field Image: " + " ".join(res["quality"]["issues"])
      )

    # Visual Inspection & SAM Controls
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
          conf_str = (
              f"{r['confidence']:.2f}"
              if r["confidence"] is not None
              else "0.50"
          )
          if st.checkbox(
              f"#{r['id']} {r['concept']} (confidence: {conf_str})",
              value=True,
              key=f"{cid}_r{r['id']}",
          ):
            chosen.append(r["id"])
        st.image(
            LesionSegmenter.render_overlay(bgr, regions, chosen),
            channels="BGR",
            use_container_width=True,
        )
        selected_pct = (
            LesionSegmenter.area_ratio(regions, chosen) * 100
        )
        st.caption(
            f"Calculated lesion footprint: **{selected_pct:.1f}%** of field"
            " area (pixel-relative)."
        )
      else:
        st.image(bgr, channels="BGR", width="stretch")
        st.info(
            "ℹ️ No focal mucosal boundary segmented by SAM 3.1. Dental"
            " structures excluded; screening triage driven by clinical findings"
            " and risk questionnaire."
        )

    st.divider()

    # Clinical Referral Memo Section
    ref = res["referral"]
    urgency = res["urgency_final"].upper()
    urg_col = URG_COLOR[res["urgency_final"]]

    st.markdown(f"## :{urg_col}[TRIAGE LEVEL: {urgency}]")
    if res["urgency_final"] != res.get("urgency_llm"):
      st.info(
          f"🛡️ Safety Floor Applied: Baseline AI triage was"
          f" '{res.get('urgency_llm')}'; escalated to '{res['urgency_final']}'"
          " by deterministic risk protocol."
      )

    st.markdown(f"**Actionable Next Step:** {ref['suggested_next_step']}")

    c_memo1, c_memo2 = st.columns(2)
    with c_memo1:
      st.markdown("### 📋 Visible Findings (Objective)")
      for f in ref["visible_findings"]:
        st.markdown(f"- {f}")
    with c_memo2:
      st.markdown("### ⚠️ Clinical Risk Profile")
      st.markdown(
          f"- **Demographics:** {q_data.get('age', 'Unknown')} y/o"
          f" {q_data.get('sex', 'Patient')}"
      )
      st.markdown(f"- **Site:** {q_data.get('anatomical_site', 'Oral cavity')}")
      st.markdown(
          f"- **Duration:** {q_data.get('lesion_duration_weeks', 0)} weeks"
      )
      st.markdown(
          f"- **Habits:** {', '.join(q_data.get('habits', [])) or 'None reported'}"
      )
      for rf in ref.get("risk_factors", []):
        st.markdown(f"- {rf}")

    st.divider()

    # Bilingual Patient Communication
    st.subheader("🗣️ Patient Communication Summaries")
    col_en, col_hi = st.columns(2)
    with col_en:
      st.markdown("**English Summary (For Community Worker / Patient):**")
      st.markdown(
          f"<div class='patient-card-en'>{ref['patient_summary_en']}</div>",
          unsafe_allow_html=True,
      )
    with col_hi:
      st.markdown("**Hindi Summary (Devanagari - मरीज़ के लिए संदेश):**")
      st.markdown(
          f"<div class='patient-card-hi'>{ref['patient_summary_hi']}</div>",
          unsafe_allow_html=True,
      )

    with st.expander("🛡️ Clinical Safety Rule & Guardrail Audit"):
      st.write(res.get("safety_flags") or ["Standard clinical screening rules."])
      st.caption(
          ref.get("confidence_note", "Clinician must confirm in person.")
      )

    # Printable Memo Download
    memo_text = f"""==================================================
ONCOLENS CLINICAL REFERRAL SLIP
Case ID: {cid}
Date: {case.get('created', '')}
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

    st.download_button(
        label="📄 Download Printable Clinical Referral Slip (.txt)",
        data=memo_text,
        file_name=f"OncoLens_Referral_{cid}.txt",
        mime="text/plain",
    )

    st.divider()

    # Clinician Review Decision Form
    st.subheader("👨‍⚕️ Clinician Review & Final Sign-Off")
    with st.form(f"review_form_{cid}"):
      decision = st.radio(
          "Reviewer Decision",
          ["Confirm referral", "Override urgency"],
          horizontal=True,
      )
      final_urg = st.selectbox(
          "Final Assigned Urgency",
          ["routine", "priority", "urgent"],
          index=["routine", "priority", "urgent"].index(res["urgency_final"]),
      )
      clinician_notes = st.text_area(
          "Clinician Decision Notes",
          value=(
              case["review"]["notes"]
              if case.get("review")
              else "Examination confirmed. Follow-up arranged at District"
              " Hospital."
          ),
      )
      if st.form_submit_button("✅ Save & Sign Decision"):
        store.save_review(
            cid,
            {
                "decision": decision,
                "final_urgency": final_urg,
                "notes": clinician_notes,
                "selected_region_ids": chosen,
            },
        )
        st.success(
            "Review saved successfully. Case status updated to REVIEWED."
        )
        st.rerun()