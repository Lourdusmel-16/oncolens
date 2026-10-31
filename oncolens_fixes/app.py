import os
import io
import glob
import json
import shutil
import cv2
import streamlit as st

from core import store
from core.quality import ImageQualityGate
from core.segment import LesionSegmenter
from pipeline import process_case, load_regions
from core.fhir_export import build_bundle
from core.slip import slip_html

st.set_page_config(page_title="OncoLens", layout="wide", page_icon="🔬")

# ---- deployment helpers (Streamlit Cloud / Hugging Face) ----
if "MODEL_API_KEY" not in os.environ:
    try:
        os.environ["MODEL_API_KEY"] = st.secrets["MODEL_API_KEY"]
    except Exception:
        pass
DEMO_MODE = os.getenv("DEMO_MODE") == "1"   # sample photos only, no uploads, no wipe button
os.makedirs("outputs/cache", exist_ok=True)
for _f in glob.glob("demo_cache/*"):        # pre-recorded API answers for the demo photos
    _dst = os.path.join("outputs/cache", os.path.basename(_f))
    if not os.path.exists(_dst):
        shutil.copy(_f, _dst)

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
st.warning("Prototype for demonstration. It assists documentation and triage only and does not diagnose. "
           "Do not upload real patient photos.")

tab_field, tab_review = st.tabs(["📲 Field (ASHA / ANM)", "👨‍⚕️ Reviewer (District Dentist)"])

# ----------------- FIELD TAB -----------------
with tab_field:
    st.subheader("1. Intraoral Photo Capture")
    if DEMO_MODE:
        demo_files = sorted(glob.glob("data/demo/*.jp*g") + glob.glob("data/demo/*.png"))
        pick = st.selectbox("Choose a sample photo (demo mode: uploads are disabled)", demo_files) if demo_files else None
        img_file = io.BytesIO(open(pick, "rb").read()) if pick else None   # BytesIO also has .getvalue()
    else:
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
        DURATIONS = {"Less than 2 weeks": 1, "2 to 3 weeks": 2, "More than 3 weeks": 4,
                     "More than 3 months": 13, "Not sure": None}
        dur_label = st.selectbox("How long has the change been there?", list(DURATIONS), index=4)
        weeks = DURATIONS[dur_label]

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
                    "lesion_duration_weeks": weeks,
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
        if st.button("🔄 Sync & Process Queue", disabled=not pending, width="stretch"):
            prog = st.progress(0.0)
            errors = []
            for idx, case in enumerate(pending):
                try:
                    with st.spinner(f"Processing case #{case['id']}..."):
                        res = process_case(case["id"], case["image_path"], case["questionnaire"], verbose=False)
                        store.save_result(case["id"], res)
                except Exception as e:
                    errors.append(f"Case #{case['id']} failed: {e}")
                prog.progress((idx + 1) / len(pending))
            st.session_state["sync_errors"] = errors  # survive the rerun below
            st.rerun()

    for msg in st.session_state.pop("sync_errors", []):
        st.error(msg)

    if not done:
        st.info("No processed cases yet. Submit a case in the Field tab and click 'Sync'.")
    else:
        labels = {c["id"]: f"{c['id']} | {URG_TAG[c['result']['urgency_final']]} | {c['questionnaire'].get('anatomical_site', 'Oral')} [{c['status'].upper()}]" for c in done}
        cid = st.selectbox("Select Case for Review", list(labels), format_func=lambda k: labels[k])
        case = store.get_case(cid)
        res = case["result"]
        q_data = case["questionnaire"]
        wk = q_data.get("lesion_duration_weeks")
        dur_txt = "Not sure" if wk is None else f"{wk} weeks"
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
            st.markdown(f"- **Duration:** {dur_txt}")
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
Lesion Duration: {dur_txt}
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

        with open(os.path.join(folder, "overlay.jpg"), "rb") as f_ov:
            overlay_bytes = f_ov.read()
        dl1, dl2 = st.columns(2)
        dl1.download_button("Download FHIR Bundle (.json)", data=json.dumps(build_bundle(case, overlay_bytes), indent=2),
                            file_name=f"OncoLens_FHIR_{cid}.json", mime="application/json")
        dl2.download_button("Download printable slip (.html, Hindi OK)", data=slip_html(case),
                            file_name=f"OncoLens_Slip_{cid}.html", mime="text/html")

        st.divider()
        st.subheader("👨‍⚕️ Clinician Review & Final Sign-Off")
        with st.form(f"review_form_{cid}"):
            decision = st.radio("Reviewer Decision", ["Confirm referral", "Override urgency"], horizontal=True)
            final_urg = st.selectbox("Final Assigned Urgency", ["routine", "priority", "urgent"],
                                     index=["routine", "priority", "urgent"].index(res["urgency_final"]))
            clinician_notes = st.text_area("Clinician Decision Notes",
                                           value=case["review"]["notes"] if case.get("review") else "")
            if st.form_submit_button("✅ Save & Sign Decision"):
                order = ["routine", "priority", "urgent"]
                if order.index(final_urg) < order.index(res["urgency_final"]) and not clinician_notes.strip():
                    st.error("Lowering urgency below the system triage requires a clinician note.")
                else:
                    store.save_review(cid, {"decision": decision, "final_urgency": final_urg,
                                            "notes": clinician_notes, "selected_region_ids": chosen})
                    st.success("Review saved. Case status updated to REVIEWED.")
                    st.rerun()

        st.divider()
        with st.expander("🗑️ Manage / Reset Case History"):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Delete Current Case**")
                if st.button(f"🗑️ Delete Case #{cid}", key=f"del_{cid}", width="stretch"):
                    store.delete_case(cid)
                    st.success(f"Case #{cid} deleted.")
                    st.rerun()
            with col_d2:
                st.markdown("**Reset Entire Case History**")
                confirm_wipe = st.checkbox("Confirm: Wipe all cases", key="confirm_wipe")
                if st.button("⚠️ Wipe All History", disabled=(not confirm_wipe) or DEMO_MODE, type="primary", width="stretch"):
                    store.clear_all_cases()
                    st.success("Queue completely cleared.")
                    st.rerun()
