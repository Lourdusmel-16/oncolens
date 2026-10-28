import os
import json
import cv2
import streamlit as st

from core import store
from core.quality import ImageQualityGate
from core.segment import LesionSegmenter
from pipeline import process_case, load_regions

st.set_page_config(page_title="OncoLens", layout="wide")

HABITS = ["Khaini / gutka (smokeless tobacco)", "Supari / areca nut / betel quid", "Bidi / cigarette smoking",
          "Alcohol", "None reported"]
SITES = ["Left buccal mucosa", "Right buccal mucosa", "Tongue", "Floor of mouth", "Palate",
         "Gingiva", "Lip", "Other / multiple"]
RED_FLAGS = {
    "non_healing_ulcer": "Ulcer that has not healed",
    "bleeding": "Bleeding from the area",
    "neck_lump": "Lump in the neck",
    "hard_or_fixed": "Area feels hard or fixed",
    "numbness": "Numbness",
    "difficulty_opening_mouth": "Difficulty opening mouth",
}
URG_TAG = {"routine": "[GREEN]", "priority": "[AMBER]", "urgent": "[RED]"}
URG_COLOR = {"routine": "green", "priority": "orange", "urgent": "red"}

st.title("OncoLens")
st.caption("Oral lesion documentation & referral copilot. Assistive tool only: it does NOT diagnose. "
           "A licensed clinician must examine every patient.")

tab_field, tab_review = st.tabs(["Field (ASHA / ANM)", "Reviewer (District dentist)"])

# ------------------------------------------------------------------ FIELD
with tab_field:
    st.subheader("1. Capture photo")
    src = st.radio("Photo source", ["Camera", "Upload"], horizontal=True)
    img_file = st.camera_input("Intraoral photo (flash on)") if src == "Camera" else \
        st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

    gate_ok, img_bytes = False, None
    if img_file is not None:
        img_bytes = img_file.getvalue()
        os.makedirs("outputs", exist_ok=True)
        tmp = "outputs/_capture.jpg"
        with open(tmp, "wb") as f:
            f.write(img_bytes)
        q = ImageQualityGate().evaluate(tmp)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpness", q["blur_score"])
        c2.metric("Glare", f"{q['glare_ratio']*100:.1f}%")
        c3.metric("Quality gate", "PASS" if q["is_acceptable"] else "RETAKE")
        if q["is_acceptable"]:
            gate_ok = True
            st.success("Photo is clear enough.")
        else:
            for issue in q["issues"]:
                st.warning(issue)
            gate_ok = st.checkbox("Use this photo anyway (reviewer will be told quality was low)")

    st.subheader("2. Questionnaire")
    with st.form("questionnaire"):
        a, b = st.columns(2)
        age = a.number_input("Age", 1, 110, 45)
        sex = b.selectbox("Sex", ["Male", "Female", "Other"])
        site = st.selectbox("Where in the mouth?", SITES)
        habits = st.multiselect("Habits", HABITS)
        weeks = st.number_input("How many weeks has the change been there?", 0, 520, 0)
        st.markdown("**Warning signs (tick only if present)**")
        red = {k: st.checkbox(label, key=f"rf_{k}") for k, label in RED_FLAGS.items()}
        extra = st.text_input("Other symptoms (optional)")
        submitted = st.form_submit_button("Save case to offline queue")

    if submitted:
        if img_bytes is None:
            st.error("Please take or upload a photo first.")
        elif not gate_ok:
            st.error("Retake the photo (or tick 'use anyway').")
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
            st.success(f"Saved offline as case {cid}. It will be analysed when you sync.")

    st.info(f"Cases waiting to sync: {len(store.list_cases('pending'))}")

# ------------------------------------------------------------------ REVIEWER
with tab_review:
    pending = store.list_cases("pending")
    st.subheader("Sync")
    st.write(f"{len(pending)} case(s) pending.")
    if st.button("Sync & analyse pending cases", disabled=not pending):
        bar = st.progress(0.0)
        for i, case in enumerate(pending):
            try:
                with st.spinner(f"Analysing {case['id']} ..."):
                    res = process_case(case["id"], case["image_path"], case["questionnaire"], verbose=False)
                store.save_result(case["id"], res)
            except Exception as e:  # keep going; one bad case must not block the queue
                st.error(f"{case['id']}: {e}")
            bar.progress((i + 1) / len(pending))
        st.rerun()

    done = store.list_cases("processed") + store.list_cases("reviewed")
    if not done:
        st.info("No analysed cases yet. Save a case in the Field tab, then sync.")
    else:
        labels = {c["id"]: f"{c['id']}  {URG_TAG[c['result']['urgency_final']]} {c['result']['urgency_final']}"
                  f"  [{c['status']}]" for c in done}
        cid = st.selectbox("Case", list(labels), format_func=lambda k: labels[k])
        case = store.get_case(cid)
        res = case["result"]
        folder = os.path.join("outputs/cases", cid)
        regions = load_regions(cid, res)
        bgr = cv2.imread(os.path.join(folder, "resized.jpg"))

        if not res["image_is_intraoral"]:
            st.error("The model says this does not look like an intraoral photo. Ask for a retake.")
        if not res["quality"]["is_acceptable"]:
            st.warning("Field photo was flagged low quality: " + " ".join(res["quality"]["issues"]))

        left, right = st.columns(2)
        left.markdown("**Original photo**")
        left.image(bgr, channels="BGR", width="stretch")

        right.markdown("**SAM 3.1 candidate regions** (untick any that are teeth, gums, or glare)")
        chosen = []
        for r in regions:
            conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "n/a"
            if right.checkbox(f"#{r['id']} {r['concept']} (conf {conf})", value=True, key=f"{cid}_r{r['id']}"):
                chosen.append(r["id"])
        right.image(LesionSegmenter.render_overlay(bgr, regions, chosen), channels="BGR", width="stretch")
        right.caption(f"Selected area: {LesionSegmenter.area_ratio(regions, chosen)*100:.1f}% of image "
                      "(relative, not millimetres).")

        ref = res["referral"]
        st.markdown(f"### :{URG_COLOR[res['urgency_final']]}[Triage: {res['urgency_final'].upper()}]")
        if res["urgency_final"] != res["urgency_llm"]:
            st.caption(f"AI suggested '{res['urgency_llm']}'; raised by safety rules.")
        st.write("**Next step:** " + ref["suggested_next_step"])
        c1, c2 = st.columns(2)
        c1.markdown("**Visible findings**\n" + "\n".join(f"- {x}" for x in ref["visible_findings"]))
        c2.markdown("**Risk factors**\n" + "\n".join(f"- {x}" for x in ref["risk_factors"]))
        st.caption(ref["confidence_note"])
        with st.expander("Patient summary (English / Hindi)"):
            st.write(ref["patient_summary_en"])
            st.write(ref["patient_summary_hi"])
        with st.expander("Safety rule log"):
            st.write(res["safety_flags"] or ["No rules triggered."])

        st.subheader("Reviewer decision")
        with st.form(f"decision_{cid}"):
            decision = st.radio("Decision", ["Confirm referral", "Override urgency"], horizontal=True)
            final_urg = st.selectbox("Final urgency", ["routine", "priority", "urgent"],
                                     index=["routine", "priority", "urgent"].index(res["urgency_final"]))
            notes = st.text_area("Clinician notes")
            if st.form_submit_button("Save decision"):
                store.save_review(cid, {"decision": decision, "final_urgency": final_urg, "notes": notes,
                                        "selected_region_ids": chosen})
                st.success("Decision saved.")
        if case["review"]:
            st.caption(f"Saved decision: {json.dumps(case['review'])}")
