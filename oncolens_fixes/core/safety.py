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
    "\u0915\u0947\u0935\u0932 \u0921\u0949\u0915\u094d\u091f\u0930 \u0939\u0940 \u091c\u093e\u0901\u091a "
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
    duration_unknown = weeks is None   # "Not sure" in the form
    habit = has_risk_habit(questionnaire.get("habits", []))
    red = questionnaire.get("red_flags", {}) or {}
    n_red = sum(1 for k in RED_FLAG_KEYS if red.get(k))

    if habit and duration_unknown:
        floor = "priority"
        flags.append("Floor 'priority': duration unknown, treated as persistent (safe default) with risk habit.")
    elif habit and weeks >= PERSISTENT_WEEKS:
        floor = "priority"
        flags.append(f"Floor 'priority': lesion >= {PERSISTENT_WEEKS} weeks with risk habit.")

    if n_red >= 1 and floor == "routine":
        floor = "priority"
        flags.append("Floor 'priority': at least one clinical red flag reported.")

    if not duration_unknown and weeks >= URGENT_WEEKS and n_red >= 1:
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
