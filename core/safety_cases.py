"""The safety scenario matrix, shared by tests/test_safety.py and the app's Safety and Evidence tab."""
from typing import Any, Dict, List
from core.safety import apply_safety

KEYS = ("non_healing_ulcer", "bleeding", "neck_lump", "hard_or_fixed", "numbness", "difficulty_opening_mouth")
NONE = {k: False for k in KEYS}


def flags(**on) -> Dict[str, bool]:
    return {**NONE, **{k: True for k in on}}


def referral(urgency: str = "routine") -> Dict[str, Any]:
    return {"urgency": urgency, "visible_findings": ["Pale patch on left cheek"], "risk_factors": [],
            "suggested_next_step": "Dental review", "image_quality_notes": "ok", "confidence_note": "x",
            "patient_summary_en": "A change was seen.", "patient_summary_hi": "x"}


# (scenario, AI urgency, habits, weeks, red flags, regions found, expected final urgency)
CASES = [
    ("No habit, short duration, AI routine",        "routine",  [],         1,    NONE,                               1, "routine"),
    ("Tobacco + 6 weeks, AI routine",               "routine",  ["Khaini"], 6,    NONE,                               1, "priority"),
    ("Tobacco + 1 week, AI routine",                "routine",  ["Khaini"], 1,    NONE,                               1, "routine"),
    ("Tobacco + duration unknown, AI routine",      "routine",  ["Khaini"], None, NONE,                               1, "priority"),
    ("Ulcer + 4 weeks (no habit), AI routine",      "routine",  [],         4,    flags(non_healing_ulcer=1),         1, "urgent"),
    ("Bleeding + hard area, AI routine",            "routine",  [],         1,    flags(bleeding=1, hard_or_fixed=1), 1, "urgent"),
    ("One red flag, 1 week, no habit, AI routine",  "routine",  [],         1,    flags(neck_lump=1),                 1, "priority"),
    ("No SAM region + tobacco, AI routine",         "routine",  ["Bidi"],   1,    NONE,                               0, "priority"),
    ("AI says urgent, no risk factors",             "urgent",   [],         1,    NONE,                               1, "urgent"),
    ("AI says priority, tobacco + 6 weeks",         "priority", ["Khaini"], 6,    NONE,                               1, "priority"),
]


def wording_guard_ok() -> bool:
    r = referral()
    r["visible_findings"].append("Looks like carcinoma")
    r["patient_summary_en"] = "This is not cancer."
    out = apply_safety(r, {"habits": [], "lesion_duration_weeks": 1, "red_flags": NONE}, 1)
    return (out["referral"]["visible_findings"] == ["Pale patch on left cheek"]
            and "cancer" not in out["referral"]["patient_summary_en"])


def run_matrix() -> List[Dict[str, Any]]:
    rows = []
    for name, ai, habits, weeks, red, n, expect in CASES:
        q = {"habits": habits, "lesion_duration_weeks": weeks, "red_flags": red}
        got = apply_safety(referral(ai), q, n)["urgency_final"]
        rows.append({"scenario": name, "ai_said": ai, "expected": expect, "got": got, "passed": got == expect})
    ok = wording_guard_ok()
    rows.append({"scenario": "Diagnostic wording removed or replaced", "ai_said": "n/a",
                 "expected": "removed", "got": "removed" if ok else "kept", "passed": ok})
    return rows
