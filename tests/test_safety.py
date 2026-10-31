"""Run from project root:  python -m tests.test_safety
Scenario matrix: proves the safety floors work no matter what the AI says."""
from core.safety import apply_safety

def referral(urgency="routine"):
    return {"urgency": urgency, "visible_findings": ["Pale patch on left cheek"], "risk_factors": [],
            "suggested_next_step": "Dental review", "image_quality_notes": "ok", "confidence_note": "x",
            "patient_summary_en": "A change was seen.", "patient_summary_hi": "x"}

NONE = {k: False for k in ("non_healing_ulcer", "bleeding", "neck_lump", "hard_or_fixed", "numbness", "difficulty_opening_mouth")}
def flags(**on): return {**NONE, **{k: True for k in on}}

# (name, AI urgency, habits, weeks, red_flags, regions found, expected final)
CASES = [
    ("No habit, short duration, AI routine",        "routine",  [],                1,    NONE,                     1, "routine"),
    ("Tobacco + 6 weeks, AI routine",               "routine",  ["Khaini"],        6,    NONE,                     1, "priority"),
    ("Tobacco + 1 week, AI routine",                "routine",  ["Khaini"],        1,    NONE,                     1, "routine"),
    ("Tobacco + duration unknown, AI routine",      "routine",  ["Khaini"],        None, NONE,                     1, "priority"),
    ("Ulcer + 4 weeks (no habit), AI routine",      "routine",  [],                4,    flags(non_healing_ulcer=1), 1, "urgent"),
    ("Bleeding + hard area, AI routine",            "routine",  [],                1,    flags(bleeding=1, hard_or_fixed=1), 1, "urgent"),
    ("One red flag, 1 week, no habit, AI routine",  "routine",  [],                1,    flags(neck_lump=1),       1, "priority"),
    ("No SAM region + tobacco, AI routine",         "routine",  ["Bidi"],          1,    NONE,                     0, "priority"),
    ("AI says urgent, no risk factors",             "urgent",   [],                1,    NONE,                     1, "urgent"),   # floor never lowers
    ("AI says priority, tobacco + 6 weeks",         "priority", ["Khaini"],        6,    NONE,                     1, "priority"),
]
bad = 0
for name, ai, habits, weeks, red, n, expect in CASES:
    q = {"habits": habits, "lesion_duration_weeks": weeks, "red_flags": red}
    got = apply_safety(referral(ai), q, n)["urgency_final"]
    ok = got == expect; bad += (not ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:46s} expected={expect:8s} got={got}")

r = referral(); r["visible_findings"].append("Looks like carcinoma"); r["patient_summary_en"] = "This is not cancer."
out = apply_safety(r, {"habits": [], "lesion_duration_weeks": 1, "red_flags": NONE}, 1)
ok = out["referral"]["visible_findings"] == ["Pale patch on left cheek"] and "cancer" not in out["referral"]["patient_summary_en"]
print(f"{'PASS' if ok else 'FAIL'}  Diagnostic wording is removed / replaced"); bad += (not ok)
print("\nALL PASSED" if not bad else f"\n{bad} FAILED")
