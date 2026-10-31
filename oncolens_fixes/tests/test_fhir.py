"""Run from project root:  python -m tests.test_fhir     (pip install fhir.resources first)
Builds a FHIR bundle for the newest processed case in your queue and validates it."""
import json
from core import store
from core.fhir_export import build_bundle
from fhir.resources.R4B.bundle import Bundle

cases = store.list_cases("processed") + store.list_cases("reviewed")
assert cases, "Process a case in the app first (Sync), then run this test."
case = store.get_case(cases[0]["id"])
bundle = build_bundle(case)
Bundle.model_validate(bundle)
print("Valid FHIR R4B Bundle with", len(bundle["entry"]), "entries for case", case["id"])
with open("outputs/sample_fhir_bundle.json", "w", encoding="utf-8") as f:
    json.dump(bundle, f, indent=2)
print("Saved outputs/sample_fhir_bundle.json")
