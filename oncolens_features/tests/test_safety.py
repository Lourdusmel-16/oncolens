"""Run from project root:  python -m tests.test_safety
Scenario matrix: proves the safety floors work no matter what the AI says."""
from core.safety_cases import run_matrix

rows = run_matrix()
for r in rows:
    print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['scenario']:46s} expected={r['expected']:8s} got={r['got']}")
bad = sum(not r["passed"] for r in rows)
print("\nALL PASSED" if not bad else f"\n{bad} FAILED")
