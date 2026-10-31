"""Printable referral slip as HTML (browser 'Print > Save as PDF' renders Hindi correctly)."""
from html import escape
from typing import Any, Dict

COLORS = {"routine": "#15803d", "priority": "#c2410c", "urgent": "#b91c1c"}


def slip_html(case: Dict[str, Any]) -> str:
    q, res, ref = case["questionnaire"], case["result"], case["result"]["referral"]
    wk = q.get("lesion_duration_weeks")
    urg = res["urgency_final"]
    li = lambda items: "".join(f"<li>{escape(x)}</li>" for x in items) or "<li>None</li>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>OncoLens referral {escape(case['id'])}</title>
<style>body{{font-family:'Segoe UI','Noto Sans Devanagari',Arial,sans-serif;max-width:760px;margin:24px auto;color:#111}}
h1{{font-size:20px;margin:0}} .tag{{display:inline-block;padding:4px 12px;border-radius:6px;color:#fff;background:{COLORS[urg]};font-weight:700}}
table{{border-collapse:collapse;width:100%}} td{{border:1px solid #ccc;padding:6px 8px;vertical-align:top}} .note{{font-size:12px;color:#555}}
@media print{{button{{display:none}}}}</style></head><body>
<h1>OncoLens clinical referral slip</h1><p class="note">Case {escape(case['id'])} | {escape(str(case.get('created', '')))}</p>
<p>Triage: <span class="tag">{urg.upper()}</span></p>
<table><tr><td>Age / sex</td><td>{escape(str(q.get('age', 'N/A')))} / {escape(str(q.get('sex', 'N/A')))}</td></tr>
<tr><td>Site</td><td>{escape(q.get('anatomical_site', 'Oral cavity'))}</td></tr>
<tr><td>Duration</td><td>{'Not sure' if wk is None else f'{wk} weeks'}</td></tr>
<tr><td>Habits</td><td>{escape(', '.join(q.get('habits') or []) or 'None reported')}</td></tr></table>
<h3>Visible findings</h3><ul>{li(ref['visible_findings'])}</ul>
<h3>Recommended action</h3><p>{escape(ref['suggested_next_step'])}</p>
<h3>Patient summary</h3><p>{escape(ref['patient_summary_en'])}</p><p>{escape(ref['patient_summary_hi'])}</p>
<p class="note">Assistive documentation only; not a diagnosis. A licensed clinician must examine the patient.</p>
<button onclick="window.print()">Print / Save as PDF</button></body></html>"""
