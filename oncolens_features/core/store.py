"""SQLite queue for the offline-first field app, plus referral follow-up tracking and an audit log.
Tables: cases (the queue), followups (referral loop), audit (append-only activity log, no PHI)."""
import os
import json
import sqlite3
import uuid
import shutil
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("ONCOLENS_DB", "outputs/oncolens.db")
CASE_ROOT = "outputs/cases"

FOLLOWUP_STATUSES = ["referred", "attended", "no_show", "closed"]
# Days allowed to reach the district dentist. Prototype policy: a clinician should confirm these.
DUE_DAYS = {"urgent": 7, "priority": 14, "routine": 30}


@contextmanager
def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS cases(
        id TEXT PRIMARY KEY, created TEXT, status TEXT, image_path TEXT,
        questionnaire TEXT, result TEXT, review TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS followups(
        case_id TEXT PRIMARY KEY, status TEXT, due_date TEXT, outcome_note TEXT, updated TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS audit(
        id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, ts TEXT, action TEXT, detail TEXT)""")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(c: sqlite3.Connection, case_id: str, action: str, detail: str = "") -> None:
    c.execute("INSERT INTO audit(case_id, ts, action, detail) VALUES (?,?,?,?)", (case_id, _now(), action, detail))


# ---------------------------------------------------------------- cases
def new_case(image_bytes: bytes, questionnaire: Dict[str, Any]) -> str:
    case_id = datetime.now().strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    folder = os.path.join(CASE_ROOT, case_id)
    os.makedirs(folder, exist_ok=True)
    image_path = os.path.join(folder, "photo.jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    with db() as c:
        c.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?)",
                  (case_id, _now(), "pending", image_path, json.dumps(questionnaire), None, None))
        _log(c, case_id, "case_created", "consent recorded" if questionnaire.get("consent") else "consent not recorded")
    return case_id


def list_cases(status: Optional[str] = None) -> List[Dict[str, Any]]:
    with db() as c:
        rows = c.execute("SELECT * FROM cases" + (" WHERE status=?" if status else "") +
                         " ORDER BY created DESC", (status,) if status else ()).fetchall()
    return [_row(r) for r in rows]


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    with db() as c:
        r = c.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return _row(r) if r else None


def save_result(case_id: str, result: Dict[str, Any]) -> None:
    with db() as c:
        c.execute("UPDATE cases SET status='processed', result=? WHERE id=?", (json.dumps(result), case_id))
        _log(c, case_id, "analysed", f"system triage: {result.get('urgency_final')}")


def save_review(case_id: str, review: Dict[str, Any]) -> None:
    """Clinician sign-off. Also opens (or refreshes) the referral follow-up record."""
    with db() as c:
        c.execute("UPDATE cases SET status='reviewed', review=? WHERE id=?", (json.dumps(review), case_id))
        _log(c, case_id, "clinician_decision", f"{review.get('decision')} -> {review.get('final_urgency')}")
        due = (date.today() + timedelta(days=DUE_DAYS[review["final_urgency"]])).isoformat()
        c.execute("INSERT OR IGNORE INTO followups(case_id, status, due_date, outcome_note, updated) VALUES (?,?,?,?,?)",
                  (case_id, "referred", due, "", _now()))
        c.execute("UPDATE followups SET due_date=? WHERE case_id=? AND status='referred'", (due, case_id))


def delete_case(case_id: str) -> None:
    with db() as c:
        c.execute("DELETE FROM cases WHERE id=?", (case_id,))
        c.execute("DELETE FROM followups WHERE case_id=?", (case_id,))
        _log(c, case_id, "case_deleted")
    folder = os.path.join(CASE_ROOT, case_id)
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)


def clear_all_cases() -> None:
    with db() as c:
        c.execute("DELETE FROM cases")
        c.execute("DELETE FROM followups")
        _log(c, "*", "all_cases_wiped")
    if os.path.exists(CASE_ROOT):
        shutil.rmtree(CASE_ROOT, ignore_errors=True)
        os.makedirs(CASE_ROOT, exist_ok=True)


# ---------------------------------------------------------------- follow-up tracker
def get_followup(case_id: str) -> Optional[Dict[str, Any]]:
    with db() as c:
        r = c.execute("SELECT * FROM followups WHERE case_id=?", (case_id,)).fetchone()
    return dict(r) if r else None


def list_followups() -> List[Dict[str, Any]]:
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM followups ORDER BY due_date").fetchall()]


def set_followup(case_id: str, status: str, outcome_note: str = "") -> None:
    if status not in FOLLOWUP_STATUSES:
        raise ValueError(f"status must be one of {FOLLOWUP_STATUSES}")
    with db() as c:
        c.execute("UPDATE followups SET status=?, outcome_note=?, updated=? WHERE case_id=?",
                  (status, outcome_note, _now(), case_id))
        _log(c, case_id, "followup_updated", status)


def is_overdue(fu: Optional[Dict[str, Any]], today: Optional[str] = None) -> bool:
    """Overdue = still 'referred' after the due date."""
    return bool(fu) and fu["status"] == "referred" and fu["due_date"] < (today or date.today().isoformat())


# ---------------------------------------------------------------- audit log
def get_audit(case_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    with db() as c:
        if case_id:
            rows = c.execute("SELECT * FROM audit WHERE case_id=? ORDER BY id DESC LIMIT ?", (case_id, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    for k in ("questionnaire", "result", "review"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d
