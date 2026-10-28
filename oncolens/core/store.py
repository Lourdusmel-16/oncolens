"""Tiny SQLite queue = the 'offline-first' story. Field app writes cases locally (status=pending);
'Sync' later processes them (status=processed); reviewer decision -> status=reviewed."""
import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("ONCOLENS_DB", "outputs/oncolens.db")
CASE_ROOT = "outputs/cases"


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS cases(
        id TEXT PRIMARY KEY, created TEXT, status TEXT, image_path TEXT,
        questionnaire TEXT, result TEXT, review TEXT)""")
    return conn


def new_case(image_bytes: bytes, questionnaire: Dict[str, Any]) -> str:
    case_id = datetime.now().strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    folder = os.path.join(CASE_ROOT, case_id)
    os.makedirs(folder, exist_ok=True)
    image_path = os.path.join(folder, "photo.jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    with _conn() as c:
        c.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?)",
                  (case_id, datetime.now().isoformat(timespec="seconds"), "pending",
                   image_path, json.dumps(questionnaire), None, None))
    return case_id


def list_cases(status: Optional[str] = None) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM cases" + (" WHERE status=?" if status else "") +
                         " ORDER BY created DESC", (status,) if status else ()).fetchall()
    return [_row(r) for r in rows]


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        r = c.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return _row(r) if r else None


def save_result(case_id: str, result: Dict[str, Any]) -> None:
    with _conn() as c:
        c.execute("UPDATE cases SET status='processed', result=? WHERE id=?", (json.dumps(result), case_id))


def save_review(case_id: str, review: Dict[str, Any]) -> None:
    with _conn() as c:
        c.execute("UPDATE cases SET status='reviewed', review=? WHERE id=?", (json.dumps(review), case_id))


def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    for k in ("questionnaire", "result", "review"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d
