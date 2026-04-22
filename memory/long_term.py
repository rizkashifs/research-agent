import json
import sqlite3
from pathlib import Path
from config import SQLITE_DB_PATH


class LongTermMemory:
    def __init__(self, db_path: str = SQLITE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        schema = Path(__file__).parent / "schema.sql"
        conn = self._connect()
        with conn:
            conn.executescript(schema.read_text())
        conn.close()

    def create_session(self, query: str) -> int:
        conn = self._connect()
        with conn:
            cur = conn.execute(
                "INSERT INTO sessions (query) VALUES (?)", (query,)
            )
            session_id = cur.lastrowid
        conn.close()
        return session_id

    def save_note(self, content: str, topic: str, session_id: int | None = None):
        conn = self._connect()
        with conn:
            conn.execute(
                "INSERT INTO notes (session_id, content, topic) VALUES (?, ?, ?)",
                (session_id, content, topic),
            )
        conn.close()

    def get_recent_notes(self, limit: int = 5) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT content, topic, timestamp FROM notes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def save_short_term_snapshot(self, session_id: int, messages: list[dict]):
        conn = self._connect()
        with conn:
            conn.execute(
                "INSERT INTO short_term_snapshots (session_id, messages) VALUES (?, ?)",
                (session_id, json.dumps(messages)),
            )
        conn.close()

    def load_last_short_term_snapshot(self) -> list[dict]:
        conn = self._connect()
        row = conn.execute(
            "SELECT messages FROM short_term_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row["messages"])
        return []

    def get_last_session_id(self) -> int | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT id FROM sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row["id"] if row else None
