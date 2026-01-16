import sqlite3
import json
import time
from typing import List, Dict, Any
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"

class Memory:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages (session_id)")
            
            # Trace Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    trace_data TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_session ON traces (session_id)")

    def save_trace(self, session_id: str, trace_data: List[Dict[str, Any]]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO traces (session_id, trace_data, timestamp) VALUES (?, ?, ?)",
                (session_id, json.dumps(trace_data), time.time())
            )

    def add_message(self, session_id: str, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time())
            )

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            # Reverse to get chronological order
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def clear_history(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
