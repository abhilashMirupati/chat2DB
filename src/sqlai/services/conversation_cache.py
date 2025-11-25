from __future__ import annotations

import atexit
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


def _normalize_schema_name(schema: str) -> str:
    """
    Normalize schema name to lowercase for SQLite storage and retrieval.
    This ensures consistent schema name handling regardless of case in user input.
    """
    if not schema:
        return schema
    return schema.lower().strip()


class ConversationCache:
    """
    SQLite-backed cache of successful question/answer interactions.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_name TEXT NOT NULL,
                question TEXT NOT NULL,
                sql_text TEXT NOT NULL,
                plan_json TEXT,
                summary_text TEXT,
                chart_json TEXT,
                created_at REAL NOT NULL,
                UNIQUE(schema_name, question)
            )
            """
        )
        self.conn.commit()
        atexit.register(self.conn.close)

    def save_interaction(
        self,
        schema: str,
        question: str,
        sql_text: str,
        plan: Optional[Dict[str, object]] = None,
        summary: Optional[str] = None,
        chart: Optional[Dict[str, object]] = None,
    ) -> None:
        schema = _normalize_schema_name(schema)
        payload_plan = json.dumps(plan or {}, ensure_ascii=False)
        payload_chart = json.dumps(chart or {}, ensure_ascii=False) if chart else None
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO saved_queries
                    (schema_name, question, sql_text, plan_json, summary_text, chart_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schema,
                    question.strip(),
                    sql_text.strip(),
                    payload_plan,
                    summary.strip() if summary else None,
                    payload_chart,
                    time.time(),
                ),
            )
            self.conn.commit()

    def list_interactions(self, schema: str, limit: int = 20) -> List[Dict[str, object]]:
        schema = _normalize_schema_name(schema)
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, question, sql_text, summary_text, created_at
                FROM saved_queries
                WHERE schema_name = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (schema, limit),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "question": row[1],
                "sql": row[2],
                "summary": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def get_interaction(self, entry_id: int) -> Optional[Dict[str, object]]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, schema_name, question, sql_text, plan_json, summary_text, chart_json, created_at
                FROM saved_queries
                WHERE id = ?
                """,
                (entry_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        plan = json.loads(row[4]) if row[4] else {}
        chart = json.loads(row[6]) if row[6] else None
        return {
            "id": row[0],
            "schema": row[1],
            "question": row[2],
            "sql": row[3],
            "plan": plan,
            "summary": row[5],
            "chart": chart,
            "created_at": row[7],
        }

    def get_recent_questions(self, schema: str, limit: int = 20) -> List[Dict[str, object]]:
        """
        Get recent questions for similarity checking.
        Returns questions with their SQL for semantic comparison.
        """
        schema = _normalize_schema_name(schema)
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, question, sql_text, summary_text, created_at
                FROM saved_queries
                WHERE schema_name = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (schema, limit),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "question": row[1],
                "sql": row[2],
                "summary": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

