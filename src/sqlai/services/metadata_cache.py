from __future__ import annotations

import atexit
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Optional


class MetadataCache:
    """
    SQLite-backed cache for table metadata (descriptions, sample values, schema hashes).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS table_metadata (
                schema_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                description TEXT,
                samples_json TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (schema_name, table_name)
            )
            """
        )
        self.conn.commit()
        atexit.register(self.conn.close)

    def fetch(self, schema: str, table_name: str) -> Optional[Dict[str, object]]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT schema_hash, description, samples_json
                FROM table_metadata
                WHERE schema_name = ? AND table_name = ?
                """,
                (schema, table_name),
            )
            row = cursor.fetchone()
        if not row:
            return None
        samples = json.loads(row[2]) if row[2] else {}
        return {
            "schema_hash": row[0],
            "description": row[1],
            "samples": samples,
        }

    def upsert(
        self,
        schema: str,
        table_name: str,
        schema_hash: str,
        description: Optional[str],
        samples: Optional[Dict[str, list]] = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO table_metadata
                (schema_name, table_name, schema_hash, description, samples_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    schema,
                    table_name,
                    schema_hash,
                    description,
                    json.dumps(samples or {}),
                    time.time(),
                ),
            )
            self.conn.commit()

    def update_samples(
        self,
        schema: str,
        table_name: str,
        schema_hash: str,
        samples: Dict[str, list],
    ) -> None:
        existing = self.fetch(schema, table_name)
        if existing and existing["schema_hash"] == schema_hash:
            with self._lock:
                self.conn.execute(
                    """
                    UPDATE table_metadata
                    SET samples_json = ?, updated_at = ?
                    WHERE schema_name = ? AND table_name = ?
                    """,
                    (json.dumps(samples), time.time(), schema, table_name),
                )
                self.conn.commit()

    def iter_metadata(self):
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT schema_name, table_name, schema_hash, description, samples_json, updated_at
                FROM table_metadata
                ORDER BY schema_name, table_name
                """
            )
            rows = cursor.fetchall()
        for row in rows:
            yield {
                "schema_name": row[0],
                "table_name": row[1],
                "schema_hash": row[2],
                "description": row[3],
                "samples": json.loads(row[4]) if row[4] else {},
                "updated_at": row[5],
            }

    def fetch_schema(self, schema: str) -> Dict[str, Dict[str, object]]:
        """
        Return all cached metadata entries for a schema in a single query.
        """
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT table_name, schema_hash, description, samples_json
                FROM table_metadata
                WHERE schema_name = ?
                """,
                (schema,),
            )
            rows = cursor.fetchall()

        cached: Dict[str, Dict[str, object]] = {}
        for row in rows:
            table_name, schema_hash, description, samples_json = row
            cached[table_name] = {
                "schema_hash": schema_hash,
                "description": description,
                "samples": json.loads(samples_json) if samples_json else {},
            }
        return cached

