from __future__ import annotations

import atexit
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence


CardType = Literal["table", "column", "relationship"]


@dataclass
class GraphCardRecord:
    schema: str
    table: str
    card_type: CardType
    identifier: str
    schema_hash: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=lambda: time.time())

    @property
    def vector_id(self) -> str:
        existing = self.metadata.get("vector_id")
        if existing:
            return existing
        vector_id = f"{self.schema}:{self.table}:{self.card_type}:{self.identifier}"
        self.metadata["vector_id"] = vector_id
        return vector_id


class GraphCache:
    """
    SQLite-backed cache for rendered graph cards (table/column/relationship nodes).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_cards (
                schema_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                card_type TEXT NOT NULL,
                identifier TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (schema_name, table_name, card_type, identifier)
            )
            """
        )
        self.conn.commit()
        atexit.register(self.conn.close)

    def get_table_hash(self, schema: str, table_name: str) -> Optional[str]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT schema_hash
                FROM graph_cards
                WHERE schema_name = ? AND table_name = ? AND card_type = 'table'
                LIMIT 1
                """,
                (schema, table_name),
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def get_schema_hashes(self, schema: str) -> Dict[str, str]:
        """
        Return all cached schema hashes for a schema in a single query.
        Returns dict mapping table_name -> schema_hash.
        """
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT DISTINCT table_name, schema_hash
                FROM graph_cards
                WHERE schema_name = ? AND card_type = 'table'
                """,
                (schema,),
            )
            rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    def replace_table_cards(
        self,
        schema: str,
        table_name: str,
        schema_hash: str,
        cards: Sequence[GraphCardRecord],
    ) -> None:
        payload = [
            (
                card.schema,
                card.table,
                card.card_type,
                card.identifier,
                schema_hash,
                card.text,
                json.dumps({**card.metadata, "vector_id": card.vector_id}),
                card.updated_at,
            )
            for card in cards
        ]
        with self._lock:
            self.conn.execute(
                """
                DELETE FROM graph_cards
                WHERE schema_name = ? AND table_name = ?
                """,
                (schema, table_name),
            )
            if payload:
                self.conn.executemany(
                    """
                    INSERT INTO graph_cards (
                        schema_name,
                        table_name,
                        card_type,
                        identifier,
                        schema_hash,
                        text,
                        metadata_json,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
            self.conn.commit()

    def delete_tables(self, schema: str, tables: Iterable[str]) -> None:
        tables = list(tables)
        if not tables:
            return
        with self._lock:
            self.conn.executemany(
                """
                DELETE FROM graph_cards
                WHERE schema_name = ? AND table_name = ?
                """,
                [(schema, table) for table in tables],
            )
            self.conn.commit()

    def list_tables(self, schema: str) -> List[str]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT DISTINCT table_name
                FROM graph_cards
                WHERE schema_name = ?
                """,
                (schema,),
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def get_cards_for_table(self, schema: str, table_name: str) -> List[GraphCardRecord]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT card_type, identifier, schema_hash, text, metadata_json, updated_at
                FROM graph_cards
                WHERE schema_name = ? AND table_name = ?
                ORDER BY card_type, identifier
                """,
                (schema, table_name),
            )
            rows = cursor.fetchall()
        records: List[GraphCardRecord] = []
        for row in rows:
            metadata = json.loads(row[4]) if row[4] else {}
            records.append(
                GraphCardRecord(
                    schema=schema,
                    table=table_name,
                    card_type=row[0],  # type: ignore[arg-type]
                    identifier=row[1],
                    schema_hash=row[2],
                    text=row[3],
                    metadata=metadata,
                    updated_at=row[5],
                )
            )
        return records

    def iter_cards(self, schema: Optional[str] = None) -> Iterable[GraphCardRecord]:
        query = """
            SELECT schema_name, table_name, card_type, identifier, schema_hash, text, metadata_json, updated_at
            FROM graph_cards
        """
        params: Sequence[str] = ()
        if schema:
            query += " WHERE schema_name = ?"
            params = (schema,)
        with self._lock:
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
        for row in rows:
            metadata = json.loads(row[6]) if row[6] else {}
            yield GraphCardRecord(
                schema=row[0],
                table=row[1],
                card_type=row[2],  # type: ignore[arg-type]
                identifier=row[3],
                schema_hash=row[4],
                text=row[5],
                metadata=metadata,
                updated_at=row[7],
            )


