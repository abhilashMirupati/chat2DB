"""
Graph-oriented prompt context helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlai.database.schema_introspector import ColumnMetadata, ForeignKeyDetail, TableSummary


@dataclass
class TableCard:
    schema: Optional[str]
    name: str
    columns: List[ColumnMetadata]
    row_estimate: Optional[int]
    comment: Optional[str]

    def render(self) -> str:
        header = f"{self.schema}.{self.name}" if self.schema else self.name
        row_text = f"rows≈{self.row_estimate}" if self.row_estimate is not None else "rows≈unknown"
        lines = [f"{header} | {row_text}"]
        if self.comment:
            lines.append(f"  comment: {self.comment}")
        column_desc = ", ".join(
            f"{col.name} ({col.type}{', nullable' if col.nullable else ''})"
            for col in self.columns
        )
        if column_desc:
            lines.append(f"  columns: {column_desc}")
        return "\n".join(lines)


@dataclass
class ColumnCard:
    schema: Optional[str]
    table: str
    column: ColumnMetadata
    sample_values: Optional[List[str]] = None

    def render(self) -> str:
        schema_prefix = f"{self.schema}." if self.schema else ""
        parts = [
            f"{schema_prefix}{self.table}.{self.column.name}",
            f"type={self.column.type}",
            f"nullable={self.column.nullable}",
        ]
        if self.column.comment:
            parts.append(f"comment={self.column.comment}")
        if self.column.default:
            parts.append(f"default={self.column.default}")
        if self.sample_values:
            joined = ", ".join(self.sample_values[:5])
            parts.append(f"values≈[{joined}]")
        return " | ".join(parts)

    def fact(self) -> str:
        schema_prefix = f"{self.schema}." if self.schema else ""
        return (
            f"{schema_prefix}{self.table}.{self.column.name}: "
            f"type={self.column.type}, nullable={self.column.nullable}, "
            f"default={self.column.default}"
            + (
                f", values≈{self.sample_values[:5]}"
                if self.sample_values
                else ""
            )
        )


@dataclass
class RelationshipCard:
    schema: Optional[str]
    table: str
    detail: ForeignKeyDetail

    def render(self) -> str:
        left_schema = f"{self.schema}." if self.schema else ""
        left = ",".join(self.detail.constrained_columns) or "??"
        right_schema = f"{self.detail.referred_schema}." if self.detail.referred_schema else left_schema
        right_cols = ",".join(self.detail.referred_columns) or "??"
        return (
            f"{left_schema}{self.table}[{left}] -> "
            f"{right_schema}{self.detail.referred_table}[{right_cols}]"
        )


class GraphContext:
    def __init__(
        self,
        schema: Optional[str],
        tables: List[TableCard],
        relationships: List[RelationshipCard],
        column_cards: List[ColumnCard],
        sensitive_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.schema = schema
        self.tables = tables
        self.relationships = relationships
        self.column_cards = column_cards
        self.sensitive_columns: List[str] = list(sensitive_columns or [])
        self.schemas: Set[str] = {schema} if schema else set()
        self.schemas.update(card.schema for card in tables if card.schema)

    def prepare_prompt_inputs(
        self,
        *,
        question: str,
        dialect_guide: str,
        token_budget: int,
        row_cap: int,
        tables: Optional[List[TableCard]] = None,
        columns: Optional[List[ColumnCard]] = None,
        relationships: Optional[List[RelationshipCard]] = None,
    ) -> Dict[str, str]:
        # Use provided tables/columns/relationships if given, otherwise fall back to ranking
        # Note: Check for None explicitly (not just truthiness) to allow empty lists
        selected_tables = tables if tables is not None else self.rank_tables(question, max_cards=6)
        selected_columns = columns if columns is not None else self.rank_columns(question, selected_tables, max_cards=10)
        selected_relationships = relationships if relationships is not None else self.relationships_for_tables(selected_tables)

        schema_names = ", ".join(sorted(filter(None, self.schemas))) or "(not specified)"
        table_cards_text = "\n\n".join(card.render() for card in selected_tables) or "None"
        column_cards_text = "\n".join(card.render() for card in selected_columns) or "None"
        column_facts_text = "\n".join(card.fact() for card in selected_columns) or "None"
        relationship_map_text = "\n".join(rel.render() for rel in selected_relationships) or "None"
        value_anchor_text = "No value anchors collected."
        
        # Log what's being included in the prompt for verification
        from sqlai.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Prompt context summary: %d tables, %d columns, %d relationships", 
                   len(selected_tables), len(selected_columns), len(selected_relationships))
        logger.debug("Table cards text length: %d chars", len(table_cards_text))
        logger.debug("Column cards text length: %d chars", len(column_cards_text))
        logger.debug("Relationship map text length: %d chars", len(relationship_map_text))
        # Log a sample to verify content is present
        if table_cards_text and table_cards_text != "None":
            logger.debug("Table cards sample (first 200 chars): %s", table_cards_text[:200])
        if column_cards_text and column_cards_text != "None":
            logger.debug("Column cards sample (first 200 chars): %s", column_cards_text[:200])
        if relationship_map_text and relationship_map_text != "None":
            logger.debug("Relationship map sample (first 200 chars): %s", relationship_map_text[:200])

        prompt = {
            "dialect_guide": dialect_guide,
            "schemas_short": schema_names,
            "k_tables": str(len(selected_tables)),
            "table_cards": table_cards_text,
            "k_columns": str(len(selected_columns)),
            "column_cards": column_cards_text,
            "relationship_map": relationship_map_text,
            "column_facts": column_facts_text,
            "value_anchors": value_anchor_text,
            "default_schema": self.schema or "(not set)",
            "token_budget": str(token_budget),
            "row_cap": str(row_cap),
            "sensitive_columns": str(self.sensitive_columns or []),
            "user_question": question,
        }
        prompt["__selected_tables__"] = [card.name for card in selected_tables]
        prompt["__selected_columns__"] = [
            (card.table, card.column.name) for card in selected_columns
        ]
        return prompt

    def update_column_samples(
        self,
        samples: Dict[str, Dict[str, List[str]]],
    ) -> None:
        if not samples:
            return
        # Update column cards
        for card in self.column_cards:
            table_samples = samples.get(card.table)
            if table_samples:
                values = table_samples.get(card.column.name)
                if values:
                    card.sample_values = values[:5]
        # Update table metadata
        for table_card in self.tables:
            table_samples = samples.get(table_card.name)
            if not table_samples:
                continue
            for column in table_card.columns:
                values = table_samples.get(column.name)
                if values:
                    object.__setattr__(column, "sample_values", values[:5])

    def serialize_metadata(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "tables": [
                {
                    "schema": card.schema,
                    "name": card.name,
                    "row_estimate": card.row_estimate,
                    "columns": [
                        {
                            "name": column.name,
                            "type": column.type,
                            "nullable": column.nullable,
                            "default": str(column.default) if column.default is not None else None,
                            "comment": column.comment,
                        }
                        for column in card.columns
                    ],
                }
                for card in self.tables
            ],
            "relationships": [
                {
                    "from_schema": rel.schema,
                    "from_table": rel.table,
                    "from_columns": rel.detail.constrained_columns,
                    "to_schema": rel.detail.referred_schema,
                    "to_table": rel.detail.referred_table,
                    "to_columns": rel.detail.referred_columns,
                }
                for rel in self.relationships
            ],
        }

    def find_columns(self, keywords: Sequence[str]) -> List[ColumnCard]:
        lowered = [kw.lower() for kw in keywords]
        matches: List[ColumnCard] = []
        for card in self.column_cards:
            haystack = f"{card.table}.{card.column.name} {card.column.comment or ''}".lower()
            if all(keyword in haystack for keyword in lowered):
                matches.append(card)
        return matches

    def rank_tables(self, question: str, max_cards: int) -> List[TableCard]:
        question_terms = _tokenize(question)
        scored: List[Tuple[int, TableCard]] = []
        for card in self.tables:
            tokens = {card.name.lower()}
            tokens.update(col.name.lower() for col in card.columns)
            score = sum(1 for term in question_terms if term in tokens)
            scored.append((score, card))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [card for score, card in scored if score > 0][:max_cards]
        if not selected:
            selected = [card for _, card in scored[:max_cards]]
        return selected

    def rank_columns(
        self,
        question: str,
        tables: List[TableCard],
        max_cards: int,
    ) -> List[ColumnCard]:
        question_terms = _tokenize(question)
        relevant_tables = {card.name for card in tables}
        scored: List[Tuple[int, ColumnCard]] = []
        for card in self.column_cards:
            if card.table not in relevant_tables:
                continue
            tokens = {card.column.name.lower(), card.table.lower()}
            score = sum(1 for term in question_terms if term in tokens)
            scored.append((score, card))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [card for score, card in scored if score > 0][:max_cards]
        if not selected:
            selected = [card for _, card in scored[:max_cards]]
        return selected

    def relationships_for_tables(self, tables: List[TableCard]) -> List[RelationshipCard]:
        table_names = {card.name for card in tables}
        rels = [
            rel
            for rel in self.relationships
            if rel.table in table_names or rel.detail.referred_table in table_names
        ]
        return rels  # Return ALL relationships, no truncation


def build_graph_context(
    summaries: Iterable[TableSummary],
    schema: Optional[str],
) -> GraphContext:
    tables: List[TableCard] = []
    column_cards: List[ColumnCard] = []
    relationships: List[RelationshipCard] = []

    for summary in summaries:
        table_card = TableCard(
            schema=schema,
            name=summary.name,
            columns=summary.columns,
            row_estimate=summary.row_estimate,
            comment=summary.comment,
        )
        tables.append(table_card)
        for column in summary.columns:
            column_cards.append(
                ColumnCard(
                    schema=schema,
                    table=summary.name,
                    column=column,
                    sample_values=column.sample_values,
                )
            )
        for fk in summary.foreign_keys:
            relationships.append(
                RelationshipCard(
                    schema=schema,
                    table=summary.name,
                    detail=fk,
                )
            )

    return GraphContext(
        schema=schema,
        tables=tables,
        relationships=relationships,
        column_cards=column_cards,
    )


def _tokenize(text: str) -> Set[str]:
    return {token for token in _split_words(text.lower()) if token}


def _split_words(text: str) -> List[str]:
    word = ""
    words: List[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            word += char
        else:
            if word:
                words.append(word)
                word = ""
    if word:
        words.append(word)
    return words

