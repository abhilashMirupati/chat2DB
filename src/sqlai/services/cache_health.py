from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

from sqlai.services.graph_cache import GraphCache, GraphCardRecord
from sqlai.services.metadata_cache import MetadataCache


def metadata_table_names(metadata_cache: MetadataCache, schema: str) -> Set[str]:
    """
    Return the set of table names that have usable metadata in the cache.

    A table counts as "usable" when it has a schema hash and at least one of
    description or samples populated. This mirrors the information we rely on
    for prompting and validation.
    """
    cached_metadata = metadata_cache.fetch_schema(schema)
    table_names: Set[str] = set()
    total_entries = len(cached_metadata)
    for table_name, entry in cached_metadata.items():
        if not entry:
            continue
        if not entry.get("schema_hash"):
            continue
        if not (entry.get("description") or entry.get("samples")):
            continue
        table_names.add(table_name)
    
    # Log if filtering removed entries (for debugging)
    if total_entries > len(table_names):
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "metadata_table_names: Schema '%s' has %s total entries, %s usable (filtered out %s entries without hash/description/samples)",
            schema,
            total_entries,
            len(table_names),
            total_entries - len(table_names),
        )
    return table_names


def graph_vector_id_map(graph_cache: GraphCache, schema: str) -> Dict[str, Set[str]]:
    """
    Build a mapping of table -> set(vector_id) for all graph cards in cache.
    """
    table_vectors: Dict[str, Set[str]] = {}
    for card in graph_cache.iter_cards(schema):
        table_vectors.setdefault(card.table, set()).add(_vector_id_for_card(card))
    return table_vectors


def diff_vector_maps(
    expected: Dict[str, Set[str]],
    actual: Dict[str, Set[str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Compare expected (graph cache) vs actual (vector store) vector IDs.

    Returns:
        missing: table -> vector IDs that are expected but absent in vector store.
        orphaned: table -> vector IDs present in vector store without graph cards.
    """
    missing: Dict[str, Set[str]] = {}
    for table, expected_ids in expected.items():
        actual_ids = actual.get(table, set())
        diff = expected_ids - actual_ids
        if diff:
            missing[table] = diff

    orphaned: Dict[str, Set[str]] = {}
    for table, actual_ids in actual.items():
        expected_ids = expected.get(table)
        if expected_ids is None:
            orphaned[table] = actual_ids
            continue
        extra = actual_ids - expected_ids
        if extra:
            orphaned[table] = extra

    return missing, orphaned


def _vector_id_for_card(card: GraphCardRecord) -> str:
    """
    Ensure we always return the canonical vector_id, even if metadata is missing.
    """
    # GraphCardRecord.vector_id property persists the vector id into metadata if missing.
    return card.vector_id

