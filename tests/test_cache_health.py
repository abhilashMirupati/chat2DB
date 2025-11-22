import unittest
from typing import Dict

from sqlai.services.cache_health import (
    diff_vector_maps,
    graph_vector_id_map,
    metadata_table_names,
)
from sqlai.services.graph_cache import GraphCardRecord


class FakeMetadataCache:
    def __init__(self, entries: Dict[str, Dict[str, Dict[str, object]]]) -> None:
        self._entries = entries

    def fetch_schema(self, schema: str) -> Dict[str, Dict[str, object]]:
        return self._entries.get(schema, {})


class FakeGraphCache:
    def __init__(self, cards) -> None:
        self._cards = cards

    def iter_cards(self, schema: str | None = None):
        for card in self._cards:
            if schema is None or card.schema == schema:
                yield card


class TestCacheHealth(unittest.TestCase):
    def test_metadata_table_names_filters_incomplete_entries(self) -> None:
        cache = FakeMetadataCache(
            {
                "sales": {
                    "orders": {
                        "schema_hash": "abc",
                        "description": "Orders table",
                    },
                    "customers": {
                        "schema_hash": "def",
                        "samples": {"name": ["Alice"]},
                    },
                    "missing_hash": {
                        "schema_hash": "",
                        "description": "incomplete",
                    },
                    "missing_content": {
                        "schema_hash": "ghi",
                    },
                }
            }
        )

        result = metadata_table_names(cache, "sales")

        self.assertEqual(result, {"orders", "customers"})

    def test_graph_vector_id_map_uses_canonical_vector_ids(self) -> None:
        cards = [
            GraphCardRecord(
                schema="sales",
                table="orders",
                card_type="table",
                identifier="__table__",
                schema_hash="abc",
                text="orders table",
                metadata={"vector_id": "custom-vector"},
            ),
            GraphCardRecord(
                schema="sales",
                table="orders",
                card_type="column",
                identifier="order_id",
                schema_hash="abc",
                text="order_id column",
                metadata={},
            ),
        ]
        cache = FakeGraphCache(cards)

        result = graph_vector_id_map(cache, "sales")

        self.assertIn("orders", result)
        self.assertEqual(
            result["orders"],
            {"custom-vector", "sales:orders:column:order_id"},
        )

    def test_diff_vector_maps(self) -> None:
        expected = {
            "orders": {"orders:table", "orders:column:id"},
            "customers": {"customers:table"},
        }
        actual = {
            "orders": {"orders:table"},
            "customers": {"customers:table", "customers:column:extra"},
            "legacy": {"legacy:table"},
        }

        missing, orphaned = diff_vector_maps(expected, actual)

        self.assertEqual(missing, {"orders": {"orders:column:id"}})
        self.assertEqual(
            orphaned,
            {
                "customers": {"customers:column:extra"},
                "legacy": {"legacy:table"},
            },
        )


if __name__ == "__main__":
    unittest.main()

