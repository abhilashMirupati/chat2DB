"""
Comprehensive tests for multi-hop FK expansion in SemanticRetriever.

Tests various scenarios including:
- Normal multi-hop expansion
- Circular references
- Missing tables
- Deep chains
- Edge cases that could break the implementation
"""

import sys
import unittest
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlai.database.schema_introspector import ColumnMetadata, ForeignKeyDetail
from sqlai.graph.context import ColumnCard, GraphContext, RelationshipCard, TableCard
from sqlai.semantic.retriever import SemanticRetriever


def create_column_metadata(name: str, type_str: str = "VARCHAR(255)", nullable: bool = True) -> ColumnMetadata:
    """Helper to create ColumnMetadata."""
    return ColumnMetadata(
        name=name,
        type=type_str,
        nullable=nullable,
        default=None,
        comment=None,
        sample_values=None,
    )


def create_fk_detail(
    constrained_cols: List[str],
    referred_table: str,
    referred_cols: List[str],
    referred_schema: Optional[str] = None,
) -> ForeignKeyDetail:
    """Helper to create ForeignKeyDetail."""
    return ForeignKeyDetail(
        constrained_columns=constrained_cols,
        referred_schema=referred_schema,
        referred_table=referred_table,
        referred_columns=referred_cols,
    )


class TestFKExpansion(unittest.TestCase):
    """Test suite for FK expansion with various edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal retriever instance - we only need the FK expansion method
        # We'll create it with minimal config to avoid heavy dependencies
        from sqlai.config import EmbeddingConfig
        # Use ollama provider which doesn't require API keys
        config = EmbeddingConfig(
            provider="ollama",
            model="test-model",
            base_url=None,
            api_key=None,
        )
        # Create retriever without vector store to avoid chromadb dependency
        self.retriever = SemanticRetriever(config, vector_store=None)

    def create_mock_graph_context(
        self,
        tables: List[TableCard],
        relationships: List[RelationshipCard],
    ) -> GraphContext:
        """Helper to create a GraphContext from tables and relationships."""
        # Build column cards from tables
        column_cards: List[ColumnCard] = []
        for table in tables:
            for col in table.columns:
                column_cards.append(
                    ColumnCard(
                        schema=table.schema,
                        table=table.name,
                        column=col,
                        sample_values=None,
                    )
                )
        
        return GraphContext(
            schema=None,
            tables=tables,
            relationships=relationships,
            column_cards=column_cards,
        )

    def test_basic_multi_hop_expansion(self):
        """Test basic 5-table chain: orders -> customers -> addresses -> regions -> countries."""
        # Create 5 connected tables
        orders_table = TableCard(
            schema=None,
            name="orders",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("customer_id"),
                create_column_metadata("order_date"),
            ],
            row_estimate=1000,
            comment=None,
        )
        
        customers_table = TableCard(
            schema=None,
            name="customers",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("name"),
                create_column_metadata("address_id"),
            ],
            row_estimate=500,
            comment=None,
        )
        
        addresses_table = TableCard(
            schema=None,
            name="addresses",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("street"),
                create_column_metadata("region_id"),
            ],
            row_estimate=300,
            comment=None,
        )
        
        regions_table = TableCard(
            schema=None,
            name="regions",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("name"),
                create_column_metadata("country_id"),
            ],
            row_estimate=50,
            comment=None,
        )
        
        countries_table = TableCard(
            schema=None,
            name="countries",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("name"),
            ],
            row_estimate=10,
            comment=None,
        )
        
        tables = [orders_table, customers_table, addresses_table, regions_table, countries_table]
        
        # Create relationships (chain)
        relationships = [
            RelationshipCard(
                schema=None,
                table="orders",
                detail=create_fk_detail(["customer_id"], "customers", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="customers",
                detail=create_fk_detail(["address_id"], "addresses", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="addresses",
                detail=create_fk_detail(["region_id"], "regions", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="regions",
                detail=create_fk_detail(["country_id"], "countries", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        # Start with just orders
        selected = [orders_table]
        
        # Expand with max_depth=4 (should get all 5 tables)
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=20,
            max_depth=4,
        )
        
        # Should have all 5 tables
        self.assertEqual(len(expanded), 5, "Should expand to all 5 tables in chain")
        table_names = {t.name for t in expanded}
        self.assertEqual(
            table_names,
            {"orders", "customers", "addresses", "regions", "countries"},
            "Should include all tables in the chain",
        )
        
        # Verify order (orders should be first, then customers, etc.)
        self.assertEqual(expanded[0].name, "orders", "First table should be orders")
        self.assertIn("customers", [t.name for t in expanded], "Should include customers")

    def test_circular_reference(self):
        """Test circular FK reference (should not cause infinite loop)."""
        # Create tables with circular reference: A -> B -> A
        table_a = TableCard(
            schema=None,
            name="table_a",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("b_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_b = TableCard(
            schema=None,
            name="table_b",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("a_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        tables = [table_a, table_b]
        
        relationships = [
            RelationshipCard(
                schema=None,
                table="table_a",
                detail=create_fk_detail(["b_id"], "table_b", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_b",
                detail=create_fk_detail(["a_id"], "table_a", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        # Start with table_a
        selected = [table_a]
        
        # Expand with max_depth=3
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        # Should have both tables (but not infinite loop)
        self.assertEqual(len(expanded), 2, "Should have both tables, no infinite loop")
        table_names = {t.name for t in expanded}
        self.assertEqual(table_names, {"table_a", "table_b"}, "Should include both tables")

    def test_missing_table_in_graph(self):
        """Test FK reference to table that doesn't exist in graph (should skip gracefully)."""
        orders_table = TableCard(
            schema=None,
            name="orders",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("customer_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        tables = [orders_table]
        
        # FK to non-existent table
        relationships = [
            RelationshipCard(
                schema=None,
                table="orders",
                detail=create_fk_detail(["customer_id"], "non_existent_table", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        selected = [orders_table]
        
        # Should not crash, just return original tables
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        self.assertEqual(len(expanded), 1, "Should only have orders table")
        self.assertEqual(expanded[0].name, "orders", "Should be orders table")

    def test_max_depth_limit(self):
        """Test that max_depth limit is respected."""
        # Create a 10-table chain
        tables = []
        relationships = []
        
        for i in range(10):
            table = TableCard(
                schema=None,
                name=f"table_{i}",
                columns=[
                    create_column_metadata("id"),
                    create_column_metadata(f"next_id") if i < 9 else create_column_metadata("id"),
                ],
                row_estimate=100,
                comment=None,
            )
            tables.append(table)
            
            if i < 9:
                rel = RelationshipCard(
                    schema=None,
                    table=f"table_{i}",
                    detail=create_fk_detail(["next_id"], f"table_{i+1}", ["id"]),
                )
                relationships.append(rel)
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        selected = [tables[0]]  # Start with table_0
        
        # Expand with max_depth=3 (should only get 4 tables: 0, 1, 2, 3)
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=20,
            max_depth=3,
        )
        
        # Should have at most 4 tables (0 + 3 levels)
        self.assertLessEqual(len(expanded), 4, "Should respect max_depth=3")
        table_names = {t.name for t in expanded}
        self.assertIn("table_0", table_names, "Should include starting table")
        self.assertIn("table_1", table_names, "Should include level 1")
        self.assertIn("table_2", table_names, "Should include level 2")
        self.assertIn("table_3", table_names, "Should include level 3")
        self.assertNotIn("table_4", table_names, "Should NOT include level 4 (exceeds max_depth)")

    def test_max_expansion_limit(self):
        """Test that max_expansion limit is respected."""
        # Create a star pattern: center table connected to 10 other tables
        center_table = TableCard(
            schema=None,
            name="center",
            columns=[create_column_metadata("id")] + [
                create_column_metadata(f"ref_{i}_id") for i in range(10)
            ],
            row_estimate=100,
            comment=None,
        )
        
        tables = [center_table]
        relationships = []
        
        for i in range(10):
            ref_table = TableCard(
                schema=None,
                name=f"ref_{i}",
                columns=[create_column_metadata("id")],
                row_estimate=100,
                comment=None,
            )
            tables.append(ref_table)
            
            rel = RelationshipCard(
                schema=None,
                table="center",
                detail=create_fk_detail([f"ref_{i}_id"], f"ref_{i}", ["id"]),
            )
            relationships.append(rel)
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        selected = [center_table]
        
        # Expand with max_expansion=5 (should only get 6 tables: center + 5 refs)
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=5,
            max_depth=3,
        )
        
        # Should have at most 6 tables (center + 5 refs)
        self.assertLessEqual(len(expanded), 6, "Should respect max_expansion=5")
        self.assertIn("center", {t.name for t in expanded}, "Should include center table")

    def test_empty_selection(self):
        """Test with empty selection (should return empty list)."""
        graph = self.create_mock_graph_context([], [])
        
        expanded = self.retriever._expand_tables_with_fk_references(
            [],
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        self.assertEqual(len(expanded), 0, "Should return empty list for empty selection")

    def test_no_relationships(self):
        """Test with tables but no relationships (should return original tables)."""
        table = TableCard(
            schema=None,
            name="isolated_table",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        graph = self.create_mock_graph_context([table], [])
        
        selected = [table]
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        self.assertEqual(len(expanded), 1, "Should return original table")
        self.assertEqual(expanded[0].name, "isolated_table", "Should be the same table")

    def test_bidirectional_expansion(self):
        """Test expansion in both directions (outgoing and incoming FKs)."""
        # A -> B <- C (B is referenced by both A and C)
        table_a = TableCard(
            schema=None,
            name="table_a",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("b_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_b = TableCard(
            schema=None,
            name="table_b",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        table_c = TableCard(
            schema=None,
            name="table_c",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("b_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        tables = [table_a, table_b, table_c]
        
        relationships = [
            RelationshipCard(
                schema=None,
                table="table_a",
                detail=create_fk_detail(["b_id"], "table_b", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_c",
                detail=create_fk_detail(["b_id"], "table_b", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        # Start with table_b
        selected = [table_b]
        
        # Should expand to include both A and C (incoming FKs)
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=2,
        )
        
        table_names = {t.name for t in expanded}
        self.assertIn("table_a", table_names, "Should include table_a (incoming FK)")
        self.assertIn("table_c", table_names, "Should include table_c (incoming FK)")
        self.assertIn("table_b", table_names, "Should include table_b (original)")

    def test_duplicate_tables_in_selection(self):
        """Test with duplicate tables in initial selection (should deduplicate)."""
        table = TableCard(
            schema=None,
            name="orders",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        graph = self.create_mock_graph_context([table], [])
        
        # Duplicate table in selection
        selected = [table, table, table]
        
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        # Should deduplicate
        self.assertEqual(len(expanded), 1, "Should deduplicate duplicate tables")
        self.assertEqual(expanded[0].name, "orders", "Should be orders table")

    def test_complex_network(self):
        """Test complex network with multiple paths to same table."""
        # Network: A -> B -> D, A -> C -> D (two paths to D)
        table_a = TableCard(
            schema=None,
            name="table_a",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("b_id"),
                create_column_metadata("c_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_b = TableCard(
            schema=None,
            name="table_b",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("d_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_c = TableCard(
            schema=None,
            name="table_c",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("d_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_d = TableCard(
            schema=None,
            name="table_d",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        tables = [table_a, table_b, table_c, table_d]
        
        relationships = [
            RelationshipCard(
                schema=None,
                table="table_a",
                detail=create_fk_detail(["b_id"], "table_b", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_a",
                detail=create_fk_detail(["c_id"], "table_c", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_b",
                detail=create_fk_detail(["d_id"], "table_d", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_c",
                detail=create_fk_detail(["d_id"], "table_d", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        selected = [table_a]
        
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        # Should have all 4 tables, D should only appear once (deduplicated)
        table_names = {t.name for t in expanded}
        self.assertEqual(len(table_names), 4, "Should have all 4 tables")
        self.assertEqual(table_names, {"table_a", "table_b", "table_c", "table_d"})
        
        # Verify D appears only once in the list
        d_count = sum(1 for t in expanded if t.name == "table_d")
        self.assertEqual(d_count, 1, "Table D should appear only once (deduplicated)")

    def test_schema_qualified_tables(self):
        """Test with schema-qualified tables."""
        table_a = TableCard(
            schema="schema1",
            name="table_a",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("b_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_b = TableCard(
            schema="schema1",
            name="table_b",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        tables = [table_a, table_b]
        
        relationships = [
            RelationshipCard(
                schema="schema1",
                table="table_a",
                detail=create_fk_detail(["b_id"], "table_b", ["id"], referred_schema="schema1"),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        selected = [table_a]
        
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        self.assertEqual(len(expanded), 2, "Should expand to both tables")
        table_names = {t.name for t in expanded}
        self.assertEqual(table_names, {"table_a", "table_b"})

    def test_multiple_selected_tables_reference_same_table(self):
        """Test deduplication when multiple selected tables reference the same table.
        
        Scenario:
        - Table A and Table B are both selected (semantically matched)
        - Both A and B have FK to Table C
        - Table C should only appear once in the result (deduplicated)
        """
        table_a = TableCard(
            schema=None,
            name="orders",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("customer_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_b = TableCard(
            schema=None,
            name="invoices",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("customer_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_c = TableCard(
            schema=None,
            name="customers",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        tables = [table_a, table_b, table_c]
        
        # Both A and B reference C
        relationships = [
            RelationshipCard(
                schema=None,
                table="orders",
                detail=create_fk_detail(["customer_id"], "customers", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="invoices",
                detail=create_fk_detail(["customer_id"], "customers", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        # Both A and B are selected (semantically matched)
        selected = [table_a, table_b]
        
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        # Should have all 3 tables, but C should appear only once
        table_names = {t.name for t in expanded}
        self.assertEqual(len(table_names), 3, "Should have all 3 unique tables")
        self.assertEqual(table_names, {"orders", "invoices", "customers"})
        
        # Verify C appears only once in the list (not duplicated)
        c_count = sum(1 for t in expanded if t.name == "customers")
        self.assertEqual(c_count, 1, "Table C should appear only once (deduplicated)")

    def test_selected_table_already_in_expansion_path(self):
        """Test when a selected table is also in the expansion path from another selected table.
        
        Scenario:
        - Table A is selected
        - Table B is also selected (semantically matched)
        - A has FK to B
        - B should not be added again during expansion (already in selected)
        """
        table_a = TableCard(
            schema=None,
            name="orders",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("customer_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_b = TableCard(
            schema=None,
            name="customers",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        tables = [table_a, table_b]
        
        relationships = [
            RelationshipCard(
                schema=None,
                table="orders",
                detail=create_fk_detail(["customer_id"], "customers", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        # Both A and B are selected (B is semantically matched, not just from FK)
        selected = [table_a, table_b]
        
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        # Should have both tables, but B should appear only once
        table_names = {t.name for t in expanded}
        self.assertEqual(len(table_names), 2, "Should have both tables")
        self.assertEqual(table_names, {"orders", "customers"})
        
        # Verify B appears only once (not added again during expansion)
        b_count = sum(1 for t in expanded if t.name == "customers")
        self.assertEqual(b_count, 1, "Table B should appear only once (already in selected)")

    def test_deep_chain_with_multiple_paths(self):
        """Test deep chain where multiple selected tables create paths to same table.
        
        Scenario:
        - Table A and Table B are both selected
        - A -> C -> E
        - B -> D -> E
        - Table E should appear only once (deduplicated from both paths)
        """
        table_a = TableCard(
            schema=None,
            name="table_a",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("c_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_b = TableCard(
            schema=None,
            name="table_b",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("d_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_c = TableCard(
            schema=None,
            name="table_c",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("e_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_d = TableCard(
            schema=None,
            name="table_d",
            columns=[
                create_column_metadata("id"),
                create_column_metadata("e_id"),
            ],
            row_estimate=100,
            comment=None,
        )
        
        table_e = TableCard(
            schema=None,
            name="table_e",
            columns=[create_column_metadata("id")],
            row_estimate=100,
            comment=None,
        )
        
        tables = [table_a, table_b, table_c, table_d, table_e]
        
        relationships = [
            RelationshipCard(
                schema=None,
                table="table_a",
                detail=create_fk_detail(["c_id"], "table_c", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_b",
                detail=create_fk_detail(["d_id"], "table_d", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_c",
                detail=create_fk_detail(["e_id"], "table_e", ["id"]),
            ),
            RelationshipCard(
                schema=None,
                table="table_d",
                detail=create_fk_detail(["e_id"], "table_e", ["id"]),
            ),
        ]
        
        graph = self.create_mock_graph_context(tables, relationships)
        
        # Both A and B are selected
        selected = [table_a, table_b]
        
        expanded = self.retriever._expand_tables_with_fk_references(
            selected,
            graph,
            max_expansion=10,
            max_depth=3,
        )
        
        # Should have all 5 tables
        table_names = {t.name for t in expanded}
        self.assertEqual(len(table_names), 5, "Should have all 5 tables")
        self.assertEqual(table_names, {"table_a", "table_b", "table_c", "table_d", "table_e"})
        
        # Verify E appears only once (reached via both paths: A->C->E and B->D->E)
        e_count = sum(1 for t in expanded if t.name == "table_e")
        self.assertEqual(e_count, 1, "Table E should appear only once (deduplicated from both paths)")


if __name__ == "__main__":
    unittest.main()

