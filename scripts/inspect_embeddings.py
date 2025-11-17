#!/usr/bin/env python3
"""
Inspect embeddings stored in ChromaDB vector store.

This script shows what graph cards have been embedded and stored in the vector store.
"""

import json
import sys
from pathlib import Path

try:
    import chromadb
except ImportError:
    print("Error: chromadb is not installed. Install it with: pip install chromadb")
    sys.exit(1)

from sqlai.config import EmbeddingConfig, VectorStoreConfig, load_embedding_config, load_vector_store_config


def verify_cards_vs_embeddings(graph_cards_db: Path, chroma_collection) -> None:
    """Verify that all cards in graph_cards.db have embeddings in ChromaDB."""
    import sqlite3
    
    if not graph_cards_db.exists():
        print(f"\n⚠️  Graph cards database not found at: {graph_cards_db}")
        return
    
    # Read all cards from graph_cards.db
    conn = sqlite3.connect(str(graph_cards_db))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT schema_name, table_name, card_type, identifier, metadata_json
        FROM graph_cards
    """)
    cards = cursor.fetchall()
    conn.close()
    
    # Get all embeddings from ChromaDB
    chroma_results = chroma_collection.get(include=["metadatas"])
    chroma_ids = set(chroma_results.get("ids", []))
    
    # Build expected vector IDs from cards
    expected_ids = set()
    cards_by_id = {}
    for schema, table, card_type, identifier, metadata_json in cards:
        metadata = json.loads(metadata_json) if metadata_json else {}
        vector_id = metadata.get("vector_id") or f"{schema}:{table}:{card_type}:{identifier}"
        expected_ids.add(vector_id)
        cards_by_id[vector_id] = {
            "schema": schema,
            "table": table,
            "card_type": card_type,
            "identifier": identifier,
        }
    
    # Compare
    missing_in_chroma = expected_ids - chroma_ids
    orphaned_in_chroma = chroma_ids - expected_ids
    
    print("\n" + "="*80)
    print("Verification: Graph Cards vs ChromaDB Embeddings")
    print("="*80)
    print(f"\nTotal cards in graph_cards.db: {len(cards)}")
    print(f"Total embeddings in ChromaDB: {len(chroma_ids)}")
    print(f"Expected vector IDs: {len(expected_ids)}")
    
    if missing_in_chroma:
        print(f"\n⚠️  WARNING: {len(missing_in_chroma)} card(s) missing embeddings in ChromaDB:")
        for vector_id in sorted(missing_in_chroma)[:10]:
            card_info = cards_by_id.get(vector_id, {})
            print(f"  - {vector_id} ({card_info.get('card_type', 'unknown')})")
        if len(missing_in_chroma) > 10:
            print(f"  ... and {len(missing_in_chroma) - 10} more")
    else:
        print("\n✅ All cards have embeddings in ChromaDB")
    
    if orphaned_in_chroma:
        print(f"\n⚠️  WARNING: {len(orphaned_in_chroma)} orphaned embedding(s) in ChromaDB (no matching card):")
        for vector_id in sorted(orphaned_in_chroma)[:10]:
            print(f"  - {vector_id}")
        if len(orphaned_in_chroma) > 10:
            print(f"  ... and {len(orphaned_in_chroma) - 10} more")
    else:
        print("\n✅ No orphaned embeddings found")
    
    if not missing_in_chroma and not orphaned_in_chroma:
        print("\n✅ Perfect sync: All cards have embeddings, all embeddings have cards")
    
    print("="*80)


def inspect_embeddings():
    """Inspect and display all embeddings in the vector store."""
    embedding_config = load_embedding_config()
    vector_config = load_vector_store_config()
    
    # Compute the store path (same logic as VectorStoreManager)
    cache_dir = Path(".cache")
    namespace = f"{embedding_config.provider}__{embedding_config.model.replace('/', '_')}"
    store_path = cache_dir / "vector_store" / namespace
    
    if not store_path.exists():
        print(f"Vector store not found at: {store_path}")
        print("Run prewarm_metadata.py first to generate embeddings.")
        return
    
    print(f"Loading vector store from: {store_path}")
    
    # First, let's inspect the SQLite database structure directly
    import sqlite3
    db_path = store_path / "chroma.sqlite3"
    if db_path.exists():
        print("\n" + "="*80)
        print("ChromaDB SQLite Database Structure:")
        print("="*80)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\nTables found: {len(tables)}")
        for (table_name,) in tables:
            print(f"  - {table_name}")
        
        # Show structure of embeddings table
        print("\n" + "-"*80)
        print("'embeddings' table structure:")
        print("-"*80)
        try:
            cursor.execute("PRAGMA table_info(embeddings)")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  Column: {col[1]} | Type: {col[2]} | Nullable: {not col[3]}")
        except Exception as e:
            print(f"  Could not read embeddings table: {e}")
        
        # Show structure of segments table (where vectors are actually stored)
        print("\n" + "-"*80)
        print("'segments' table structure (where vectors are organized):")
        print("-"*80)
        try:
            cursor.execute("PRAGMA table_info(segments)")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  Column: {col[1]} | Type: {col[2]} | Nullable: {not col[3]}")
            
            # Check if segments table has data
            cursor.execute("SELECT COUNT(*) FROM segments")
            seg_count = cursor.fetchone()[0]
            print(f"\n  Total segments: {seg_count}")
            
            # Check for BLOB columns that might contain vectors
            blob_cols = [col[1] for col in columns if 'BLOB' in col[2].upper() or 'BYTE' in col[2].upper()]
            if blob_cols:
                print(f"  Columns that may contain vector data: {', '.join(blob_cols)}")
        except Exception as e:
            print(f"  Could not read segments table: {e}")
        
        # Show sample data from embeddings table
        print("\n" + "-"*80)
        print("Sample from 'embeddings' table (first 3 rows):")
        print("-"*80)
        try:
            cursor.execute("SELECT embedding_id, segment_id FROM embeddings LIMIT 3")
            rows = cursor.fetchall()
            for row in rows:
                print(f"  embedding_id: {row[0]}")
                print(f"  segment_id: {row[1]}")
                print()
        except Exception as e:
            print(f"  Could not read embeddings data: {e}")
        
        # Check segment_metadata table
        print("\n" + "-"*80)
        print("'segment_metadata' table structure:")
        print("-"*80)
        try:
            cursor.execute("PRAGMA table_info(segment_metadata)")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  Column: {col[1]} | Type: {col[2]} | Nullable: {not col[3]}")
            
            # Show sample segment metadata
            cursor.execute("SELECT * FROM segment_metadata LIMIT 5")
            rows = cursor.fetchall()
            if rows:
                print("\n  Sample segment metadata:")
                for row in rows:
                    print(f"    {dict(zip([col[1] for col in columns], row))}")
        except Exception as e:
            print(f"  Could not read segment_metadata table: {e}")
        
        # Check segments table data
        print("\n" + "-"*80)
        print("'segments' table data:")
        print("-"*80)
        try:
            cursor.execute("SELECT * FROM segments")
            seg_rows = cursor.fetchall()
            cursor.execute("PRAGMA table_info(segments)")
            seg_columns = [col[1] for col in cursor.fetchall()]
            for row in seg_rows:
                seg_dict = dict(zip(seg_columns, row))
                print(f"  Segment: {seg_dict}")
        except Exception as e:
            print(f"  Could not read segments data: {e}")
        
        # Check if seq_id BLOB contains vector data
        print("\n" + "-"*80)
        print("Checking 'seq_id' BLOB column size (might contain vector references):")
        print("-"*80)
        try:
            cursor.execute("SELECT embedding_id, LENGTH(seq_id) as seq_size FROM embeddings LIMIT 5")
            seq_rows = cursor.fetchall()
            for row in seq_rows:
                print(f"  {row[0]}: seq_id size = {row[1]} bytes")
        except Exception as e:
            print(f"  Could not check seq_id: {e}")
        
        # Check database size breakdown
        print("\n" + "-"*80)
        print("Database size analysis:")
        print("-"*80)
        db_size = db_path.stat().st_size
        print(f"  Total database size: {db_size / 1024:.2f} KB ({db_size / (1024*1024):.2f} MB)")
        
        # Estimate vector data size
        estimated_vector_size = 34 * 768 * 4  # 34 embeddings × 768 dims × 4 bytes per float
        print(f"  Estimated vector data size: {estimated_vector_size / 1024:.2f} KB")
        print(f"  Other data (metadata, indexes, etc.): {(db_size - estimated_vector_size) / 1024:.2f} KB")
        
        # Check for any tables that might contain vector data
        print("\n" + "-"*80)
        print("Checking for tables that might contain actual vector data:")
        print("-"*80)
        vector_segment_id = None
        for (table_name,) in tables:
            if table_name in ['embeddings', 'segments', 'segment_metadata']:
                continue
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                if count > 0:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    cols = cursor.fetchall()
                    has_blob = any('BLOB' in str(col[2]).upper() for col in cols)
                    if has_blob or count > 0:
                        print(f"  {table_name}: {count} rows, has BLOB: {has_blob}")
            except Exception:
                pass
        
        # Check segment metadata for the VECTOR segment
        print("\n" + "-"*80)
        print("VECTOR segment metadata (where actual embeddings are stored):")
        print("-"*80)
        try:
            cursor.execute("""
                SELECT segment_id FROM segments 
                WHERE type LIKE '%vector%' OR scope = 'VECTOR'
            """)
            vector_segments = cursor.fetchall()
            if vector_segments:
                for (seg_id,) in vector_segments:
                    print(f"  Vector segment ID: {seg_id}")
                    cursor.execute("""
                        SELECT key, str_value, int_value, float_value 
                        FROM segment_metadata 
                        WHERE segment_id = ?
                    """, (seg_id,))
                    meta_rows = cursor.fetchall()
                    if meta_rows:
                        print(f"    Metadata entries: {len(meta_rows)}")
                        for key, str_val, int_val, float_val in meta_rows[:5]:
                            val = str_val or int_val or float_val
                            print(f"      {key}: {val}")
        except Exception as e:
            print(f"  Could not read vector segment metadata: {e}")
        
        # Check for segment files (where vectors might be stored)
        print("\n" + "-"*80)
        print("Files in vector store directory:")
        print("-"*80)
        all_files = list(store_path.iterdir())
        segment_files = [f for f in all_files if f.suffix == ".segment"]
        other_files = [f for f in all_files if f.suffix != ".segment" and f.is_file()]
        
        if segment_files:
            print(f"  Segment files ({len(segment_files)}):")
            for seg_file in segment_files[:5]:  # Show first 5
                size_mb = seg_file.stat().st_size / (1024 * 1024)
                print(f"    {seg_file.name} ({size_mb:.2f} MB)")
            if len(segment_files) > 5:
                print(f"    ... and {len(segment_files) - 5} more segment files")
        else:
            print("  No .segment files found")
        
        if other_files:
            print(f"\n  Other files ({len(other_files)}):")
            for f in other_files[:10]:
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name} ({size_kb:.2f} KB)")
        
        # Check if vectors are stored in-memory or in a different format
        print("\n" + "-"*80)
        print("Vector Storage Analysis:")
        print("-"*80)
        print("""
ChromaDB stores vectors using one of these methods:
1. In-memory (for small collections) - vectors loaded into memory on startup
2. Segment files (.segment) - binary files containing vector data
3. SQLite BLOB columns - vectors stored directly in database (less common)

Based on your setup:
- Embedding dimension: 768 (google/embeddinggemma-300m)
- Total embeddings: 34
- Estimated vector size: ~3 KB per embedding (768 floats × 4 bytes)
- Total vector data: ~100 KB

The vectors are likely stored:
- In-memory (most likely for this size)
- Or in the segment metadata/index structures
- ChromaDB's HNSW index keeps vectors accessible for similarity search

The actual 768-dimensional float arrays are managed internally by ChromaDB
and accessed through the segment_id lookup mechanism.
        """)
        
        conn.close()
        
        print("\n" + "="*80)
        print("How ChromaDB Stores Embeddings:")
        print("="*80)
        print("""
 ChromaDB uses a multi-table structure:
 1. 'embeddings' table: Maps your embedding_id to segment_id (METADATA ONLY)
 2. 'segments' table: Organizes vectors into segments for efficient retrieval    
 3. Vector data: Stored in HNSW index structure (NOT directly in SQLite tables)
 4. 'segment_id': Links embeddings to their storage location
 
 When you query ChromaDB:
 - You provide an embedding_id (e.g., 'AGENT_DEMO:test_sets:table:__table__')    
 - ChromaDB looks up the segment_id in the 'embeddings' table
 - It then retrieves the actual vector from the HNSW index (loaded in memory)
 - The vector is a high-dimensional array (768 floats for embeddinggemma-300m)
 
 WHERE ACTUAL VECTORS ARE STORED:
 ⚠️  The actual 768-dimensional float arrays are NOT stored as binary BLOBs 
     in the SQLite tables you can see.
 
 ✅ They are stored in ChromaDB's internal HNSW (Hierarchical Navigable Small World)
     index structure, which is:
     - Persisted in the SQLite database but in ChromaDB's proprietary format
     - Loaded into memory when the collection is opened
     - Optimized for fast similarity search (not human-readable)
     - The segment type 'urn:chroma:segment/vector/hnsw-local-persisted' indicates
       this is a persisted HNSW index
 
 The chroma.sqlite3 file contains:
 ✅ Metadata: embedding_id → segment_id mappings
 ✅ HNSW Index: The actual vectors in optimized format (not directly queryable)
 ✅ Metadata: Table/column info, schema hashes, etc.
 
 You CANNOT directly query the actual vector values from SQLite - ChromaDB's
 Python API handles this internally when you call collection.query() or .get().
        """)
        print("="*80 + "\n")
    
    client = chromadb.PersistentClient(path=str(store_path))
    collection = client.get_or_create_collection(
        name=vector_config.collection,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Verify cards vs embeddings
    graph_cards_db = cache_dir / "graph_cards.db"
    verify_cards_vs_embeddings(graph_cards_db, collection)
    
    # Get all embeddings
    results = collection.get(include=["metadatas", "documents", "embeddings"])
    
    total_count = len(results["ids"])
    print(f"\nTotal embeddings: {total_count}\n")
    
    if total_count == 0:
        print("No embeddings found in the vector store.")
        return
    
    # Show embedding dimension info
    embeddings_list = results.get("embeddings")
    if embeddings_list is not None and len(embeddings_list) > 0:
        embedding_dim = len(embeddings_list[0])
        print(f"Embedding dimension: {embedding_dim} (vectors are {embedding_dim}-dimensional)")
        print("Note: Actual vector values are stored as binary data in ChromaDB.\n")
    
    # Group by table
    by_table: dict[str, list[dict]] = {}
    for idx, vector_id in enumerate(results["ids"]):
        metadata = results["metadatas"][idx] if results["metadatas"] else {}
        document = results["documents"][idx] if results["documents"] else ""
        
        schema = metadata.get("schema", "unknown")
        table = metadata.get("table", "unknown")
        card_type = metadata.get("card_type", "unknown")
        identifier = metadata.get("identifier", "unknown")
        
        key = f"{schema}.{table}" if schema != "unknown" else table
        if key not in by_table:
            by_table[key] = []
        
        by_table[key].append({
            "vector_id": vector_id,
            "card_type": card_type,
            "identifier": identifier,
            "document": document,
            "metadata": metadata,
        })
    
    # Display grouped by table
    for table_key in sorted(by_table.keys()):
        cards = by_table[table_key]
        print(f"\n{'='*80}")
        print(f"Table: {table_key}")
        print(f"  Cards: {len(cards)}")
        print(f"{'='*80}")
        
        for card in cards:
            print(f"\n  Card Type: {card['card_type']}")
            print(f"  Identifier: {card['identifier']}")
            print(f"  Vector ID: {card['vector_id']}")
            print(f"  Document (first 200 chars):")
            doc_preview = card['document'][:200]
            if len(card['document']) > 200:
                doc_preview += "..."
            print(f"    {doc_preview}")
            print(f"  Full Metadata:")
            print(f"    {json.dumps(card['metadata'], indent=6)}")
    
    print(f"\n{'='*80}")
    print(f"Summary: {len(by_table)} table(s) with {total_count} total embedding(s)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    inspect_embeddings()

