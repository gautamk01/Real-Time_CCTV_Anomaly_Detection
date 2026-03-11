"""RAG Bootstrap — Seed the knowledge base from existing alerts.

Run once to import all existing alert investigations into the RAG
knowledge base for immediate retrieval during new investigations.

Usage:
    python -m rag.bootstrap
    python -m rag.bootstrap --alerts-dir alerts --db-dir rag_db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Seed RAG knowledge base from existing alert data"
    )
    parser.add_argument(
        "--alerts-dir", type=str, default="alerts",
        help="Path to alerts directory (default: alerts)",
    )
    parser.add_argument(
        "--db-dir", type=str, default="rag_db",
        help="Path for ChromaDB storage (default: rag_db)",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear existing knowledge base before ingesting",
    )
    args = parser.parse_args()

    from rag.knowledge_base import KnowledgeBase

    print("\n" + "=" * 60)
    print("  RAG KNOWLEDGE BASE BOOTSTRAP")
    print("=" * 60)

    kb = KnowledgeBase(
        persist_dir=args.db_dir,
        embedding_model=args.embedding_model,
    )

    if args.clear:
        print(f"\n🗑️  Clearing existing knowledge base...")
        kb.clear()
        print(f"   Done. Count: {kb.count()}")

    print(f"\n📥 Ingesting alerts from: {args.alerts_dir}")
    count_before = kb.count()
    ingested = kb.ingest_existing_alerts(Path(args.alerts_dir))
    count_after = kb.count()

    print(f"\n{'=' * 60}")
    print(f"  BOOTSTRAP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Cases before: {count_before}")
    print(f"  New ingested: {ingested}")
    print(f"  Total cases:  {count_after}")
    print(f"  DB location:  {args.db_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
