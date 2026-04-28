#!/usr/bin/env python3
# services/ingest/test_storage.py
#
# Testet storage.py und die vollständige M1-Pipeline:
# Parsing → Chunking → Embedding → PostgreSQL + MinIO
#
# Aufruf:
#   python test_storage.py --pdf ../../docu/Konzepte/Hundegesetz.pdf
#   python test_storage.py --pdf ../../docu/Konzepte/Hundegesetz.pdf --limit 10
#   python test_storage.py --verify  # nur DB-Inhalt prüfen ohne neuen Ingest

import argparse
import asyncio
import sys
import uuid
import time
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.services.parser import TikaParser
from app.services.chunker import ChunkingRouter
from app.services.embedder import Embedder
from app.services.storage import DocumentStorage


def parse_args():
    parser = argparse.ArgumentParser(description="NDI Storage + Pipeline Test")
    parser.add_argument("--pdf",    type=str, default=None,  help="PDF-Datei für Pipeline-Test")
    parser.add_argument("--limit",  type=int, default=0,     help="Max. Chunks (0=alle)")
    parser.add_argument("--verify", action="store_true",     help="Nur DB-Inhalt prüfen")
    parser.add_argument("--class",  dest="force_class", choices=["A","B","C"],
                        default=None, help="Dokumentklasse erzwingen")
    return parser.parse_args()


def print_separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


async def verify_database():
    """Zeigt den aktuellen Inhalt der Datenbank."""
    print_separator("Datenbank-Überblick")

    conn = await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )

    # Dokumente
    docs = await conn.fetch("""
        SELECT doc_id, source_type, title, ingest_ts
        FROM norm_documents
        ORDER BY ingest_ts DESC
        LIMIT 10
    """)
    print(f"\n  norm_documents: {len(docs)} Einträge (letzte 10)")
    print(f"  {'─'*55}")
    for doc in docs:
        print(f"  {str(doc['doc_id'])[:8]}... | {doc['source_type']:<12} | {doc['title'][:40]}")

    # Chunk-Statistik
    stats = await conn.fetch("""
        SELECT
            nd.title,
            nc.doc_class,
            nc.chunk_type,
            COUNT(*) as cnt,
            AVG(nc.confidence_weight)::NUMERIC(3,2) as avg_confidence,
            SUM(CASE WHEN nc.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
        FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        GROUP BY nd.title, nc.doc_class, nc.chunk_type
        ORDER BY nd.title, nc.doc_class, cnt DESC
    """)

    print(f"\n  norm_chunks – Statistik nach Dokument/Klasse/Typ:")
    print(f"  {'─'*55}")
    current_title = None
    for row in stats:
        if row["title"] != current_title:
            current_title = row["title"]
            print(f"\n  📄 {current_title[:50]}")
        print(f"     Klasse {row['doc_class']} | {row['chunk_type']:<15} | "
              f"{row['cnt']:>4} Chunks | "
              f"Confidence: {row['avg_confidence']} | "
              f"Embeddings: {row['with_embedding']}/{row['cnt']}")

    # Gesamt
    total = await conn.fetchrow("SELECT COUNT(*) as cnt FROM norm_chunks")
    with_emb = await conn.fetchrow(
        "SELECT COUNT(*) as cnt FROM norm_chunks WHERE embedding IS NOT NULL"
    )
    print(f"\n  Gesamt: {total['cnt']} Chunks | "
          f"Mit Embedding: {with_emb['cnt']} | "
          f"Ohne: {total['cnt'] - with_emb['cnt']}")

    await conn.close()


async def run_pipeline(args):
    """Führt die vollständige Pipeline für ein PDF-Dokument aus."""
    filepath = Path(args.pdf)
    if not filepath.exists():
        print(f"\n  Fehler: Datei nicht gefunden: {filepath}")
        sys.exit(1)

    doc_id = str(uuid.uuid4())

    print_separator(f"M1 Pipeline: {filepath.name}")
    print(f"  doc_id: {doc_id}")

    storage  = DocumentStorage()
    parser   = TikaParser()
    chunker  = ChunkingRouter()
    embedder = Embedder()

    class FakeMeta:
        source_type   = "gesetz"
        title         = filepath.stem
        jurisdiction  = None
        valid_from    = None
        valid_to      = None
        norm_reference = None
        version       = "1.0"
        language      = "de"
        register_scope = None

    t_total = time.time()

    # ── Schritt 1: MinIO ──────────────────────────────────────────────────────
    print(f"\n  [1/5] Rohdokument → MinIO...")
    t = time.time()
    content    = filepath.read_bytes()
    minio_path = await storage.store_raw_document(
        doc_id=doc_id, filename=filepath.name, content=content
    )
    print(f"        ✓ {minio_path}  ({time.time()-t:.2f}s)")

    # ── Schritt 2: Dokument-Datensatz ─────────────────────────────────────────
    print(f"\n  [2/5] Metadaten → PostgreSQL (norm_documents)...")
    t = time.time()
    internal_id = await storage.create_document_record(
        doc_id=doc_id, metadata=FakeMeta(), minio_path=minio_path
    )
    print(f"        ✓ internal_id={internal_id[:8]}...  ({time.time()-t:.2f}s)")

    # ── Schritt 3: Parsing ────────────────────────────────────────────────────
    print(f"\n  [3/5] Tika-Parsing...")
    t = time.time()
    result = await parser.parse(content=content, filename=filepath.name)
    print(f"        ✓ Klasse={result.doc_class_hint} | "
          f"{result.char_count:,} Zeichen | "
          f"{result.word_count:,} Wörter  ({time.time()-t:.2f}s)")

    # ── Schritt 4: Chunking ───────────────────────────────────────────────────
    print(f"\n  [4/5] Chunking (Klasse {args.force_class or result.doc_class_hint})...")
    t = time.time()
    chunks = chunker.route_and_chunk(
        text=result.text, structure=result.structure,
        doc_id=doc_id, metadata=FakeMeta(),
        doc_class_override=args.force_class,
    )
    limit   = args.limit if args.limit > 0 else len(chunks)
    subset  = chunks[:limit]
    print(f"        ✓ {len(chunks)} Chunks gesamt | verarbeite {len(subset)}  ({time.time()-t:.2f}s)")

    # ── Schritt 5: Embedding ──────────────────────────────────────────────────
    print(f"\n  [5/5] Embeddings berechnen ({embedder.active_name})...")
    t = time.time()
    embedded = await embedder.embed_chunks(subset)
    print(f"        ✓ {len(embedded)} Embeddings | "
          f"dim={len(embedded[0].embedding) if embedded[0].embedding else '–'}  "
          f"({time.time()-t:.2f}s)")

    # ── Schritt 6: PostgreSQL speichern ───────────────────────────────────────
    print(f"\n  [6/6] Chunks + Embeddings → PostgreSQL (norm_chunks)...")
    t = time.time()
    inserted = await storage.store_chunks(embedded)
    print(f"        ✓ {inserted} Chunks gespeichert  ({time.time()-t:.2f}s)")

    elapsed = time.time() - t_total
    print(f"\n  {'─'*55}")
    print(f"  ✅ Pipeline abgeschlossen in {elapsed:.1f}s")
    print(f"     Dokument: {filepath.name}")
    print(f"     doc_id:   {doc_id}")
    print(f"     Chunks:   {inserted} gespeichert")

    await storage.close()
    return doc_id


async def main():
    args = parse_args()

    if args.verify:
        await verify_database()
        return

    if args.pdf:
        doc_id = await run_pipeline(args)
        # Danach DB-Inhalt anzeigen
        await verify_database()
    else:
        print("\n  Verwendung:")
        print("  python test_storage.py --pdf ../../docu/Konzepte/Hundegesetz.pdf")
        print("  python test_storage.py --pdf ../../docu/Konzepte/Hundegesetz.pdf --limit 20")
        print("  python test_storage.py --verify")

    print(f"\n{'='*65}")
    print("  Storage-Test abgeschlossen")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
