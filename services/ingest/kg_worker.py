#!/usr/bin/env python3
# services/ingest/kg_worker.py
#
# CLI für den Knowledge Graph Export.
#
# Aufruf:
#   python kg_worker.py --export               # inkrementell
#   python kg_worker.py --export --full        # kompletter Re-Export
#   python kg_worker.py --export --doc-id <id> # ein Dokument
#   python kg_worker.py --ping                 # Fuseki-Verbindung prüfen
#   python kg_worker.py --stats                # Export-Statistik
#   python kg_worker.py --query-svo "Meldebehörde"
#   python kg_worker.py --query-entities GESETZ

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent))
from app.core.config import settings
from app.services.knowledge_graph import KnowledgeGraphService


def parse_args():
    p = argparse.ArgumentParser(description="NDI Knowledge Graph Worker")
    p.add_argument("--export",          action="store_true")
    p.add_argument("--full",            action="store_true",
                   help="Kompletter Re-Export (nicht inkrementell)")
    p.add_argument("--doc-id",          type=str, default=None)
    p.add_argument("--ping",            action="store_true")
    p.add_argument("--stats",           action="store_true")
    p.add_argument("--query-svo",       type=str, metavar="SUBJEKT")
    p.add_argument("--query-entities",  type=str, metavar="LABEL")
    return p.parse_args()


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():
    args = parse_args()
    kg   = KnowledgeGraphService()

    # ── Ping ──────────────────────────────────────────────────────────────────
    if args.ping:
        sep("Fuseki Verbindungstest")
        ok = await kg.ping()
        if ok:
            print(f"\n  ✅ Fuseki erreichbar: {kg.sparql_endpoint}")
        else:
            print(f"\n  ❌ Fuseki NICHT erreichbar!")
            print(f"     Endpoint: {kg.sparql_endpoint}")
            print(f"     → docker compose up -d")
        return

    # ── Export ────────────────────────────────────────────────────────────────
    if args.export:
        sep("Knowledge Graph Export")
        print(f"\n  Modus: {'Vollständig' if args.full else 'Inkrementell'}")
        if args.doc_id:
            print(f"  Dokument: {args.doc_id}")

        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.postgres_db,
        )
        try:
            result = await kg.export_all(
                conn,
                doc_id=args.doc_id,
                incremental=not args.full,
            )
            print(f"\n  Tripel SVO:    {result.triples_svo:>8,}")
            print(f"  Tripel NER:    {result.triples_ner:>8,}")
            print(f"  Tripel Normen: {result.triples_norms:>8,}")
            print(f"  ─────────────────────")
            print(f"  Gesamt:        {result.total_triples:>8,}")
            if result.errors:
                print(f"\n  ⚠️  {len(result.errors)} Fehler:")
                for e in result.errors:
                    print(f"     {e}")
            else:
                print(f"\n  ✅ Export erfolgreich")
        finally:
            await conn.close()
        return

    # ── Statistik ─────────────────────────────────────────────────────────────
    if args.stats:
        sep("KG-Export-Statistik")
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.postgres_db,
        )
        try:
            rows = await conn.fetch(
                "SELECT * FROM kg_sync_stats ORDER BY title"
            )
            if not rows:
                print("\n  Noch keine Exporte durchgeführt.")
            else:
                print(f"\n  {'Dokument':<35} {'Export':>8} {'Aussteh':>8} {'Tripel':>8}")
                print(f"  {'─'*65}")
                for r in rows:
                    print(f"  {str(r['title'])[:34]:<35} "
                          f"{r['exportiert'] or 0:>8} "
                          f"{r['ausstehend'] or 0:>8} "
                          f"{r['tripel_gesamt'] or 0:>8,}")
        finally:
            await conn.close()
        return

    # ── SPARQL-Abfragen ───────────────────────────────────────────────────────
    if args.query_svo:
        sep(f"SVOs für: {args.query_svo}")
        results = await kg.query_svo_by_subject(args.query_svo)
        if not results:
            print("\n  Keine Ergebnisse.")
        else:
            print(f"\n  {'Subjekt':<20} {'Prädikat':<20} {'Objekt':<20} Typ")
            print(f"  {'─'*75}")
            for r in results:
                print(f"  {str(r.get('subjekt',''))[:19]:<20} "
                      f"{str(r.get('praedikat',''))[:19]:<20} "
                      f"{str(r.get('objekt',''))[:19]:<20} "
                      f"{r.get('normtyp','')}")
        return

    if args.query_entities:
        sep(f"Entitäten Label: {args.query_entities}")
        results = await kg.query_entities_by_label(args.query_entities)
        if not results:
            print("\n  Keine Ergebnisse.")
        else:
            print(f"\n  {'Text':<35} {'Häufigkeit':>12}")
            print(f"  {'─'*50}")
            for r in results:
                print(f"  {str(r.get('text',''))[:34]:<35} "
                      f"{r.get('haeufigkeit','0'):>12}")
        return

    # Kein Argument
    print("Aufruf: python kg_worker.py --help")


if __name__ == "__main__":
    asyncio.run(main())
