#!/usr/bin/env python3
# services/ingest/ingest_report.py
#
# Vollständige Auswertung des Ingest-Ergebnisses.
#
# Aufruf:
#   python ingest_report.py                      # alle Dokumente
#   python ingest_report.py --doc <titel>        # ein Dokument
#   python ingest_report.py --export             # CSV-Export
#   python ingest_report.py --export --doc <titel>

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent))
from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Argumente
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="NDI Ingest-Report",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--doc",    type=str, default=None,
                        help="Nur dieses Dokument auswerten (Titelsuche)")
    parser.add_argument("--export", action="store_true",
                        help="Ergebnis als CSV exportieren")
    parser.add_argument("--out",    type=str, default=None,
                        help="CSV-Ausgabepfad (Standard: ingest_report_DATUM.csv)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Ausgabe-Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def sep(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def sub(title: str):
    print(f"\n  {'─'*55}")
    print(f"  {title}")
    print(f"  {'─'*55}")


def row(label: str, value, width: int = 35):
    print(f"  {label:<{width}} {value}")


def bar(value: int, total: int, width: int = 25) -> str:
    if total == 0:
        return "─" * width + "  –%"
    pct    = value / total
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled) + f"  {pct*100:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Datenbankabfragen
# ─────────────────────────────────────────────────────────────────────────────

async def get_connection():
    return await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
    )


async def report_jobs(conn, doc_filter: str = None):
    """Ingest-Job Übersicht."""
    sep("Ingest-Jobs")

    where = "WHERE filename ILIKE $1" if doc_filter else ""
    params = [f"%{doc_filter}%"] if doc_filter else []

    rows = await conn.fetch(f"""
        SELECT filename, status, doc_class, chunk_count,
               started_at, finished_at,
               EXTRACT(EPOCH FROM (finished_at - started_at))::INT AS dauer_sek
        FROM ingest_jobs
        {where}
        ORDER BY started_at DESC
        LIMIT 50
    """, *params)

    STATUS_ICON = {"done":"✅","error":"❌","queued":"⏳",
                   "parsing":"🔍","chunking":"✂️","embedding":"🧮","storing":"💾"}

    done_count  = sum(1 for r in rows if r["status"] == "done")
    error_count = sum(1 for r in rows if r["status"] == "error")

    row("Gesamte Jobs",    len(rows))
    row("Erfolgreich",     f"✅ {done_count}")
    row("Fehlerhaft",      f"❌ {error_count}")
    print()

    for r in rows:
        icon = STATUS_ICON.get(r["status"], "❓")
        dauer = f"{r['dauer_sek']}s" if r["dauer_sek"] else "–"
        print(f"  {icon} {r['filename']:<40} "
              f"Klasse={r['doc_class'] or '–'}  "
              f"Chunks={r['chunk_count'] or 0}  "
              f"Dauer={dauer}")
        if r["status"] == "error":
            err = await conn.fetchval(
                "SELECT error_message FROM ingest_jobs WHERE filename=$1 LIMIT 1",
                r["filename"]
            )
            if err:
                print(f"     ⚠  Fehler: {err[:80]}")


async def report_documents(conn, doc_filter: str = None):
    """Dokument-Übersicht."""
    sep("Dokumente in der Datenbank")

    where = "WHERE nd.title ILIKE $1" if doc_filter else ""
    params = [f"%{doc_filter}%"] if doc_filter else []

    rows = await conn.fetch(f"""
        SELECT
            nd.title,
            nd.source_type,
            nd.doc_id,
            nd.ingest_ts,
            COUNT(nc.id)                                        AS chunks,
            SUM(CASE WHEN nc.embedding   IS NOT NULL THEN 1 ELSE 0 END) AS mit_emb,
            SUM(CASE WHEN nc.abbrev_map  IS NOT NULL THEN 1 ELSE 0 END) AS mit_abbrev,
            SUM(CASE WHEN nc.content_original IS NOT NULL THEN 1 ELSE 0 END) AS mit_orig,
            MAX(nc.doc_class)                                   AS klasse,
            SUM(nc.token_count)                                 AS tokens_gesamt
        FROM norm_documents nd
        LEFT JOIN norm_chunks nc ON nc.doc_id = nd.id
        {where}
        GROUP BY nd.title, nd.source_type, nd.doc_id, nd.ingest_ts
        ORDER BY nd.ingest_ts DESC
    """, *params)

    if not rows:
        print("\n  Keine Dokumente gefunden.")
        return rows

    total_chunks = sum(r["chunks"] or 0 for r in rows)
    total_emb    = sum(r["mit_emb"] or 0 for r in rows)

    row("Dokumente gesamt",  len(rows))
    row("Chunks gesamt",     f"{total_chunks:,}")
    row("Mit Embedding",     f"{total_emb:,} / {total_chunks:,}")
    row("Embedding-Quote",
        f"{total_emb/total_chunks*100:.1f}%" if total_chunks else "–")

    for r in rows:
        chunks     = r["chunks"]   or 0
        mit_emb    = r["mit_emb"]  or 0
        mit_abbrev = r["mit_abbrev"] or 0
        tokens     = r["tokens_gesamt"] or 0
        emb_ok     = mit_emb == chunks

        print(f"\n  📄 {r['title']}")
        print(f"     Typ:      {r['source_type']}  |  Klasse: {r['klasse'] or '–'}")
        print(f"     Ingest:   {r['ingest_ts'].strftime('%d.%m.%Y %H:%M') if r['ingest_ts'] else '–'}")
        print(f"     Chunks:   {chunks}  |  Tokens: {tokens:,}")
        print(f"     Embedding: {bar(mit_emb, chunks)}  ({mit_emb}/{chunks})"
              + ("  ✅" if emb_ok else "  ⚠️"))
        print(f"     Abkürzungen aufgelöst in {mit_abbrev} Chunks")

    return rows


async def report_chunks(conn, doc_filter: str = None):
    """Chunk-Typen und Hierarchie-Analyse."""
    sep("Chunk-Analyse")

    where = "WHERE nd.title ILIKE $1" if doc_filter else ""
    params = [f"%{doc_filter}%"] if doc_filter else []

    # Chunk-Typen
    rows_type = await conn.fetch(f"""
        SELECT
            nc.doc_class,
            nc.chunk_type,
            COUNT(*)                         AS anzahl,
            AVG(nc.token_count)::INT         AS avg_tokens,
            AVG(nc.confidence_weight)::NUMERIC(3,2) AS avg_conf
        FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {where}
        GROUP BY nc.doc_class, nc.chunk_type
        ORDER BY nc.doc_class, anzahl DESC
    """, *params)

    sub("Chunk-Typen nach Klasse")
    current_class = None
    for r in rows_type:
        if r["doc_class"] != current_class:
            current_class = r["doc_class"]
            print(f"\n  Klasse {current_class}:")
        print(f"    {r['chunk_type']:<15} "
              f"{r['anzahl']:>5} Chunks  "
              f"ø {r['avg_tokens']:>4} Token  "
              f"Conf: {r['avg_conf']}")

    # Hierarchie-Verteilung
    rows_hier = await conn.fetch(f"""
        SELECT
            nc.hierarchy_level,
            COUNT(*)               AS anzahl,
            AVG(nc.token_count)::INT AS avg_tokens
        FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {where}
        GROUP BY nc.hierarchy_level
        ORDER BY nc.hierarchy_level
    """, *params)

    sub("Hierarchie-Verteilung")
    total_chunks = sum(r["anzahl"] for r in rows_hier) or 1
    for r in rows_hier:
        print(f"  Level {r['hierarchy_level']}:  "
              f"{bar(r['anzahl'], total_chunks, 20)}  "
              f"{r['anzahl']} Chunks  ø {r['avg_tokens']} Token")


async def report_abbreviations(conn, doc_filter: str = None):
    """Abkürzungs-Auswertung."""
    sep("Abkürzungsauflösung")

    where = "nd.title ILIKE $1 AND" if doc_filter else ""
    params = [f"%{doc_filter}%"] if doc_filter else []

    rows = await conn.fetch(f"""
        SELECT
            elem->>'abbrev'   AS abkürzung,
            elem->>'resolved' AS auflösung,
            elem->>'label'    AS label,
            COUNT(*)          AS in_chunks
        FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id,
             jsonb_array_elements(nc.abbrev_map) AS elem
        WHERE {where} nc.abbrev_map IS NOT NULL
        GROUP BY abkürzung, auflösung, label
        ORDER BY in_chunks DESC
    """, *params)

    if not rows:
        print("\n  Keine Abkürzungen aufgelöst.")
        return

    print(f"\n  {len(rows)} verschiedene Abkürzungen aufgelöst:\n")
    for r in rows:
        print(f"  [{r['label']:<12}] "
              f"{r['abkürzung']:<15} → "
              f"{r['auflösung'][:45]:<45}  "
              f"({r['in_chunks']} Chunks)")


async def report_quality(conn, doc_filter: str = None):
    """Qualitätskennzahlen."""
    sep("Qualitätskennzahlen")

    where = "WHERE nd.title ILIKE $1" if doc_filter else ""
    params = [f"%{doc_filter}%"] if doc_filter else []

    # Embedding-Vollständigkeit
    emb = await conn.fetchrow(f"""
        SELECT
            COUNT(*)                                              AS gesamt,
            SUM(CASE WHEN nc.embedding IS NOT NULL THEN 1 END)   AS mit_emb,
            SUM(CASE WHEN nc.embedding IS NULL THEN 1 END)       AS ohne_emb
        FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {where}
    """, *params)

    row("Chunks gesamt",       f"{emb['gesamt']:,}")
    ohne = emb["ohne_emb"] or 0
    row("Mit Embedding",
        f"{emb['mit_emb']:,}  ✅" if ohne == 0
        else f"{emb['mit_emb']:,}  ⚠️  ({ohne} fehlen)")

    # Token-Verteilung
    tok = await conn.fetchrow(f"""
        SELECT
            MIN(nc.token_count)   AS min_tok,
            MAX(nc.token_count)   AS max_tok,
            AVG(nc.token_count)::INT AS avg_tok,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nc.token_count)::INT AS median_tok
        FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {where}
    """, *params)

    print()
    row("Token min / max",     f"{tok['min_tok']} / {tok['max_tok']}")
    row("Token Durchschnitt",  f"{tok['avg_tok']}")
    row("Token Median",        f"{tok['median_tok']}")

    # Leere Chunks
    empty = await conn.fetchval(f"""
        SELECT COUNT(*) FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {where} {'AND' if where else 'WHERE'} LENGTH(TRIM(nc.content)) < 10
    """, *params)
    row("Sehr kurze Chunks (<10 Z.)", f"{empty}"
        + ("  ✅" if empty == 0 else "  ⚠️  prüfen"))

    # Chunks ohne norm_reference (nur Klasse A)
    no_ref = await conn.fetchval(f"""
        SELECT COUNT(*) FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {where} {'AND' if where else 'WHERE'}
        nc.doc_class = 'A' AND nc.norm_reference IS NULL
    """, *params)
    row("Klasse A ohne norm_reference", f"{no_ref}"
        + ("  ✅" if no_ref == 0 else "  ⚠️  prüfen"))


async def export_csv(conn, doc_filter: str = None, out_path: str = None):
    """CSV-Export aller Chunks."""
    sep("CSV-Export")

    where = "WHERE nd.title ILIKE $1" if doc_filter else ""
    params = [f"%{doc_filter}%"] if doc_filter else []

    rows = await conn.fetch(f"""
        SELECT
            nd.title,
            nd.source_type,
            nc.doc_class,
            nc.chunk_type,
            nc.hierarchy_level,
            nc.norm_reference,
            nc.section_path,
            nc.confidence_weight,
            nc.token_count,
            CASE WHEN nc.embedding IS NOT NULL THEN 'ja' ELSE 'nein' END AS embedding,
            CASE WHEN nc.abbrev_map IS NOT NULL THEN 'ja' ELSE 'nein' END AS abbrev_aufgelöst,
            LEFT(nc.content, 300) AS text_vorschau,
            LEFT(nc.content_original, 300) AS text_original
        FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {where}
        ORDER BY nd.title, nc.hierarchy_level, nc.id
    """, *params)

    if not out_path:
        ts       = datetime.now().strftime("%m%d%H%M%S")
        out_path = f"ingest_report_{ts}.csv"

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "titel", "typ", "klasse", "chunk_typ", "level",
                "norm_referenz", "section_path", "confidence",
                "tokens", "embedding", "abbrev_aufgelöst",
                "text_vorschau", "text_original",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "titel":           r["title"],
                "typ":             r["source_type"],
                "klasse":          r["doc_class"],
                "chunk_typ":       r["chunk_type"],
                "level":           r["hierarchy_level"],
                "norm_referenz":   r["norm_reference"] or "",
                "section_path":    r["section_path"] or "",
                "confidence":      r["confidence_weight"],
                "tokens":          r["token_count"],
                "embedding":       r["embedding"],
                "abbrev_aufgelöst": r["abbrev_aufgelöst"],
                "text_vorschau":   r["text_vorschau"] or "",
                "text_original":   r["text_original"] or "",
            })

    print(f"\n  ✅ {len(rows)} Chunks exportiert → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Hauptprogramm
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    print(f"\n{'='*65}")
    print(f"  NDI Ingest-Report")
    print(f"  {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    if args.doc:
        print(f"  Filter: '{args.doc}'")
    print(f"{'='*65}")

    conn = await get_connection()

    try:
        await report_jobs(conn,          args.doc)
        await report_documents(conn,     args.doc)
        await report_chunks(conn,        args.doc)
        await report_abbreviations(conn, args.doc)
        await report_quality(conn,       args.doc)

        if args.export:
            await export_csv(conn, args.doc, args.out)

    finally:
        await conn.close()

    print(f"\n{'='*65}")
    print(f"  Report abgeschlossen")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
