#!/usr/bin/env python3
# services/ingest/ingest_cli.py
#
# Kommandozeilen-Interface für die NDI Ingest-Pipeline.
# Verwendet dieselbe IngestService-Logik wie der FastAPI-Endpoint.
#
# Aufruf:
#   python ingest_cli.py --pdf ../../docu/Konzepte/Hundegesetz.pdf
#   python ingest_cli.py --pdf ../../docu/Konzepte/Hundegesetz.pdf --class A
#   python ingest_cli.py --pdf ../../docu/Konzepte/Hundegesetz.pdf --limit 20
#   python ingest_cli.py --pdf ../../docu/Konzepte/MNR-RAG-Architektur_v1.1.pdf \
#                        --source-type fachkonzept --title "MNR Architektur" --version "1.1"
#   python ingest_cli.py --jobs          # letzte Jobs anzeigen
#   python ingest_cli.py --status <job_id>

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from datetime import datetime

import asyncpg

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.services.ingest_service import IngestService


# ─────────────────────────────────────────────────────────────────────────────
# Argument-Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="NDI Ingest-CLI – Dokumente direkt in die Pipeline einspeisen",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Hauptaktion
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--pdf", "--file", "-f",
        dest="filepath",
        type=str,
        metavar="PFAD",
        help="Dokument-Pfad (PDF, DOCX, HTML, TXT, RTF)",
    )
    action.add_argument(
        "--jobs",
        action="store_true",
        help="Letzte Ingest-Jobs anzeigen",
    )
    action.add_argument(
        "--status",
        type=str,
        metavar="JOB_ID",
        help="Status eines bestimmten Jobs abfragen",
    )

    # Dokument-Metadaten
    parser.add_argument(
        "--source-type", "-t",
        default="gesetz",
        choices=["gesetz","verordnung","standard","fachkonzept","leitfaden","lastenheft","auslegung"],
        help="Dokumenttyp (Standard: gesetz)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Dokumenttitel (Standard: Dateiname ohne Erweiterung)",
    )
    parser.add_argument(
        "--jurisdiction",
        type=str,
        default=None,
        metavar="KÜRZEL",
        help="Jurisdiktion, z.B. BW, BY, NDS (optional)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Dokumentversion, z.B. 1.1 (optional)",
    )
    parser.add_argument(
        "--norm-reference",
        type=str,
        default=None,
        help="Normreferenz, z.B. 'NHundG' (optional)",
    )

    # Pipeline-Steuerung
    parser.add_argument(
        "--class", "-c",
        dest="force_class",
        choices=["A", "B", "C"],
        default=None,
        help="Dokumentklasse erzwingen (Standard: automatische Erkennung)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        metavar="N",
        help="Nur N Chunks verarbeiten (0 = alle, für Tests)",
    )
    parser.add_argument(
        "--jobs-limit",
        type=int,
        default=20,
        help="Anzahl Jobs bei --jobs (Standard: 20)",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metadata-Klasse (kompatibel mit IngestService)
# ─────────────────────────────────────────────────────────────────────────────

class CliMetadata:
    def __init__(self, args, filepath: Path):
        self.source_type    = args.source_type
        self.title          = args.title or filepath.stem
        self.jurisdiction   = args.jurisdiction
        self.valid_from     = None
        self.valid_to       = None
        self.norm_reference = args.norm_reference
        self.version        = args.version
        self.language       = "de"
        self.register_scope = None
        # Priorität 1: source_type explizit per CLI übergeben?
        # True wenn der Nutzer --source-type angegeben hat
        # (nicht den Default "gesetz")
        self._source_type_explicit  = args.source_type != "gesetz"
        self._source_type_from_yaml = False


# ─────────────────────────────────────────────────────────────────────────────
# Aktionen
# ─────────────────────────────────────────────────────────────────────────────

async def run_ingest(args):
    """Dokument durch die Pipeline schicken."""
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"\n  Fehler: Datei nicht gefunden: {filepath}")
        sys.exit(1)

    doc_id   = str(uuid.uuid4())
    job_id   = str(uuid.uuid4())
    metadata = CliMetadata(args, filepath)

    print(f"\n{'='*65}")
    print(f"  NDI Ingest-CLI")
    print(f"{'='*65}")
    print(f"  Datei:       {filepath.name}  ({filepath.stat().st_size/1024:.1f} KB)")
    print(f"  Typ:         {metadata.source_type}")
    print(f"  Titel:       {metadata.title}")
    print(f"  Klasse:      {args.force_class or 'automatisch'}")
    print(f"  Chunk-Limit: {args.limit if args.limit else 'alle'}")
    print(f"  doc_id:      {doc_id}")
    print(f"  job_id:      {job_id}")
    print(f"{'─'*65}")

    service = IngestService()

    try:
        file_content = filepath.read_bytes()

        result = await service.run_pipeline(
            doc_id=doc_id,
            job_id=job_id,
            file_content=file_content,
            filename=filepath.name,
            metadata=metadata,
            doc_class_override=args.force_class,
            chunk_limit=args.limit if args.limit > 0 else None,
        )

        print(f"\n{'─'*65}")
        if result.status == "done":
            print(f"  ✅ Pipeline abgeschlossen")
            print(f"     Dokumentklasse: {result.doc_class}")
            print(f"     Chunks:         {result.chunk_count}")
            print(f"     doc_id:         {result.doc_id}")
            print(f"     job_id:         {result.job_id}")
        else:
            print(f"  ❌ Pipeline fehlgeschlagen")
            print(f"     Fehler: {result.error}")

    finally:
        await service.close()

    print(f"{'='*65}\n")


async def show_jobs(limit: int):
    """Letzte Ingest-Jobs anzeigen."""
    conn = await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )

    rows = await conn.fetch(
        """
        SELECT job_id, filename, status, doc_class,
               chunk_count, started_at, finished_at
        FROM ingest_jobs
        ORDER BY started_at DESC
        LIMIT $1
        """,
        limit,
    )
    await conn.close()

    print(f"\n{'='*65}")
    print(f"  Letzte {limit} Ingest-Jobs")
    print(f"{'='*65}")

    STATUS_ICONS = {
        "done":      "✅",
        "error":     "❌",
        "queued":    "⏳",
        "parsing":   "🔍",
        "chunking":  "✂️ ",
        "embedding": "🧮",
        "storing":   "💾",
    }

    for row in rows:
        icon     = STATUS_ICONS.get(row["status"], "❓")
        duration = ""
        if row["started_at"] and row["finished_at"]:
            secs = (row["finished_at"] - row["started_at"]).total_seconds()
            duration = f"  {secs:.0f}s"

        print(f"\n  {icon} {row['status']:<10} {row['filename'][:40]:<40}{duration}")
        print(f"     job_id:  {row['job_id']}")
        print(f"     Klasse:  {row['doc_class'] or '–'}  |  "
              f"Chunks: {row['chunk_count'] or '–'}  |  "
              f"Start: {row['started_at'].strftime('%d.%m.%Y %H:%M') if row['started_at'] else '–'}")

    if not rows:
        print("\n  Keine Jobs gefunden.")

    print(f"\n{'='*65}\n")


async def show_status(job_id: str):
    """Status eines einzelnen Jobs anzeigen."""
    service = IngestService()
    try:
        status = await service.get_job_status(job_id)
        if not status:
            print(f"\n  Job '{job_id}' nicht gefunden.")
            return

        STATUS_ICONS = {
            "done": "✅", "error": "❌", "queued": "⏳",
            "parsing": "🔍", "chunking": "✂️", "embedding": "🧮", "storing": "💾",
        }
        icon = STATUS_ICONS.get(status["status"], "❓")

        print(f"\n{'='*65}")
        print(f"  Job-Status")
        print(f"{'='*65}")
        print(f"  {icon} Status:    {status['status']}")
        print(f"  Datei:         {status['filename']}")
        print(f"  job_id:        {status['job_id']}")
        print(f"  doc_id:        {status['doc_id']}")
        print(f"  Dokumentkl.:  {status['doc_class'] or '–'}")
        print(f"  Chunks:        {status['chunk_count'] or '–'}")
        print(f"  Gestartet:     {status['started_at'] or '–'}")
        print(f"  Abgeschlossen: {status['finished_at'] or 'läuft noch'}")
        if status["error_message"]:
            print(f"  Fehler:        {status['error_message']}")
        print(f"{'='*65}\n")
    finally:
        await service.close()


# ─────────────────────────────────────────────────────────────────────────────
# Hauptprogramm
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    if args.jobs:
        await show_jobs(args.jobs_limit)
    elif args.status:
        await show_status(args.status)
    else:
        await run_ingest(args)


if __name__ == "__main__":
    asyncio.run(main())
