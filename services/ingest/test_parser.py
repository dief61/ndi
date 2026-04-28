#!/usr/bin/env python3
# services/ingest/test_parser.py
#
# Schnelltest für den TikaParser – ohne FastAPI, direkt ausführbar.
# Aufruf: python test_parser.py <pfad-zur-datei>
#
# Beispiel:
#   python test_parser.py ../../docs/Konzepte/MNR-RAG-Architektur\ v1.1.pdf

import asyncio
import sys
from pathlib import Path

# Projektpfad ergänzen damit app.* importierbar ist
sys.path.insert(0, str(Path(__file__).parent))

from app.services.parser import TikaParser


async def main():
    if len(sys.argv) < 2:
        print("Verwendung: python test_parser.py <datei>")
        print("Beispiel:   python test_parser.py ../../docs/Konzepte/MNR-RAG-Architektur v1.1.pdf")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Datei nicht gefunden: {filepath}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  NDI Tika-Parser Test")
    print(f"{'='*60}")
    print(f"  Datei:  {filepath.name}")
    print(f"  Größe:  {filepath.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")

    content = filepath.read_bytes()

    parser = TikaParser()
    print("Sende Dokument an Tika...")

    result = await parser.parse(content=content, filename=filepath.name)

    print(f"\n── Ergebnis ─────────────────────────────────────────────")
    print(f"  Content-Type:     {result.content_type}")
    print(f"  Dokumentklasse:   Klasse {result.doc_class_hint}")
    print(f"  Zeichen:          {result.char_count:,}")
    print(f"  Wörter:           {result.word_count:,}")
    print(f"\n── Struktur ─────────────────────────────────────────────")
    print(f"  §-Marker:         {result.structure.has_paragraph_markers} "
          f"({result.structure.paragraph_count} gefunden)")
    print(f"  Heading-Hierarchie: {result.structure.has_heading_hierarchy} "
          f"({result.structure.heading_count} gefunden)")
    print(f"  Inhaltsverzeichnis: {result.structure.has_toc}")

    if result.structure.headings:
        print(f"\n── Erste Überschriften ──────────────────────────────────")
        for h in result.structure.headings[:10]:
            print(f"  {h.strip()}")

    print(f"\n── Textvorschau (erste 500 Zeichen) ─────────────────────")
    print(result.text[:500])
    print(f"\n── Tika-Metadaten ───────────────────────────────────────")
    for k, v in list(result.metadata.items())[:10]:
        print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
