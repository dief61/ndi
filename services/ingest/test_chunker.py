#!/usr/bin/env python3
# services/ingest/test_chunker.py
#
# Aufruf:
#   python test_chunker.py [PDF] [--limit N] [--output DATEI] [--class A|B|C]
#                          [--config YAML] [--no-synthetic]
#
# Beispiele:
#   python test_chunker.py ../../docu/Konzepte/Hundegesetz.pdf
#   python test_chunker.py ../../docu/Konzepte/Hundegesetz.pdf --class A
#   python test_chunker.py ../../docu/Konzepte/Hundegesetz.pdf --limit 10 --output chunks.txt
#   python test_chunker.py ../../docu/Konzepte/Hundegesetz.pdf --no-synthetic
#   python test_chunker.py ../../docu/Konzepte/Hundegesetz.pdf --config test_chunker_config.yaml

import argparse
import asyncio
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.services.parser import TikaParser
from app.services.chunker import ChunkingRouter

# ── YAML optional ─────────────────────────────────────────────────────────────
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ── Synthetische Testdaten ────────────────────────────────────────────────────

KLASSE_A_TEXT = """
§ 1 Anwendungsbereich
Dieses Gesetz regelt die Erhebung, Speicherung und Übermittlung von Meldedaten.
Es gilt für alle Personen, die eine Wohnung in Deutschland beziehen.

(1) Die Meldebehörde ist verpflichtet, die Meldedaten der betroffenen Person zu speichern.
(2) Die Daten dürfen nicht ohne Einwilligung an Dritte übermittelt werden.
(3) Im Sinne dieses Gesetzes ist eine betroffene Person jede natürliche Person,
die eine Wohnung anmeldet.

§ 2 Meldepflicht
Jede Person, die eine Wohnung bezieht, muss sich innerhalb von zwei Wochen anmelden.
Die Anmeldung hat bei der zuständigen Meldebehörde zu erfolgen.

(1) Die Anmeldepflicht obliegt der einziehenden Person.
(2) Bei Minderjährigen sind die Erziehungsberechtigten zuständig.
(3) Die Meldebehörde kann Ausnahmen gewähren, sofern nicht öffentliche Interessen
entgegenstehen.

§ 3 Meldedaten
Die Meldebehörde speichert folgende Grunddaten:
Als Grunddaten sind zu speichern: Familienname, Vornamen, Geburtsdatum, Geburtsort.

(1) Familienname und Vornamen sind vollständig zu erfassen.
(2) Das Geburtsdatum muss im Format JJJJ-MM-TT gespeichert werden.
(3) Im Sinne dieses Gesetzes ist der Geburtsort die Gemeinde, in der die Person geboren wurde.
"""

KLASSE_C_TEXT = """
Häufig gestellte Fragen zur Ummeldung. Wenn Sie umziehen, müssen Sie sich ummelden.
Die Ummeldung ist innerhalb von zwei Wochen nach dem Einzug vorzunehmen. Sie können
die Ummeldung persönlich oder in manchen Kommunen auch online durchführen. Bringen Sie
bitte Ihren Personalausweis und eine Wohnungsgeberbestätigung mit. Diese Bestätigung
erhalten Sie von Ihrem Vermieter. Ohne diese Bestätigung kann die Ummeldung nicht
vorgenommen werden. Bei Fragen wenden Sie sich bitte an Ihre zuständige Meldebehörde.
Die Öffnungszeiten finden Sie auf der Website Ihrer Gemeinde. Eine Ummeldung ist
gebührenfrei. Es fallen keine Kosten für Sie an.
"""


# ── Konfiguration laden ───────────────────────────────────────────────────────

def load_config(config_path: str | None) -> dict:
    """Lädt YAML-Konfigurationsdatei. Gibt leeres Dict zurück wenn nicht vorhanden."""
    defaults = {
        "output": {"limit": 0, "file": None, "text_preview_length": 200},
        "document": {"force_class": None},
        "synthetic_tests": {"enabled": True},
        "chunking": {
            "class_a": {"token_limit_parent": 1024, "token_limit_child": 256, "token_limit_satz": 128},
            "class_b": {"token_limit_parent": 128, "token_limit_child": 512, "overlap_ratio": 0.15},
            "class_c": {"token_min": 50, "token_max": 384, "overlap_ratio": 0.20},
        },
    }

    if not config_path:
        # Automatisch test_chunker_config.yaml im gleichen Verzeichnis suchen
        auto_path = Path(__file__).parent / "test_chunker_config.yaml"
        if auto_path.exists():
            config_path = str(auto_path)

    if config_path and YAML_AVAILABLE:
        path = Path(config_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            # Tief mergen
            for section, values in loaded.items():
                if section in defaults and isinstance(values, dict):
                    defaults[section].update(values)
                else:
                    defaults[section] = values
            print(f"  Konfiguration geladen: {path}")
        else:
            print(f"  Warnung: Konfigurationsdatei nicht gefunden: {path}")
    elif config_path and not YAML_AVAILABLE:
        print("  Warnung: PyYAML nicht installiert. pip install pyyaml")

    return defaults


# ── Argument-Parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="NDI Chunker-Test",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Pfad zur PDF-Datei",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Maximale Anzahl Chunks ausgeben (0 = alle)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="DATEI",
        help="Ausgabe zusätzlich in Datei schreiben",
    )
    parser.add_argument(
        "--class", "-c",
        dest="force_class",
        choices=["A", "B", "C"],
        default=None,
        metavar="KLASSE",
        help=(
            "Dokumentklasse erzwingen (überschreibt automatische Erkennung).\n"
            "  -c A  → Klasse A (§-basiertes Chunking)\n"
            "  -c B  → Klasse B (Kapitel-basiertes Chunking)\n"
            "  -c C  → Klasse C (Semantisches Sliding-Window)"
        ),
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        default=False,
        help="Synthetische Klasse-A- und Klasse-C-Tests überspringen",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="YAML",
        help="Pfad zur YAML-Konfigurationsdatei (Standard: test_chunker_config.yaml)",
    )
    return parser.parse_args()


# ── Ausgabe-Klasse ────────────────────────────────────────────────────────────

class Output:
    """Schreibt gleichzeitig auf Konsole und optional in eine Datei."""

    def __init__(self, filepath: str | None, text_preview_length: int = 200):
        self.file = None
        self.text_preview_length = text_preview_length
        if filepath:
            self.file = open(filepath, "w", encoding="utf-8")
            self.file.write(
                f"NDI Chunker-Test – Ausgabe\n"
                f"Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*65}\n\n"
            )

    def write(self, text: str = ""):
        print(text)
        if self.file:
            self.file.write(text + "\n")

    def close(self):
        if self.file:
            self.file.close()
            print(f"\n  → Ausgabe gespeichert: {self.file.name}")


# ── Ausgabe-Hilfsfunktionen ───────────────────────────────────────────────────

def print_separator(out: Output, title: str):
    out.write(f"\n{'='*65}")
    out.write(f"  {title}")
    out.write(f"{'='*65}")


def print_chunk_summary(out: Output, chunks, doc_class: str, limit: int | None,
                        text_preview_length: int = 200):
    total = len(chunks)
    anzuzeigen = chunks if (limit is None or limit == 0) else chunks[:limit]
    gekuerzt = total > len(anzuzeigen)

    out.write(f"\n  Klasse {doc_class}: {total} Chunks erstellt")
    out.write(f"  {'─'*55}")

    type_counter  = Counter(c.chunk_type for c in chunks)
    level_counter = Counter(c.hierarchy_level for c in chunks)

    out.write(f"  Chunk-Typen:    {dict(type_counter)}")
    out.write(f"  Hierarchie:     {dict(level_counter)}")
    out.write(f"  Confidence:     {set(c.confidence_weight for c in chunks)}")

    label = (
        f"Ausgabe: {len(anzuzeigen)} von {total} Chunks (--limit {limit}):"
        if gekuerzt else f"Alle {total} Chunks:"
    )
    out.write(f"\n  {label}")

    for i, chunk in enumerate(anzuzeigen):
        out.write(
            f"\n  [{i+1:03d}/{total}] chunk_type={chunk.chunk_type} | "
            f"level={chunk.hierarchy_level} | "
            f"tokens={chunk.token_count} | "
            f"confidence={chunk.confidence_weight}"
        )
        if chunk.norm_reference:
            out.write(f"       norm_ref:   {chunk.norm_reference}")
        if chunk.cross_references:
            out.write(f"       cross_refs: {chunk.cross_references}")
        if chunk.section_path:
            out.write(f"       section:    {chunk.section_path}")
        if chunk.heading_breadcrumb:
            out.write(f"       breadcrumb: {chunk.heading_breadcrumb[:70]}")
        if chunk.requirement_id:
            out.write(f"       req_id:     {chunk.requirement_id}")
        if chunk.parent_chunk_id:
            out.write(f"       parent_id:  {chunk.parent_chunk_id[:8]}...")
        if chunk.overlap_with_prev:
            out.write(f"       overlap:    {chunk.overlap_with_prev}")
        # Vollständiger Text in Datei, Preview auf Konsole
        text = chunk.text.strip()
        if text_preview_length > 0 and not out.file:
            text = text[:text_preview_length]
        out.write(f"       text:       {text}")

    if gekuerzt:
        out.write(
            f"\n  ... {total - len(anzuzeigen)} weitere Chunks nicht angezeigt "
            f"(--limit 0 für alle)"
        )


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

async def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Kommandozeilenargumente überschreiben Konfiguration
    limit          = args.limit          if args.limit is not None    else cfg["output"]["limit"]
    output_file    = args.output         if args.output is not None   else cfg["output"]["file"]
    force_class    = args.force_class    if args.force_class          else cfg["document"]["force_class"]
    show_synthetic = (not args.no_synthetic) and cfg["synthetic_tests"]["enabled"]
    preview_len    = cfg["output"]["text_preview_length"]

    out    = Output(output_file, text_preview_length=preview_len)
    router = ChunkingRouter()

    out.write(f"\n  Konfiguration:")
    out.write(f"    limit:       {limit if limit else 'alle'}")
    out.write(f"    force_class: {force_class if force_class else 'automatisch'}")
    out.write(f"    synthetic:   {show_synthetic}")

    # ── Synthetische Tests ─────────────────────────────────────────────────
    if show_synthetic:
        from app.services.parser import DocumentStructure

        class FakeMeta:
            source_type = "gesetz"
            version = "1.0"
            valid_from = None

        print_separator(out, "Test 1: Klasse A – Synthetischer Gesetzestext")
        structure_a = DocumentStructure(
            has_paragraph_markers=True, paragraph_count=3,
            has_heading_hierarchy=False, has_toc=False,
        )
        chunks_a = router.route_and_chunk(
            text=KLASSE_A_TEXT, structure=structure_a,
            doc_id="test-doc-a", metadata=FakeMeta(),
            doc_class_override=force_class,
        )
        print_chunk_summary(out, chunks_a, force_class or "A", limit, preview_len)

        print_separator(out, "Test 2: Klasse C – Synthetische Handreichung")
        structure_c = DocumentStructure(
            has_paragraph_markers=False, paragraph_count=0,
            has_heading_hierarchy=False, has_toc=False,
        )
        chunks_c = router.route_and_chunk(
            text=KLASSE_C_TEXT, structure=structure_c,
            doc_id="test-doc-c", metadata=FakeMeta(),
            doc_class_override=force_class or "C",
        )
        print_chunk_summary(out, chunks_c, force_class or "C", limit, preview_len)

    # ── PDF-Test ───────────────────────────────────────────────────────────
    if args.pdf:
        filepath = Path(args.pdf)
        if filepath.exists():
            class FakeMeta:
                source_type = "gesetz"
                version = "1.0"
                valid_from = None

            print_separator(out, f"PDF-Test: {filepath.name}")
            content = filepath.read_bytes()
            parser  = TikaParser()
            out.write("  Parsing läuft...")
            result = await parser.parse(content=content, filename=filepath.name)

            erkannte_klasse = result.doc_class_hint
            genutzte_klasse = force_class or erkannte_klasse

            out.write(f"  Erkannte Klasse:  {erkannte_klasse}")
            if force_class:
                out.write(f"  Erzwungene Klasse: {force_class}  (--class Parameter)")
            out.write(f"  Zeichen: {result.char_count:,}  |  Wörter: {result.word_count:,}")
            out.write(f"  §-Marker: {result.structure.paragraph_count}  |  "
                      f"Headings: {result.structure.heading_count}  |  "
                      f"ToC: {result.structure.has_toc}")

            chunks = router.route_and_chunk(
                text=result.text,
                structure=result.structure,
                doc_id="test-doc-pdf",
                metadata=FakeMeta(),
                doc_class_override=force_class,
            )
            print_chunk_summary(out, chunks, genutzte_klasse, limit, preview_len)
        else:
            out.write(f"\n  PDF nicht gefunden: {filepath}")
    else:
        if not show_synthetic:
            out.write("\n  Kein PDF angegeben und --no-synthetic aktiv – nichts zu tun.")
        else:
            out.write("\n  Tipp: PDF als Argument übergeben:")
            out.write("  python test_chunker.py ../../docu/Konzepte/Hundegesetz.pdf")

    out.write(f"\n{'='*65}")
    out.write("  Chunker-Test abgeschlossen")
    out.write(f"{'='*65}")
    out.close()


if __name__ == "__main__":
    asyncio.run(main())
