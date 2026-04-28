#!/usr/bin/env python3
# services/ingest/test_embedder.py
#
# Testet den Embedder: Modell laden, Chunks embedden, Ähnlichkeit prüfen.
#
# Aufruf:
#   python test_embedder.py                          # synthetische Chunks
#   python test_embedder.py --pdf ../../docu/...pdf  # echtes Dokument
#   python test_embedder.py --model multilingual-e5  # anderes Modell
#   python test_embedder.py --config embedder_config.yaml

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.embedder import Embedder
from app.services.chunker import Chunk


# ── Synthetische Test-Chunks ──────────────────────────────────────────────────

SYNTHETIC_CHUNKS = [
    Chunk(
        chunk_id="test-001", doc_id="doc-a", doc_class="A",
        text="Die Meldebehörde ist verpflichtet, die Meldedaten der betroffenen Person zu speichern.",
        chunk_type="must", norm_reference="§ 1 Abs. 1 MeldeG",
        confidence_weight=1.0, hierarchy_level=2,
    ),
    Chunk(
        chunk_id="test-002", doc_id="doc-a", doc_class="A",
        text="Die Anmeldepflicht obliegt der einziehenden Person binnen zwei Wochen nach Einzug.",
        chunk_type="deadline", norm_reference="§ 2 Abs. 1 MeldeG",
        confidence_weight=1.0, hierarchy_level=2,
    ),
    Chunk(
        chunk_id="test-003", doc_id="doc-a", doc_class="A",
        text="Im Sinne dieses Gesetzes ist eine betroffene Person jede natürliche Person die eine Wohnung anmeldet.",
        chunk_type="definition", norm_reference="§ 1 Abs. 3 MeldeG",
        confidence_weight=1.0, hierarchy_level=2,
    ),
    Chunk(
        chunk_id="test-004", doc_id="doc-b", doc_class="B",
        text="Das System muss eine REST-API gemäß OpenAPI 3.1 bereitstellen.",
        chunk_type="anforderung", requirement_id="A-001",
        confidence_weight=0.85, hierarchy_level=2,
    ),
    Chunk(
        chunk_id="test-005", doc_id="doc-c", doc_class="C",
        text="Bei der Ummeldung müssen Sie Ihren Personalausweis und eine Wohnungsgeberbestätigung mitbringen.",
        chunk_type="tatbestand",
        confidence_weight=0.65, hierarchy_level=1,
    ),
]

TEST_QUERIES = [
    "Wer muss sich ummelden?",
    "Welche Daten müssen gespeichert werden?",
    "Was ist eine betroffene Person?",
    "REST-API Anforderungen",
]


def parse_args():
    parser = argparse.ArgumentParser(description="NDI Embedder-Test")
    parser.add_argument("--pdf",    type=str, default=None, help="PDF-Datei für echte Chunks")
    parser.add_argument("--model",  type=str, default=None,
                        help="Modell-Profil überschreiben (deepset-mxbai | multilingual-e5)")
    parser.add_argument("--config", type=str, default=None, help="Pfad zur embedder_config.yaml")
    parser.add_argument("--limit",  type=int, default=5,    help="Max. Chunks bei PDF-Test (0=alle)")
    return parser.parse_args()


def print_separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Einfache Cosine-Similarity ohne numpy."""
    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0


async def main():
    args = parse_args()

    # Config-Pfad
    config_path = Path(args.config) if args.config else None

    # Modell-Override: aktives Modell in Config temporär überschreiben
    if args.model:
        import yaml
        cfg_path = config_path or Path(__file__).parent / "embedder_config.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if args.model not in cfg.get("models", {}):
            print(f"  Fehler: Modell-Profil '{args.model}' nicht in Config gefunden.")
            print(f"  Verfügbare Profile: {list(cfg.get('models', {}).keys())}")
            sys.exit(1)
        cfg["active_model"] = args.model
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                          delete=False, encoding="utf-8")
        yaml.dump(cfg, tmp)
        tmp.close()
        config_path = Path(tmp.name)
        print(f"  Modell-Override: {args.model}")

    # Embedder initialisieren
    embedder = Embedder(config_path=config_path)

    print_separator("Embedder-Konfiguration")
    info = embedder.active_model_info
    for k, v in info.items():
        print(f"  {k:<15} {v}")

    # ── Test 1: Synthetische Chunks ───────────────────────────────────────────
    print_separator("Test 1: Synthetische Chunks embedden")

    t0 = time.time()
    chunks = await embedder.embed_chunks(SYNTHETIC_CHUNKS)
    elapsed = time.time() - t0

    print(f"\n  {len(chunks)} Chunks in {elapsed:.2f}s eingebettet")
    print(f"  Durchschnitt: {elapsed/len(chunks)*1000:.0f}ms pro Chunk")

    for chunk in chunks:
        emb = chunk.embedding
        print(f"\n  [{chunk.chunk_id}] {chunk.chunk_type} | "
              f"doc_class={chunk.doc_class}")
        print(f"    norm_ref:  {chunk.norm_reference or chunk.requirement_id or '–'}")
        print(f"    text:      {chunk.text[:80]}...")
        print(f"    embedding: dim={len(emb)} | "
              f"min={min(emb):.4f} | max={max(emb):.4f} | "
              f"first3={[round(x,4) for x in emb[:3]]}")

    # ── Test 2: Query-Ähnlichkeit ─────────────────────────────────────────────
    print_separator("Test 2: Query-Ähnlichkeit")
    print("  (höherer Score = semantisch ähnlicher)\n")

    for query in TEST_QUERIES:
        query_emb = embedder.embed_query(query)
        print(f"  Query: \"{query}\"")
        scores = []
        for chunk in chunks:
            sim = cosine_similarity(query_emb, chunk.embedding)
            scores.append((sim, chunk))
        scores.sort(reverse=True)
        for sim, chunk in scores[:3]:
            bar = "█" * int(sim * 20)
            print(f"    {sim:.4f} {bar:<20} {chunk.text[:60]}...")
        print()

    # ── Test 3: Cache-Test ────────────────────────────────────────────────────
    print_separator("Test 3: Cache-Effizienz")

    t0 = time.time()
    chunks_cached = await embedder.embed_chunks(SYNTHETIC_CHUNKS)
    elapsed_cached = time.time() - t0
    print(f"  Zweiter Durchlauf (Cache): {elapsed_cached*1000:.0f}ms "
          f"(statt {elapsed*1000:.0f}ms)")
    print(f"  Speedup: {elapsed/elapsed_cached:.1f}x")

    # ── Test 4: PDF (optional) ────────────────────────────────────────────────
    if args.pdf:
        filepath = Path(args.pdf)
        if filepath.exists():
            print_separator(f"Test 4: Echtes Dokument – {filepath.name}")

            from app.services.parser import TikaParser
            from app.services.chunker import ChunkingRouter

            print("  Parsing + Chunking läuft...")
            content = filepath.read_bytes()
            parser  = TikaParser()
            result  = await parser.parse(content=content, filename=filepath.name)
            router  = ChunkingRouter()

            class FakeMeta:
                source_type = "gesetz"; version = "1.0"; valid_from = None

            all_chunks = router.route_and_chunk(
                text=result.text, structure=result.structure,
                doc_id="test-pdf", metadata=FakeMeta(),
            )

            limit = args.limit if args.limit > 0 else len(all_chunks)
            subset = all_chunks[:limit]

            print(f"  {len(all_chunks)} Chunks gesamt – embedde erste {len(subset)}...")

            t0 = time.time()
            embedded = await embedder.embed_chunks(subset)
            elapsed  = time.time() - t0

            print(f"\n  {len(embedded)} Chunks in {elapsed:.2f}s eingebettet")
            print(f"  Durchschnitt: {elapsed/len(embedded)*1000:.0f}ms pro Chunk")
            print(f"  Embedding-Dimensionen: {len(embedded[0].embedding)}")
        else:
            print(f"\n  PDF nicht gefunden: {filepath}")

    print(f"\n{'='*65}")
    print("  Embedder-Test abgeschlossen")
    print(f"{'='*65}\n")

    # Temp-Datei aufräumen
    if args.model and config_path and "tmp" in str(config_path):
        import os
        os.unlink(config_path)


if __name__ == "__main__":
    asyncio.run(main())
