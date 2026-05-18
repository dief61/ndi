# services/ingest/app/services/rag/rag_pipeline.py
#
# M3 – RAG-Pipeline: Orchestrierung aller Schritte 1.1–1.6
#
# Diese Klasse ist der einzige Einstiegspunkt für RAG-Abfragen.
# Der rag_router ruft ausschließlich RAGPipeline.run() auf –
# die interne Schrittfolge bleibt vollständig gekapselt.
#
# Schrittfolge:
#   1.1  QueryTransformer  → QueryBundle
#   1.2–1.4  HybridRetriever  → list[RetrievedChunk]
#   1.5  ReRanker          → list[RetrievedChunk] (sortiert, dedupliziert)
#   1.6  ContextAssembler  → (kontext: str, traceability: list[dict])

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog
import yaml

from app.services.rag.query_transformer import QueryBundle, QueryTransformer
from app.services.rag.retriever import HybridRetriever, RetrievedChunk
from app.services.rag.reranker import ReRanker
from app.services.rag.context_assembler import ContextAssembler

logger = structlog.get_logger()

_RAG_CFG_PATH = Path(__file__).parents[3] / "rag_config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenstruktur
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """
    Vollständiges Ergebnis eines RAG-Pipeline-Laufs.
    Wird direkt vom rag_router als Response verwendet.
    """
    # Query-Metadaten
    original_query:   str
    query_typ:        str                   # NORM | ENTITY | IM | GENERAL
    norm_reference:   Optional[str]
    direktlookup:     bool

    # Ergebnis-Chunks (Re-Ranked)
    chunks:           list[RetrievedChunk]

    # Assemblierter LLM-Kontext
    kontext:          str
    traceability:     list[dict]

    # Timing und Statistik
    dauer_ms:         int
    stats:            dict = field(default_factory=dict)

    # Debug-Informationen (nur wenn debug=True)
    debug:            Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# RAG-Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestriert die vollständige RAG-Pipeline für MNR.

    Verwendung im rag_router:

        pipeline = RAGPipeline(
            pool     = pool,
            embedder = embedder,
        )
        result = await pipeline.run(
            query = "Was muss ein Hundehalter nachweisen?",
            debug = True,
        )

    Die Pipeline liest alle Konfigurationsparameter zur Laufzeit
    aus rag_config.yaml – kein Neustart erforderlich.
    """

    def __init__(
        self,
        pool,                          # asyncpg.Pool
        embedder,                      # Embedder-Instanz aus M1
        config_path: Optional[Path] = None,
    ):
        self._pool        = pool
        self._embedder    = embedder
        self._cfg_path    = config_path or _RAG_CFG_PATH

        # Komponenten (lazy oder direkt initialisiert)
        self._transformer = QueryTransformer(config_path=self._cfg_path)
        self._retriever   = HybridRetriever(
            pool        = pool,
            embedder    = embedder,
            config_path = self._cfg_path,
        )
        self._reranker    = ReRanker(config_path=self._cfg_path)
        self._assembler   = ContextAssembler(config_path=self._cfg_path)

    # ── Konfiguration ─────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            with open(self._cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    # ── Hauptmethode ──────────────────────────────────────────────────────────

    async def run(
        self,
        query:    str,
        debug:    bool = False,
    ) -> RAGResult:
        """
        Führt die vollständige RAG-Pipeline aus.

        Schritt 1.1  Query-Analyse & -Transformation
        Schritt 1.2  Klassen-sensitiver Hybrid-Retriever
        Schritt 1.3  Parent-Child-Expansion
        Schritt 1.4  Cross-Reference-Expansion (Klasse A)
        Schritt 1.5  Re-Ranking (Cross-Encoder + Klassen-Gewichtung)
        Schritt 1.6  Context Assembly (Präambeln + IM-Signale)

        Args:
            query:  Natürlichsprachliche Anfrage
            debug:  True = detaillierte Zwischenschritte in RAGResult.debug

        Returns:
            RAGResult mit Chunks, Kontext, Traceability und Statistik
        """
        t_start  = time.monotonic()
        debug_log: dict = {}

        # ── 1.1 Query-Transformation ──────────────────────────────────────────
        t1 = time.monotonic()
        bundle: QueryBundle = await self._transformer.transform(
            query = query,
            debug = debug,
        )
        ms_transform = int((time.monotonic() - t1) * 1000)

        if debug:
            debug_log["transform"] = {
                "query_typ":        bundle.query_typ,
                "norm_reference":   bundle.norm_reference,
                "direktlookup":     bundle.direktlookup,
                "vektoren_anzahl":  len(bundle.vektoren),
                "vektoren":         [
                    {"strategie": v.strategie,
                     "gewicht":   v.gewicht,
                     "text":      v.text[:80]}
                    for v in bundle.vektoren
                ],
                "metadata_filter":  bundle.metadata_filter,
                "dauer_ms":         ms_transform,
                **bundle.debug,
            }

        logger.info(
            "Pipeline 1.1 abgeschlossen",
            typ=bundle.query_typ,
            direktlookup=bundle.direktlookup,
            vektoren=len(bundle.vektoren),
            dauer_ms=ms_transform,
        )

        # ── 1.2–1.4 Retrieval ─────────────────────────────────────────────────
        t2 = time.monotonic()
        raw_chunks: list[RetrievedChunk] = await self._retriever.retrieve(bundle)
        ms_retrieval = int((time.monotonic() - t2) * 1000)

        if debug:
            debug_log["retrieval"] = {
                "chunks_gesamt":   len(raw_chunks),
                "klassen":         _klassen_verteilung(raw_chunks),
                "quellen":         _quellen_verteilung(raw_chunks),
                "dauer_ms":        ms_retrieval,
            }

        logger.info(
            "Pipeline 1.2–1.4 abgeschlossen",
            chunks=len(raw_chunks),
            dauer_ms=ms_retrieval,
        )

        # ── 1.5 Re-Ranking ────────────────────────────────────────────────────
        t3 = time.monotonic()
        ranked_chunks: list[RetrievedChunk] = self._reranker.rerank(
            query  = query,
            chunks = raw_chunks,
        )
        ms_rerank = int((time.monotonic() - t3) * 1000)

        if debug:
            debug_log["reranking"] = {
                "chunks_eingang":  len(raw_chunks),
                "chunks_ausgabe":  len(ranked_chunks),
                "top_score":       round(ranked_chunks[0].score, 4)
                                   if ranked_chunks else 0,
                "top_norm_ref":    ranked_chunks[0].norm_reference
                                   if ranked_chunks else None,
                "dauer_ms":        ms_rerank,
            }

        logger.info(
            "Pipeline 1.5 abgeschlossen",
            chunks_ein=len(raw_chunks),
            chunks_aus=len(ranked_chunks),
            dauer_ms=ms_rerank,
        )

        # ── 1.6 Context Assembly ──────────────────────────────────────────────
        t4 = time.monotonic()
        kontext, traceability = self._assembler.assemble(
            query  = query,
            chunks = ranked_chunks,
        )
        ms_assembly = int((time.monotonic() - t4) * 1000)

        if debug:
            debug_log["assembly"] = {
                "quellen_eingebaut":  len(traceability),
                "kontext_zeichen":    len(kontext),
                "kontext_vorschau":   kontext[:200],
                "dauer_ms":           ms_assembly,
            }

        logger.info(
            "Pipeline 1.6 abgeschlossen",
            quellen=len(traceability),
            kontext_len=len(kontext),
            dauer_ms=ms_assembly,
        )

        # ── Gesamtstatistik ───────────────────────────────────────────────────
        ms_gesamt = int((time.monotonic() - t_start) * 1000)
        stats     = {
            "dauer_ms_gesamt":   ms_gesamt,
            "dauer_ms_transform":  ms_transform,
            "dauer_ms_retrieval":  ms_retrieval,
            "dauer_ms_reranking":  ms_rerank,
            "dauer_ms_assembly":   ms_assembly,
            "chunks_retrieval":  len(raw_chunks),
            "chunks_final":      len(ranked_chunks),
            "quellen_kontext":   len(traceability),
        }

        logger.info(
            "RAG-Pipeline abgeschlossen",
            query_typ    = bundle.query_typ,
            direktlookup = bundle.direktlookup,
            chunks_final = len(ranked_chunks),
            dauer_ms     = ms_gesamt,
        )

        return RAGResult(
            original_query  = query,
            query_typ       = bundle.query_typ,
            norm_reference  = bundle.norm_reference,
            direktlookup    = bundle.direktlookup,
            chunks          = ranked_chunks,
            kontext         = kontext,
            traceability    = traceability,
            dauer_ms        = ms_gesamt,
            stats           = stats,
            debug           = debug_log if debug else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _klassen_verteilung(chunks: list[RetrievedChunk]) -> dict:
    """Zählt Chunks je Dokumentklasse."""
    result: dict[str, int] = {}
    for c in chunks:
        result[c.doc_class] = result.get(c.doc_class, 0) + 1
    return result


def _quellen_verteilung(chunks: list[RetrievedChunk]) -> dict:
    """Zählt Chunks je Retrieval-Quelle."""
    result: dict[str, int] = {}
    for c in chunks:
        result[c.retrieval_source] = result.get(c.retrieval_source, 0) + 1
    return result
