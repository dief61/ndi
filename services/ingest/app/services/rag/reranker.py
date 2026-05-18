# services/ingest/app/services/rag/reranker.py
#
# M3 – Schritt 1.5: Re-Ranker
#
# Aufgaben:
#   1. Duplikat-Entfernung via content_hash
#   2. Cross-Encoder Scoring (ms-marco-MiniLM)
#   3. Klassen-Gewichtung (A×1.0 | B×0.85 | C×0.65)
#   4. Direktlookup-Bonus (direkt gefundene Chunks bevorzugen)
#   5. Top-K auswählen
#
# Konfiguration: services/ingest/rag_config.yaml → reranker
#
# Fallback: Wenn Cross-Encoder-Modell nicht verfügbar ist,
#           wird der RRF-Score aus dem Retriever verwendet.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import structlog
import yaml

from app.services.rag.retriever import RetrievedChunk

logger = structlog.get_logger()

_RAG_CFG_PATH = Path(__file__).parents[3] / "rag_config.yaml"


class ReRanker:
    """
    Re-Ranker für MNR – kombiniert Cross-Encoder-Scoring
    mit Klassen-Gewichtung und Direktlookup-Bonus.

    Wird nach dem Retriever (1.2–1.4) und vor dem
    Context Assembler (1.6) aufgerufen.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._cfg_path  = config_path or _RAG_CFG_PATH
        self._model     = None     # lazy-loaded Cross-Encoder
        self._model_name: Optional[str] = None

    # ── Konfiguration ─────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            with open(self._cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def _rr_cfg(self) -> dict:
        return self._load_config().get("reranker", {})

    # ── Cross-Encoder laden ───────────────────────────────────────────────────

    def _get_model(self, model_name: str):
        """
        Lädt das Cross-Encoder-Modell lazy.
        Gibt None zurück wenn sentence_transformers nicht installiert ist.
        """
        if self._model is not None and self._model_name == model_name:
            return self._model

        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info("Cross-Encoder laden", modell=model_name)
            self._model      = CrossEncoder(model_name)
            self._model_name = model_name
            logger.info("Cross-Encoder geladen", modell=model_name)
            return self._model

        except ImportError:
            logger.warning(
                "sentence_transformers nicht installiert – "
                "Fallback auf RRF-Score",
                modell=model_name,
            )
            return None

        except Exception as e:
            logger.warning(
                "Cross-Encoder Ladefehler – Fallback auf RRF-Score",
                modell=model_name,
                error=str(e),
            )
            return None

    # ── Schritt 1: Duplikat-Entfernung ────────────────────────────────────────

    @staticmethod
    def _dedup(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Entfernt Duplikate via content_hash.
        Bei gleichen Inhalten: den Chunk mit höherem Score behalten.
        Besonders relevant für Klasse B (15% Overlap).
        """
        seen_hashes: dict[str, RetrievedChunk] = {}
        seen_ids:    set[str] = set()
        result:      list[RetrievedChunk] = []

        for c in chunks:
            # chunk_id-Duplikate (z.B. durch mehrere Vektoren + Parent-Expansion)
            if c.chunk_id in seen_ids:
                # Score aktualisieren wenn dieser Fund besser ist
                for existing in result:
                    if existing.chunk_id == c.chunk_id:
                        existing.score = max(existing.score, c.score)
                continue

            # content_hash-Duplikate (fast identischer Inhalt durch Overlap)
            content_hash = hashlib.md5(c.content.encode()).hexdigest()
            if content_hash in seen_hashes:
                existing = seen_hashes[content_hash]
                existing.score = max(existing.score, c.score)
                seen_ids.add(c.chunk_id)
                continue

            seen_hashes[content_hash] = c
            seen_ids.add(c.chunk_id)
            result.append(c)

        dedup_count = len(chunks) - len(result)
        if dedup_count > 0:
            logger.debug("Duplikate entfernt", anzahl=dedup_count,
                         vorher=len(chunks), nachher=len(result))
        return result

    # ── Schritt 2: Cross-Encoder Scoring ─────────────────────────────────────

    def _cross_encoder_score(
        self,
        query:     str,
        chunks:    list[RetrievedChunk],
        model_name: str,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """
        Berechnet Cross-Encoder-Scores für Query-Chunk-Paare.
        Cross-Encoder betrachtet Query und Chunk zusammen –
        semantisch präziser als Vektor-Cosine allein.
        """
        model = self._get_model(model_name)

        if model is None:
            # Fallback: RRF-Score unverändert lassen
            logger.debug("Cross-Encoder Fallback auf RRF-Score")
            return chunks

        # Paare: [(query, chunk_content), ...]
        pairs = [(query, c.content[:512]) for c in chunks]

        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.warning("Cross-Encoder predict fehlgeschlagen",
                           error=str(e))
            return chunks

        # Scores zuweisen
        for chunk, score in zip(chunks, scores):
            chunk.score            = float(score)
            chunk.retrieval_source = "cross_encoder"

        # Unter min_score herausfiltern
        before  = len(chunks)
        chunks  = [c for c in chunks if c.score >= min_score]
        filtered = before - len(chunks)
        if filtered > 0:
            logger.debug("Cross-Encoder Min-Score-Filter",
                         entfernt=filtered, min_score=min_score)

        return chunks

    # ── Schritt 3: Klassen-Gewichtung ─────────────────────────────────────────

    @staticmethod
    def _apply_klassen_gewichtung(
        chunks:   list[RetrievedChunk],
        gewichte: dict,
    ) -> list[RetrievedChunk]:
        """
        Multipliziert den Score mit dem Klassen-Gewicht.
        Stellt sicher dass A-Chunks vor B- und C-Chunks landen.

        A × 1.00 – Rechtstext (höchste Priorität)
        B × 0.85 – Fachdokument
        C × 0.65 – Ergänzungstext
        """
        for c in chunks:
            kw      = gewichte.get(c.doc_class, 1.0)
            c.score = c.score * kw
        return chunks

    # ── Schritt 4: Direktlookup-Bonus ─────────────────────────────────────────

    @staticmethod
    def _apply_direktlookup_boost(
        chunks: list[RetrievedChunk],
        boost:  float,
    ) -> list[RetrievedChunk]:
        """
        Direkt gefundene Chunks (norm_reference-Lookup) erhalten
        einen Score-Multiplikator damit sie immer oben stehen.
        """
        for c in chunks:
            if c.retrieval_source in ("direktlookup", "direktlookup_fuzzy"):
                c.score = c.score * boost
        return chunks

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def rerank(
        self,
        query:       str,
        chunks:      list[RetrievedChunk],
        top_k:       Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Hauptmethode – führt den vollständigen Re-Ranking-Prozess aus.

        Ablauf:
          1. Duplikat-Entfernung (content_hash)
          2. Cross-Encoder Scoring
          3. Klassen-Gewichtung
          4. Direktlookup-Bonus
          5. Sortieren + Top-K auswählen

        Returns:
            Sortierte, deduplizierte Chunk-Liste (Top-K)
        """
        if not chunks:
            return []

        cfg        = self._rr_cfg()
        model_name = cfg.get("modell", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        top_n      = cfg.get("top_n", 16)
        min_score  = cfg.get("min_score", -5.0)
        dedup      = cfg.get("dedup_enabled", True)
        gewichte   = cfg.get("klassen_gewichte", {"A": 1.0, "B": 0.85, "C": 0.65})
        dl_boost   = cfg.get("direktlookup_boost", 2.0)
        final_k    = top_k or self._load_config().get(
                        "retrieval", {}).get("top_k_final", 8)

        logger.info("Re-Ranking gestartet",
                    chunks_eingang=len(chunks),
                    top_n=top_n,
                    final_k=final_k)

        # ── 1. Duplikat-Entfernung ────────────────────────────────────────────
        if dedup:
            chunks = self._dedup(chunks)

        # ── 2. Top-N für Cross-Encoder vorauswählen ───────────────────────────
        # Vorsortierung nach RRF-Score um Cross-Encoder effizient zu nutzen
        chunks_top_n = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]

        # ── 3. Cross-Encoder Scoring ──────────────────────────────────────────
        chunks_top_n = self._cross_encoder_score(
            query, chunks_top_n, model_name, min_score
        )

        # ── 4. Klassen-Gewichtung ─────────────────────────────────────────────
        chunks_top_n = self._apply_klassen_gewichtung(chunks_top_n, gewichte)

        # ── 5. Direktlookup-Bonus ─────────────────────────────────────────────
        chunks_top_n = self._apply_direktlookup_boost(chunks_top_n, dl_boost)

        # ── 6. Sortieren und Top-K auswählen ──────────────────────────────────
        ranked = sorted(chunks_top_n, key=lambda c: c.score, reverse=True)
        result = ranked[:final_k]

        logger.info(
            "Re-Ranking abgeschlossen",
            chunks_eingang  = len(chunks),
            chunks_nach_dedup = len(chunks_top_n),
            chunks_ausgabe  = len(result),
            top_score       = round(result[0].score, 4) if result else 0,
            top_chunk_klasse= result[0].doc_class if result else "–",
        )

        return result
