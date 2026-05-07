# services/ingest/app/services/question_filter.py
#
# QuestionFilter: Erkennt Fragen in Chunks und behandelt sie
# gemäß Konfiguration (loggen, filtern oder markieren).
#
# Wird im Chunker nach der Chunk-Erzeugung angewendet.
# Konfiguration via nlp_config.yaml (Sektion question_filter).

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog
import yaml

from app.services.chunker import Chunk

logger = structlog.get_logger()

_DEFAULT_CONFIG = Path(__file__).parents[2] / "nlp_config.yaml"


def load_question_config(
    config_path: Optional[Path] = None,
    scope: str = "ingest",
) -> dict:
    """
    Lädt question_filter-Konfiguration aus nlp_config.yaml.

    Args:
        scope: "ingest" oder "nlp" – welcher Filter gelesen wird
    """
    path = config_path or _DEFAULT_CONFIG
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    qf = cfg.get("question_filter", {})

    # Scope-spezifische Einstellungen + gemeinsame detection-Parameter mergen
    scope_cfg    = qf.get(scope, {})
    detection    = qf.get("detection", {})
    merged = {**detection, **scope_cfg}
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectedQuestion:
    """Eine erkannte Frage in einem Chunk."""
    chunk_id:       str
    question_text:  str
    question_type:  str          # direkt | rhetorisch | liste | unbekannt
    context_before: str = ""
    context_after:  str = ""
    section_path:   Optional[str] = None
    chunk_position: int = 0
    confidence:     float = 0.0


@dataclass
class FilterResult:
    """Ergebnis der Fragen-Filterung für ein Dokument."""
    kept_chunks:     list[Chunk]             # Chunks die behalten werden
    filtered_chunks: list[Chunk]             # gefilterte Chunks (Fragen)
    questions:       list[DetectedQuestion]  # erkannte Fragen (zum Loggen)


# ─────────────────────────────────────────────────────────────────────────────
# QuestionFilter
# ─────────────────────────────────────────────────────────────────────────────

class QuestionFilter:
    """
    Erkennt und behandelt Fragen in Chunks.

    Konfiguration via nlp_config.yaml Sektion question_filter.
    Liest Konfiguration bei jedem Aufruf frisch (kein Caching) –
    Änderungen wirken sofort ohne Neustart.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path

    def filter(
        self,
        chunks:       list[Chunk],
        doc_class:    str,
        full_text:    str = "",
        scope:        str = "ingest",
    ) -> FilterResult:
        """
        Hauptmethode: Chunks auf Fragen prüfen und gemäß Config behandeln.

        Args:
            chunks:    Liste von Chunks aus dem ChunkingRouter
            doc_class: Dokumentklasse (A/B/C)
            full_text: Volltext des Dokuments (für Kontext-Extraktion)
            scope:     "ingest" oder "nlp" – welcher Filter-Schalter gilt

        Returns:
            FilterResult mit kept_chunks, filtered_chunks und questions
        """
        cfg = load_question_config(self.config_path, scope=scope)

        # Filter deaktiviert oder Klasse nicht betroffen?
        if not cfg.get("enabled", True):
            return FilterResult(
                kept_chunks=chunks,
                filtered_chunks=[],
                questions=[],
            )

        apply_to = cfg.get("apply_to_classes", ["B", "C"])
        if doc_class not in apply_to:
            return FilterResult(
                kept_chunks=chunks,
                filtered_chunks=[],
                questions=[],
            )

        action     = cfg.get("action", "exclude")
        ctx_window = cfg.get("context_window", 150)

        kept:     list[Chunk]             = []
        filtered: list[Chunk]             = []
        questions: list[DetectedQuestion] = []

        for i, chunk in enumerate(chunks):
            detected = self._detect_questions(chunk.text, cfg)

            if not detected:
                kept.append(chunk)
                continue

            # Kontext aus Volltext bestimmen
            context_before, context_after = self._get_context(
                full_text, chunk.text, ctx_window
            )

            for q_text, q_type, q_conf in detected:
                questions.append(DetectedQuestion(
                    chunk_id=chunk.chunk_id,
                    question_text=q_text,
                    question_type=q_type,
                    context_before=context_before,
                    context_after=context_after,
                    section_path=chunk.section_path,
                    chunk_position=i,
                    confidence=q_conf,
                ))

            # Aktion ausführen
            if action == "exclude":
                filtered.append(chunk)
                logger.debug(
                    "Frage herausgefiltert",
                    chunk_id=chunk.chunk_id,
                    fragen=len(detected),
                )

            elif action == "include_as_type":
                chunk.chunk_type = "frage"
                kept.append(chunk)

            else:  # include
                kept.append(chunk)

        if questions:
            logger.info(
                "Fragen-Filter abgeschlossen",
                doc_class=doc_class,
                action=action,
                fragen_gesamt=len(questions),
                gefiltert=len(filtered),
                behalten=len(kept),
            )

        # Duplikate entfernen (gleicher Fragetext aus Overlap)
        seen: set[str] = set()
        unique_questions = []
        for q in questions:
            key = q.question_text.strip().lower()
            if key not in seen:
                seen.add(key)
                unique_questions.append(q)

        if len(unique_questions) < len(questions):
            logger.debug(
                "Doppelte Fragen entfernt",
                gesamt=len(questions),
                unique=len(unique_questions),
            )

        return FilterResult(
            kept_chunks=kept,
            filtered_chunks=filtered,
            questions=unique_questions,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Fragen-Erkennung
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_questions(
        self,
        text: str,
        cfg: dict,
    ) -> list[tuple[str, str, float]]:
        """
        Erkennt Fragen im Text.
        Gibt Liste von (frage_text, typ, konfidenz) zurück.
        """
        min_len  = cfg.get("min_length", 10)
        max_len  = cfg.get("max_length", 300)
        req_qm   = cfg.get("require_question_mark", True)
        q_words  = [w.lower() for w in cfg.get("question_words", [])]
        rhet_pat = [p.lower() for p in cfg.get("rhetorical_patterns", [])]

        detected = []

        # Text in Sätze aufteilen
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Längen-Filter
            if len(sent) < min_len or len(sent) > max_len:
                continue

            # Fragezeichen-Pflicht
            if req_qm and not sent.rstrip().endswith("?"):
                continue

            # Konfidenz berechnen
            conf, q_type = self._classify_question(
                sent, q_words, rhet_pat
            )

            if conf > 0:
                detected.append((sent, q_type, conf))

        return detected

    def _classify_question(
        self,
        text: str,
        question_words: list[str],
        rhetorical_patterns: list[str],
    ) -> tuple[float, str]:
        """
        Klassifiziert eine potenzielle Frage und berechnet Konfidenz.

        Returns:
            (konfidenz, typ) – konfidenz=0 wenn keine Frage erkannt
        """
        text_lower  = text.lower().strip()
        first_word  = text_lower.split()[0] if text_lower.split() else ""
        has_qm      = text_lower.endswith("?")

        if not has_qm:
            return 0.0, "unbekannt"

        # Rhetorik prüfen (höchste Priorität)
        for pattern in rhetorical_patterns:
            if pattern in text_lower:
                return 0.75, "rhetorisch"

        # Fragewort am Anfang → direkte Frage
        if first_word in question_words:
            return 0.95, "direkt"

        # Listenförmige Fragen: "Was ist X? Was ist Y?" in einem Chunk
        q_count = text.count("?")
        if q_count >= 2:
            return 0.85, "liste"

        # Fragezeichen ohne Fragewort → niedrigere Konfidenz
        if has_qm:
            return 0.65, "direkt"

        return 0.0, "unbekannt"

    def _get_context(
        self,
        full_text: str,
        chunk_text: str,
        window: int,
    ) -> tuple[str, str]:
        """
        Extrahiert Text vor und nach dem Chunk aus dem Volltext.
        """
        if not full_text or not chunk_text:
            return "", ""

        pos = full_text.find(chunk_text[:50])
        if pos == -1:
            return "", ""

        before = full_text[max(0, pos - window):pos].strip()
        after  = full_text[pos + len(chunk_text):
                           pos + len(chunk_text) + window].strip()
        return before, after
