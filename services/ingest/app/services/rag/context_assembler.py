# services/ingest/app/services/rag/context_assembler.py
#
# M3 – Schritt 1.6: Context Assembly
#
# Aufgaben:
#   1. Präambel je Chunk (Klassen-Label, Normreferenz, Normtyp)
#   2. IM-Signal-Annotierungen (Entity/Attribut/Persistenz-Kandidaten)
#   3. Kontext-String für LLM zusammenbauen
#   4. Token-Limit einhalten (max_tokens aus rag_config.yaml)
#
# Konfiguration: services/ingest/rag_config.yaml → context

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
import yaml

from app.services.rag.retriever import RetrievedChunk

logger = structlog.get_logger()

_RAG_CFG_PATH = Path(__file__).parents[3] / "rag_config.yaml"

# Grobe Token-Schätzung: 1 Token ≈ 4 Zeichen (deutsch)
_ZEICHEN_PRO_TOKEN = 4


class ContextAssembler:
    """
    Baut den finalen LLM-Kontext aus den Re-Ranked-Chunks.

    Jeder Chunk erhält eine strukturierte Präambel:

    Klasse A (Rechtstext):
      [RECHTSNORM | § 3 Abs. 1 NHundG | MUST]
      [IM-ENTITY-KANDIDAT: Hundehalter → Tabelle]      ← wenn im_signale: true
      <Chunk-Inhalt>

    Klasse B (Fachdokument):
      [FACHKONZEPT | 3.2.1 Datenanforderungen > Pflichtfelder]
      <Chunk-Inhalt>

    Klasse C (Auslegungstext):
      [AUSLEGUNG | FAQ Sachkundenachweis | Konfidenz: niedrig]
      <Chunk-Inhalt>
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._cfg_path = config_path or _RAG_CFG_PATH

    # ── Konfiguration ─────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            with open(self._cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def _ctx_cfg(self) -> dict:
        return self._load_config().get("context", {})

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // _ZEICHEN_PRO_TOKEN)

    # ── Präambel-Erzeugung ────────────────────────────────────────────────────

    def _build_praembel(
        self,
        chunk:        RetrievedChunk,
        klassen_label: bool,
        traceability:  bool,
        c_konfidenz:   bool,
    ) -> str:
        """
        Baut die Kontext-Präambel für einen Chunk.

        Format: [LABEL | Referenz | Typ]
        """
        label_map = {
            "A": "RECHTSNORM",
            "B": "FACHKONZEPT",
            "C": "AUSLEGUNG",
        }
        label = label_map.get(chunk.doc_class, "NORM") if klassen_label else ""
        parts = [label] if label else []

        if traceability:
            if chunk.norm_reference:
                parts.append(chunk.norm_reference)
            elif chunk.heading_breadcrumb:
                parts.append(chunk.heading_breadcrumb)
            elif chunk.section_path:
                parts.append(f"Abschnitt {chunk.section_path}")

        if chunk.chunk_type and chunk.chunk_type not in ("tatbestand", ""):
            parts.append(chunk.chunk_type.upper())

        if c_konfidenz and chunk.doc_class == "C":
            parts.append("Konfidenz: niedrig")

        return f"[{' | '.join(parts)}]" if parts else ""

    # ── IM-Signal-Annotierungen ────────────────────────────────────────────────

    @staticmethod
    def _build_im_annotierungen(chunk: RetrievedChunk) -> list[str]:
        """
        Erzeugt IM-Signal-Annotierungen aus im_signals-Daten (M2).

        Drei Typen:
          [IM-ENTITY-KANDIDAT: Hundehalter → Tabelle]
          [IM-ATTRIBUT-KANDIDAT: Sachkunde → Spalte]
          [IM-PERSISTENZ: Datenpersistenz erforderlich]
        """
        if not chunk.im_signals:
            return []

        lines = []
        signals = chunk.im_signals

        # Entity-Kandidaten
        for entity in signals.get("entity_kandidaten", []):
            if entity:
                lines.append(f"[IM-ENTITY-KANDIDAT: {entity} → Tabelle]")

        # Entity-Definition (bei DEF-Chunks)
        if signals.get("entity_def"):
            lines.append(
                f"[IM-ENTITY-DEF: {signals['entity_def']} = kanonische Definition]"
            )

        # Attribut-Kandidaten
        for attr in signals.get("attribut_kandidaten", []):
            if attr:
                lines.append(f"[IM-ATTRIBUT-KANDIDAT: {attr} → Spalte]")

        # Persistenz-Signal
        if signals.get("persistenz"):
            lines.append("[IM-PERSISTENZ: Datenpersistenz erforderlich]")

        # Relation-Kandidaten
        for rel in signals.get("relation_kandidaten", []):
            if rel.get("von") and rel.get("zu"):
                lines.append(
                    f"[IM-RELATION: {rel['von']} → {rel['zu']}]"
                )

        return lines

    # ── Einzelner Chunk-Block ─────────────────────────────────────────────────

    def _format_chunk(
        self,
        chunk:         RetrievedChunk,
        cfg:           dict,
        chunk_nr:      int,
    ) -> str:
        """
        Formatiert einen Chunk als vollständigen Kontext-Block.

        Struktur:
          ── Quelle N ──────────────────────────────
          [RECHTSNORM | § 3 NHundG | TATBESTAND]
          [IM-ENTITY-KANDIDAT: Hundehalter → Tabelle]
          <Inhalt>
        """
        klassen_label = cfg.get("klassen_label", True)
        traceability  = cfg.get("traceability", True)
        c_konfidenz   = cfg.get("c_konfidenz_hinweis", True)
        im_signale    = cfg.get("im_signale", True)

        lines = [f"── Quelle {chunk_nr} {'─' * (40 - len(str(chunk_nr)))}"]

        # Präambel
        praembel = self._build_praembel(
            chunk, klassen_label, traceability, c_konfidenz
        )
        if praembel:
            lines.append(praembel)

        # IM-Annotierungen
        if im_signale:
            lines.extend(self._build_im_annotierungen(chunk))

        # Inhalt
        lines.append(chunk.content.strip())
        lines.append("")   # Leerzeile nach Chunk

        return "\n".join(lines)

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def assemble(
        self,
        query:    str,
        chunks:   list[RetrievedChunk],
        max_tokens: Optional[int] = None,
    ) -> tuple[str, list[dict]]:
        """
        Baut den finalen LLM-Kontext aus Re-Ranked-Chunks.

        Returns:
            (kontext_string, traceability_liste)

            kontext_string:     Vollständiger Kontext für den LLM-Prompt
            traceability_liste: Quellenverweise für jede Quelle im Kontext
        """
        cfg        = self._ctx_cfg()
        max_tok    = max_tokens or cfg.get("max_tokens", 4096)
        # Budget: Kontext darf max. 60% der Token nutzen
        # (Rest für System-Prompt + Query + Antwort)
        ctx_budget = int(max_tok * 0.60)

        traceability = []
        blocks       = []
        used_tokens  = 0

        for nr, chunk in enumerate(chunks, start=1):
            block      = self._format_chunk(chunk, cfg, nr)
            block_toks = self._estimate_tokens(block)

            # Token-Budget prüfen
            if used_tokens + block_toks > ctx_budget:
                logger.debug(
                    "Token-Budget erreicht",
                    chunks_eingebaut=nr - 1,
                    chunks_gesamt=len(chunks),
                    tokens_genutzt=used_tokens,
                    budget=ctx_budget,
                )
                break

            blocks.append(block)
            used_tokens += block_toks

            # Traceability-Eintrag
            traceability.append({
                "quelle_nr":      nr,
                "chunk_id":       chunk.chunk_id,
                "doc_class":      chunk.doc_class,
                "norm_reference": chunk.norm_reference,
                "section_path":   chunk.section_path,
                "chunk_type":     chunk.chunk_type,
                "score":          round(chunk.score, 4),
                "retrieval_src":  chunk.retrieval_source,
                "im_signals":     bool(chunk.im_signals),
            })

        # Kontext zusammenbauen
        header = (
            f"Folgende {len(blocks)} Quellen wurden für die Anfrage gefunden:\n"
            f"Anfrage: {query}\n"
            f"{'═' * 60}\n\n"
        )
        kontext = header + "\n".join(blocks)

        logger.info(
            "Context Assembly abgeschlossen",
            chunks_eingebaut = len(blocks),
            chunks_verfügbar = len(chunks),
            tokens_geschätzt = used_tokens,
            budget           = ctx_budget,
            mit_im_signalen  = sum(1 for c in chunks[:len(blocks)]
                                   if c.im_signals),
        )

        return kontext, traceability

    def get_praembel(self, chunk: RetrievedChunk) -> str:
        """Öffentliche Hilfsmethode – Präambel für einen Chunk."""
        cfg = self._ctx_cfg()
        return self._build_praembel(
            chunk,
            cfg.get("klassen_label", True),
            cfg.get("traceability", True),
            cfg.get("c_konfidenz_hinweis", True),
        )
