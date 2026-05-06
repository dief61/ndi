# services/ingest/app/services/abbrev_normalizer.py
#
# AbbrevNormalizer: Löst Abkürzungen im Text auf und erstellt
# ein Positions-Mapping für lückenlose Traceability.
#
# Wird zwischen Parser und Chunker ausgeführt:
#   TikaParser → AbbrevNormalizer → ChunkingRouter
#
# Ergebnis:
#   - resolved_text:  Text mit aufgelösten Abkürzungen (für Chunking/Embedding)
#   - original_text:  Originaltext (für Traceability)
#   - abbrev_map:     Positions-Mapping original ↔ aufgelöst

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger()

_DEFAULT_DICT = Path(__file__).parents[2] / "abbrev_dict.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AbbrevReplacement:
    """Eine einzelne Abkürzungsersetzung mit Positionsinformation."""
    abbrev:     str       # Originalabkürzung: "NHundG"
    resolved:   str       # Auflösung: "Niedersächsisches Hundegesetz"
    orig_start: int       # Startposition im Originaltext
    orig_end:   int       # Endposition im Originaltext
    res_start:  int       # Startposition im aufgelösten Text
    res_end:    int       # Endposition im aufgelösten Text
    label:      str = ""  # Entitätstyp: GESETZ, BEHÖRDE, ...


@dataclass
class NormalizationResult:
    """Ergebnis der Abkürzungsauflösung."""
    original_text:  str
    resolved_text:  str
    replacements:   list[AbbrevReplacement] = field(default_factory=list)

    @property
    def abbrev_map(self) -> list[dict]:
        """Serialisiertes Positions-Mapping für JSONB-Speicherung."""
        return [
            {
                "abbrev":     r.abbrev,
                "resolved":   r.resolved,
                "orig_start": r.orig_start,
                "orig_end":   r.orig_end,
                "res_start":  r.res_start,
                "res_end":    r.res_end,
                "label":      r.label,
            }
            for r in self.replacements
        ]

    def original_position(self, res_pos: int) -> int:
        """
        Gibt die Position im Originaltext für eine Position
        im aufgelösten Text zurück.

        Ermöglicht Traceability von NER/SVO-Positionen
        zurück zum Originaltext.
        """
        # Offset durch alle Ersetzungen vor res_pos berechnen
        offset = 0
        for r in sorted(self.replacements, key=lambda x: x.res_start):
            if r.res_start > res_pos:
                break
            if r.res_start <= res_pos <= r.res_end:
                # Position liegt innerhalb einer Ersetzung
                return r.orig_start
            # Offset durch diese Ersetzung
            offset += (r.orig_end - r.orig_start) - (r.res_end - r.res_start)

        return res_pos + offset

    def get_original_snippet(self, res_start: int, res_end: int) -> str:
        """
        Gibt den Originaltext-Ausschnitt für einen aufgelösten
        Textbereich zurück.

        Für Traceability-Anzeige in der Fachaufsicht.
        """
        orig_start = self.original_position(res_start)
        orig_end   = self.original_position(res_end)
        return self.original_text[orig_start:orig_end]


# ─────────────────────────────────────────────────────────────────────────────
# AbbrevNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class AbbrevNormalizer:
    """
    Löst Abkürzungen im Text auf und erstellt ein Positions-Mapping.

    Eigenschaften:
    - Längere Abkürzungen werden zuerst geprüft (kein Teilersatz)
    - Kontext-abhängige Auflösung (AG = Amtsgericht oder Aktiengesellschaft)
    - Normreferenzen (§ X Abs. Y) werden NICHT verändert
    - Strukturmarker (Abs., Nr., Satz) werden NICHT verändert
    - Wörterbuch wird bei jedem Aufruf frisch geladen (kein Caching)
    """

    # Strukturmarker die NIEMALS aufgelöst werden
    PROTECTED_PATTERNS = [
        r'§\s*\d+\w*(?:\s+Abs\.\s*\d+)?(?:\s+Satz\s*\d+)?',  # § 3 Abs. 1
        r'Art\.\s*\d+',                                          # Art. 13
        r'Abs\.\s*\d+',                                          # Abs. 2
        r'Nr\.\s*\d+',                                           # Nr. 3
        r'Satz\s*\d+',                                           # Satz 1
        r'\d{4}-\d{2}-\d{2}',                                   # Datum
        r'\d+\.\s*\d+\.',                                        # 1.2. (Gliederung)
    ]

    def __init__(self, dict_path: Optional[Path] = None):
        self.dict_path = dict_path or _DEFAULT_DICT

    def normalize(self, text: str) -> NormalizationResult:
        """
        Hauptmethode: Text normalisieren.

        Args:
            text: Rohtext aus Tika-Parser

        Returns:
            NormalizationResult mit aufgelöstem Text und Positions-Mapping
        """
        if not text:
            return NormalizationResult(
                original_text=text,
                resolved_text=text,
            )

        entries = self._load_dict()
        if not entries:
            return NormalizationResult(
                original_text=text,
                resolved_text=text,
            )

        # Geschützte Bereiche markieren (Normreferenzen etc.)
        protected = self._find_protected_ranges(text)

        # Abkürzungen ersetzen und Mapping aufbauen
        result_text   = text
        replacements: list[AbbrevReplacement] = []
        offset        = 0  # kumulierter Längenunterschied

        # Längere Abkürzungen zuerst (sortiert nach Länge absteigend)
        entries_sorted = sorted(entries, key=lambda e: len(e["abbrev"]), reverse=True)

        for entry in entries_sorted:
            abbrev   = entry["abbrev"]
            resolved = entry["resolved"]
            context  = entry.get("context", [])
            label    = entry.get("label", "")

            if abbrev == resolved:
                continue

            # Alle Vorkommen im aktuellen result_text finden
            pattern = re.compile(r'\b' + re.escape(abbrev) + r'\b')

            new_text  = result_text
            new_offset = offset
            delta = len(resolved) - len(abbrev)

            for match in reversed(list(pattern.finditer(result_text))):
                match_start = match.start()
                match_end   = match.end()

                # Original-Position berechnen (vor diesem Schritt)
                orig_start = match_start - offset + (delta * len([
                    r for r in replacements
                    if r.res_start < match_start
                ]))
                # Einfacherer Ansatz: direkte Positions-Berechnung
                orig_start = self._res_to_orig(
                    match_start, replacements
                )
                orig_end   = orig_start + len(abbrev)

                # Geschützte Bereiche überspringen
                if self._is_protected(orig_start, orig_end, protected):
                    continue

                # Kontext-Prüfung
                if context:
                    ctx_start = max(0, match_start - 100)
                    ctx_end   = min(len(result_text), match_end + 100)
                    ctx_text  = result_text[ctx_start:ctx_end].lower()
                    if not any(kw.lower() in ctx_text for kw in context):
                        continue

                # Ersetzung durchführen
                new_text = (
                    new_text[:match_start]
                    + resolved
                    + new_text[match_end:]
                )

                # Positions-Mapping eintragen
                replacements.append(AbbrevReplacement(
                    abbrev=abbrev,
                    resolved=resolved,
                    orig_start=orig_start,
                    orig_end=orig_end,
                    res_start=match_start,
                    res_end=match_start + len(resolved),
                    label=label,
                ))

            result_text = new_text

        # Replacements nach orig_start sortieren
        replacements.sort(key=lambda r: r.orig_start)

        count = len(replacements)
        if count > 0:
            logger.info(
                "Abkürzungen aufgelöst",
                count=count,
                abbrevs=list({r.abbrev for r in replacements}),
            )

        return NormalizationResult(
            original_text=text,
            resolved_text=result_text,
            replacements=replacements,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Hilfsmethoden
    # ─────────────────────────────────────────────────────────────────────────

    def _load_dict(self) -> list[dict]:
        """Lädt abbrev_dict.yaml – immer frisch von Disk."""
        if not self.dict_path.exists():
            logger.warning(
                "abbrev_dict.yaml nicht gefunden",
                path=str(self.dict_path),
            )
            return []
        with open(self.dict_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("abbreviations", [])

    def _find_protected_ranges(self, text: str) -> list[tuple[int, int]]:
        """Findet alle Bereiche die nicht ersetzt werden sollen."""
        ranges = []
        for pattern in self.PROTECTED_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                ranges.append((m.start(), m.end()))
        return ranges

    def _is_protected(
        self,
        start: int,
        end: int,
        protected: list[tuple[int, int]],
    ) -> bool:
        """Prüft ob eine Position in einem geschützten Bereich liegt."""
        return any(
            ps <= start < pe or ps < end <= pe
            for ps, pe in protected
        )

    def _res_to_orig(
        self,
        res_pos: int,
        replacements: list[AbbrevReplacement],
    ) -> int:
        """
        Berechnet die Originalposition für eine Position
        im teilweise aufgelösten Text.
        """
        offset = 0
        for r in sorted(replacements, key=lambda x: x.res_start):
            if r.res_start >= res_pos:
                break
            offset += (r.orig_end - r.orig_start) - (r.res_end - r.res_start)
        return res_pos + offset
