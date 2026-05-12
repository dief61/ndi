# services/ingest/app/services/doc_type_classifier.py
#
# DocTypeClassifier: Erkennt den source_type eines Dokuments automatisch
# wenn er nicht als Parameter oder in docs.yaml definiert ist.
#
# Erkannte Typen: gesetz, verordnung, auslegung, fachkonzept,
#                 leitfaden, standard, lastenheft, text
#
# Konfiguration: docs.yaml (Sektion doc_type_detection)
# Fallback:      "text" – wird immer zugewiesen wenn kein Typ erkannt wird

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenklasse
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Ergebnis der automatischen Dokumenttyp-Erkennung."""
    source_type:  str             # erkannter Dokumenttyp
    confidence:   float           # Konfidenz 0.0 – 1.0
    auto:         bool            # True = automatisch erkannt
    signals_hit:  list[str]       # welche Signale angeschlagen haben
    reason:       str             # kurze Begründung


# ─────────────────────────────────────────────────────────────────────────────
# DocTypeClassifier
# ─────────────────────────────────────────────────────────────────────────────

class DocTypeClassifier:
    """
    Erkennt den fachlichen Dokumenttyp (source_type) automatisch.

    Wird nur aufgerufen wenn:
    1. source_type NICHT als CLI/API-Parameter übergeben wurde
    2. source_type NICHT in docs.yaml für dieses Dokument definiert ist

    Liest Konfiguration aus docs.yaml (Sektion doc_type_detection).
    Fallback ist immer "text" (Konfidenz 0.0 ist akzeptiert).
    """

    FALLBACK_TYPE = "text"

    def __init__(self, docs_yaml_path: Optional[Path] = None):
        self.docs_yaml_path = docs_yaml_path

    def _load_config(self) -> dict:
        """Lädt doc_type_detection Sektion aus docs.yaml."""
        if self.docs_yaml_path and self.docs_yaml_path.exists():
            with open(self.docs_yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("doc_type_detection", {})
        return {}

    def classify(
        self,
        filename:   str,
        title:      str,
        text:       str,
        structure,                 # DocumentStructure aus parser.py
    ) -> ClassificationResult:
        """
        Klassifiziert den Dokumenttyp.

        Args:
            filename:  Dateiname mit Endung (z.B. "NHundG.pdf")
            title:     Dokumenttitel (aus Metadaten oder Dateiname)
            text:      Volltext des Dokuments (erste 5000 Zeichen reichen)
            structure: DocumentStructure aus TikaParser

        Returns:
            ClassificationResult mit Typ, Konfidenz und Begründung
        """
        cfg = self._load_config()

        if not cfg.get("enabled", True):
            return ClassificationResult(
                source_type=self.FALLBACK_TYPE,
                confidence=0.0,
                auto=True,
                signals_hit=[],
                reason="Automatische Erkennung deaktiviert",
            )

        global_min = cfg.get("global_min_confidence", 0.70)
        typen_cfg  = cfg.get("typen", {})
        text_lower = text[:5000].lower()
        text_raw   = text[:5000]

        # Alle Typen außer "text" prüfen
        candidates = []

        for typ, typ_cfg in typen_cfg.items():
            if typ == self.FALLBACK_TYPE:
                continue

            min_conf = float(typ_cfg.get("min_confidence", global_min))
            signale  = typ_cfg.get("signale", {})

            conf, hits = self._score_type(
                typ=typ,
                signale=signale,
                filename=filename,
                title=title,
                text_lower=text_lower,
                text_raw=text_raw,
                structure=structure,
            )

            if conf >= min_conf and hits:
                candidates.append((typ, conf, hits, min_conf))

        if not candidates:
            return ClassificationResult(
                source_type=self.FALLBACK_TYPE,
                confidence=0.0,
                auto=True,
                signals_hit=[],
                reason="Kein Typ erkannt – Fallback auf 'text'",
            )

        # Besten Kandidaten wählen (höchste Konfidenz)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_typ, best_conf, best_hits, _ = candidates[0]

        reason = (
            f"Automatisch erkannt: {len(best_hits)} Signal(e) "
            f"[{', '.join(best_hits[:3])}]"
        )

        logger.info(
            "Dokumenttyp automatisch erkannt",
            filename=filename,
            source_type=best_typ,
            confidence=round(best_conf, 3),
            signals=best_hits[:5],
        )

        return ClassificationResult(
            source_type=best_typ,
            confidence=round(best_conf, 3),
            auto=True,
            signals_hit=best_hits,
            reason=reason,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring-Logik
    # ─────────────────────────────────────────────────────────────────────────

    def _score_type(
        self,
        typ:        str,
        signale:    dict,
        filename:   str,
        title:      str,
        text_lower: str,
        text_raw:   str,
        structure,
    ) -> tuple[float, list[str]]:
        """
        Berechnet Konfidenz-Score für einen Dokumenttyp.
        Gibt (konfidenz, treffer_liste) zurück.
        """
        score = 0.0
        hits  = []

        # ── Titel-Muster (höchste Gewichtung: 0.40 je Treffer, max 0.40) ──
        titel_muster = signale.get("titel_muster", [])
        titel_hits = 0
        for pattern in titel_muster:
            try:
                if re.search(pattern, title or "", re.IGNORECASE):
                    hits.append(f"Titel:{pattern[:20]}")
                    titel_hits += 1
            except re.error:
                pass
        if titel_hits > 0:
            score += min(0.40, titel_hits * 0.40)

        # ── Dateiname-Muster (Gewichtung: 0.30 je Treffer, max 0.30) ──────
        datei_muster = signale.get("dateiname_muster", [])
        datei_hits = 0
        for pattern in datei_muster:
            try:
                if re.search(pattern, filename or "", re.IGNORECASE):
                    hits.append(f"Datei:{pattern[:20]}")
                    datei_hits += 1
            except re.error:
                pass
        if datei_hits > 0:
            score += min(0.30, datei_hits * 0.30)

        # ── Volltext-Signale (Gewichtung: 0.05 je Treffer, max 0.20) ───────
        volltext_signale = signale.get("volltext_signale", [])
        vt_hits = 0
        for signal in volltext_signale:
            try:
                if re.search(signal, text_raw, re.IGNORECASE):
                    hits.append(f"Text:{signal[:20]}")
                    vt_hits += 1
            except re.error:
                pass
        if vt_hits > 0:
            score += min(0.20, vt_hits * 0.05)

        # ── Struktur-Signale (Gewichtung: 0.10 je Treffer) ──────────────────
        struktur = signale.get("struktur_signale", {})

        if struktur and structure is not None:
            # Mindestanzahl §-Paragraphen
            para_min = struktur.get("paragraph_min")
            if para_min and hasattr(structure, "paragraph_count"):
                if structure.paragraph_count >= para_min:
                    hits.append(f"Struktur:paragraph_min={para_min}")
                    score += 0.10

            # Maximale §-Paragraphen (für Nicht-Rechtstexte)
            para_max = struktur.get("paragraph_max")
            if para_max is not None and hasattr(structure, "paragraph_count"):
                if structure.paragraph_count <= para_max:
                    hits.append(f"Struktur:paragraph_max={para_max}")
                    score += 0.05

            # Normstruktur (Abs., Satz, Nr.)
            if struktur.get("hat_normstruktur"):
                norm_patterns = [r"\bAbs\.\s*\d", r"\bSatz\s*\d", r"\bNr\.\s*\d"]
                norm_hits = sum(
                    1 for p in norm_patterns
                    if re.search(p, text_raw)
                )
                if norm_hits >= 2:
                    hits.append("Struktur:normstruktur")
                    score += 0.10

            # Inhaltsverzeichnis
            if struktur.get("hat_inhaltsverzeichnis"):
                if hasattr(structure, "has_toc") and structure.has_toc:
                    hits.append("Struktur:toc")
                    score += 0.10

            # Anforderungs-IDs (A-001, B-002)
            if struktur.get("hat_anforderungs_ids"):
                if re.search(r"\b[A-Z]-\d{3}\b", text_raw):
                    hits.append("Struktur:anforderungs_ids")
                    score += 0.10

            # Fragen-Quote (für auslegung)
            fragen_min = struktur.get("fragen_quote_min")
            if fragen_min and len(text_lower) > 100:
                fragesaetze = len(re.findall(r"\?", text_raw))
                saetze      = max(1, len(re.findall(r"[.!?]", text_raw)))
                quote       = fragesaetze / saetze
                if quote >= fragen_min:
                    hits.append(f"Struktur:fragen={quote:.2f}")
                    score += 0.10

        return round(min(1.0, score), 3), hits
