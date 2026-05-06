# services/ingest/app/services/nlp/ner_extractor.py
#
# NER-Extraktor: Zwei-Stufen-Architektur
#
# Stufe 1: Regelbasiert (Suffixe, Regex)    – immer aktiv, sofort verfügbar
# Stufe 2: flair/ner-german-legal           – optional, aktivierbar via Config
#
# Das Flair-Modell ist auf dem German LER-Datensatz trainiert:
# 67.000 Sätze aus deutschen Bundesgerichtsentscheidungen,
# 19 feingranulare juristische Entitätsklassen.
#
# LER-Klassen → MNR-Klassen Mapping:
#   GS  Gesetz/Norm                → GESETZ
#   RS  Rechtsprechung             → GESETZ
#   EUN Europäische Norm           → GESETZ
#   VS  Verwaltungsvorschrift      → GESETZ
#   VO  Verordnung                 → GESETZ
#   VT  Vertrag                    → GESETZ
#   GRT Gericht                    → BEHÖRDE
#   INN Institution                → BEHÖRDE
#   ORG Organisation               → BEHÖRDE
#   UN  Unternehmen                → BEHÖRDE
#   PER Person                     → ROLLE
#   RR  Richter                    → ROLLE
#   AN  Anwalt                     → ROLLE
#   LD  Land                       → ORT
#   ST  Stadt                      → ORT
#   STR Straße                     → ORT
#   LDS Landschaft                 → ORT
#   LIT Literatur                  → SONSTIGE
#   MRK Marke                      → SONSTIGE

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from app.services.nlp.nlp_processor import ChunkAnalysis, load_nlp_config

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# LER → MNR Label-Mapping
# ─────────────────────────────────────────────────────────────────────────────

LER_TO_MNR = {
    # Juristische Normen und Regelwerke → GESETZ
    "GS":  "GESETZ",    # Gesetz / Norm
    "RS":  "GESETZ",    # Rechtsprechung (Urteilsreferenz)
    "EUN": "GESETZ",    # Europäische Norm
    "VS":  "GESETZ",    # Verwaltungsvorschrift
    "VO":  "GESETZ",    # Verordnung
    "VT":  "GESETZ",    # Vertrag
    # Institutionen / Behörden → BEHÖRDE
    "GRT": "BEHÖRDE",   # Gericht
    "INN": "BEHÖRDE",   # Institution
    "ORG": "BEHÖRDE",   # Organisation
    "UN":  "BEHÖRDE",   # Unternehmen
    # Personen / Rollen → ROLLE
    "PER": "ROLLE",     # Person
    "RR":  "ROLLE",     # Richter
    "AN":  "ROLLE",     # Anwalt
    # Orte → ORT
    "LD":  "ORT",       # Land / Bundesland
    "ST":  "ORT",       # Stadt
    "STR": "ORT",       # Straße
    "LDS": "ORT",       # Landschaft
    # Sonstige
    "LIT": "SONSTIGE",  # Literatur
    "MRK": "SONSTIGE",  # Marke
}


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NEREntity:
    """Eine erkannte benannte Entität."""
    text:        str
    label:       str          # MNR-Label
    label_orig:  str = ""     # Original LER-Label (bei Flair)
    start_char:  int = 0
    end_char:    int = 0
    confidence:  float = 0.0
    source:      str = "rule" # rule | flair


# ─────────────────────────────────────────────────────────────────────────────
# Flair-Modell-Singleton
# ─────────────────────────────────────────────────────────────────────────────

class _FlairModelCache:
    """Lädt das Flair-Modell einmalig und cached es."""
    _model = None
    _model_id: Optional[str] = None

    @classmethod
    def get(cls, model_id: str):
        if cls._model is None or cls._model_id != model_id:
            logger.info("Lade Flair Legal NER-Modell", model=model_id)
            try:
                from flair.models import SequenceTagger
                cls._model    = SequenceTagger.load(model_id)
                cls._model_id = model_id
                logger.info("Flair-Modell geladen", model=model_id)
            except Exception as e:
                logger.error("Flair-Modell konnte nicht geladen werden",
                             model=model_id, error=str(e))
                raise
        return cls._model


# ─────────────────────────────────────────────────────────────────────────────
# NERExtractor
# ─────────────────────────────────────────────────────────────────────────────

class NERExtractor:
    """
    Zwei-Stufen NER-Extraktor.

    Stufe 1 (regelbasiert): Immer aktiv. Schnell, domänenspezifisch,
    ideal für Verwaltungs-spezifische Begriffe wie Behördennamen
    mit Suffix "-behörde" die im LER-Datensatz selten vorkommen.

    Stufe 2 (Flair Legal NER): Optional via nlp_config.yaml.
    Erkennt 19 feingranulare juristische Entitätsklassen aus
    deutschen Gerichtsentscheidungen.

    Kombinationsstrategien:
      merge      → Beide Stufen, Duplikate nach Konfidenz bereinigt
      flair_only → Nur Flair
      rules_only → Nur Regelbasiert
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path

    def extract(self, analysis: ChunkAnalysis) -> list[NEREntity]:
        """
        Extrahiert NER-Entitäten aus einer ChunkAnalysis.
        Liest Konfiguration frisch von Disk bei jedem Aufruf.
        """
        cfg      = load_nlp_config(self.config_path)
        ner_cfg  = cfg.get("ner", {})
        strategy = ner_cfg.get("combination_strategy", "merge")
        min_conf = ner_cfg.get("min_confidence", 0.6)

        # Vollständigen Text rekonstruieren
        if analysis.doc is not None:
            full_text = analysis.doc.text
        else:
            full_text = " ".join(s.text for s in analysis.sentences)

        rule_entities:  list[NEREntity] = []
        flair_entities: list[NEREntity] = []

        # Stufe 1: Regelbasiert
        if strategy in ("merge", "rules_only") and ner_cfg.get("rules_enabled", True):
            rule_entities = self._extract_rule_based(
                full_text,
                ner_cfg.get("patterns", {}),
            )

        # Stufe 2: Flair Legal NER
        if strategy in ("merge", "flair_only") and ner_cfg.get("flair_enabled", False):
            flair_model_id = ner_cfg.get("flair_model", "flair/ner-german-legal")
            flair_min      = ner_cfg.get("flair_min_confidence", 0.7)
            try:
                flair_entities = self._extract_flair(
                    full_text, flair_model_id, flair_min
                )
            except Exception as e:
                logger.warning("Flair-Extraktion fehlgeschlagen, nur Regelbasiert",
                               error=str(e))

        # Kombinieren
        if strategy == "merge":
            entities = self._merge(rule_entities, flair_entities)
        elif strategy == "flair_only":
            entities = flair_entities
        else:
            entities = rule_entities

        # Mindest-Konfidenz filtern und deduplizieren
        entities = [e for e in entities if e.confidence >= min_conf]
        entities = self._deduplicate(entities)

        return entities

    # ─────────────────────────────────────────────────────────────────────────
    # Stufe 1: Regelbasierte Extraktion
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_rule_based(
        self, text: str, patterns: dict
    ) -> list[NEREntity]:
        """
        Erkennt Entitäten via Suffix-, Exact- und Regex-Matching.
        Konfiguration aus nlp_config.yaml Sektion ner.patterns.
        """
        entities = []
        text_lower = text.lower()

        for label, label_patterns in patterns.items():

            # Regex-Patterns (FRIST, GESETZ)
            for pattern in label_patterns.get("regex", []):
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(NEREntity(
                        text=match.group(),
                        label=label,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.90,
                        source="rule",
                    ))

            # Exact-Match
            for word in label_patterns.get("exact", []):
                start = 0
                while True:
                    pos = text_lower.find(word.lower(), start)
                    if pos == -1:
                        break
                    if self._is_word_boundary(text_lower, pos, pos + len(word)):
                        entities.append(NEREntity(
                            text=text[pos:pos + len(word)],
                            label=label,
                            start_char=pos,
                            end_char=pos + len(word),
                            confidence=0.85,
                            source="rule",
                        ))
                    start = pos + 1

            # Suffix-Match
            for suffix in label_patterns.get("suffixes", []):
                pattern = r'\b\w+' + re.escape(suffix) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(NEREntity(
                        text=match.group(),
                        label=label,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.80,
                        source="rule",
                    ))

        return entities

    # ─────────────────────────────────────────────────────────────────────────
    # Stufe 2: Flair Legal NER
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_flair(
        self,
        text: str,
        model_id: str,
        min_confidence: float,
    ) -> list[NEREntity]:
        """
        Erkennt Entitäten mit dem Flair Legal NER-Modell.
        Mappt LER-Labels (19 Klassen) auf MNR-Labels (6 Klassen).
        """
        from flair.data import Sentence

        model    = _FlairModelCache.get(model_id)
        sentence = Sentence(text)
        model.predict(sentence)

        entities = []
        for span in sentence.get_spans("ner"):
            ler_label = span.get_label("ner").value
            score     = span.get_label("ner").score

            if score < min_confidence:
                continue

            # LER → MNR mapping
            mnr_label = LER_TO_MNR.get(ler_label, "SONSTIGE")

            # Zeichenposition im Originaltext bestimmen
            start_char = text.find(span.text)
            end_char   = start_char + len(span.text) if start_char >= 0 else 0

            entities.append(NEREntity(
                text=span.text,
                label=mnr_label,
                label_orig=ler_label,
                start_char=start_char,
                end_char=end_char,
                confidence=round(score, 3),
                source="flair",
            ))

        return entities

    # ─────────────────────────────────────────────────────────────────────────
    # Kombinieren & Bereinigen
    # ─────────────────────────────────────────────────────────────────────────

    def _merge(
        self,
        rule_entities:  list[NEREntity],
        flair_entities: list[NEREntity],
    ) -> list[NEREntity]:
        """
        Kombiniert Regel- und Flair-Ergebnisse.
        Bei Überschneidungen gewinnt die höhere Konfidenz.
        Flair-Ergebnisse ergänzen Lücken die Regeln nicht abdecken.
        """
        merged = list(flair_entities)  # Flair als Basis

        for rule_ent in rule_entities:
            # Prüfen ob Flair diese Entität bereits erkannt hat
            overlap = any(
                abs(rule_ent.start_char - fe.start_char) < 5
                and rule_ent.label == fe.label
                for fe in flair_entities
            )
            if not overlap:
                merged.append(rule_ent)

        return merged

    def _deduplicate(self, entities: list[NEREntity]) -> list[NEREntity]:
        """Entfernt Duplikate – behält höchste Konfidenz je Text+Label."""
        seen: dict[tuple, NEREntity] = {}
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e
        return list(seen.values())

    def _is_word_boundary(self, text: str, start: int, end: int) -> bool:
        before = start == 0 or not text[start - 1].isalnum()
        after  = end == len(text) or not text[end].isalnum()
        return before and after
