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

    def extract(
        self,
        analysis:     ChunkAnalysis,
        doc_filename: str = "",   # Dateiname für ner_extensions in docs.yaml
    ) -> list[NEREntity]:
        """
        Extrahiert NER-Entitäten aus einer ChunkAnalysis.

        Zweischicht-NER:
          Schicht 1: Agnostische Basis (nlp_config.yaml → agnostische_datenobjekte)
          Schicht 2: Register-spezifisch (docs.yaml → ner_extensions)

        doc_filename: Dateiname des Quelldokuments (z.B. "Hundegesetz.pdf")
                      für den Zugriff auf ner_extensions in docs.yaml.
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

        # ── Ansatz 3: Konfidenz-Filter ────────────────────────────────────
        entities = [e for e in entities if e.confidence >= min_conf]

        # ── Ansatz 1: Blacklist ───────────────────────────────────────────
        blacklist = {
            w.lower()
            for w in ner_cfg.get("blacklist", [])
        }
        if blacklist:
            entities = self._apply_blacklist(entities, blacklist)

        # ── Ansatz 2: Label-Korrekturen ───────────────────────────────────
        label_corrections = {
            k.lower(): v
            for k, v in ner_cfg.get("label_corrections", {}).items()
        }
        if label_corrections:
            entities = self._apply_label_corrections(entities, label_corrections)

        # ── Zweischicht-NER: agnostisch + register-spezifisch ───────────────
        # Schicht 1: agnostische Datenobjekte aus nlp_config.yaml
        agnostische = {
            w.lower()
            for w in ner_cfg.get("agnostische_datenobjekte", [])
            if w is not None
        }
        # Schicht 2: register-spezifische Erweiterungen aus docs.yaml
        doc_extensions = self._load_doc_extensions(doc_filename)

        if agnostische or doc_extensions:
            entities = self._apply_zweischicht_ner(
                entities, agnostische, doc_extensions
            )

        # ── Ansatz 5: §-Normreferenzen filtern ──────────────────────────
        # §-Nummern allein (§ 8) oder mit Abs./Satz (§ 10 Abs. 4) sind
        # Normreferenzen, keine Gesetzes-Entitäten.
        # Nur §-Entitäten MIT Gesetzesname behalten (§ 117 ... Versicherungsvertragsgesetz).
        # Konfigurierbar über ner.filter_norm_references (Standard: true)
        if ner_cfg.get("filter_norm_references", True):
            entities = self._filter_norm_references(entities)

        # ── Ansatz 4: Kontext-Validierung ─────────────────────────────────
        ctx_cfg = ner_cfg.get("context_validation", {})
        if ctx_cfg.get("enabled", False):
            entities = self._apply_context_validation(
                entities, full_text, ctx_cfg
            )

        # Deduplizieren
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

    # Regex: §-Nummer + optionaler Abs./Satz – aber KEIN Gesetzesname
    _NORMREF_ONLY = re.compile(
        r'^§\s*\d+\w*'               # § + Nummer
        r'(?:\s+Abs\.\s*\d+)?'       # optional: Abs. X
        r'(?:\s+Satz\s*\d+)?'        # optional: Satz X
        r'(?:\s+Nr\.\s*\d+)?'        # optional: Nr. X
        r'\s*$',                      # Ende – kein Gesetzesname dahinter
        re.IGNORECASE,
    )

    def _filter_norm_references(
        self,
        entities: list[NEREntity],
    ) -> list[NEREntity]:
        """
        Ansatz 5: §-Nummern ohne Gesetzesname aus GESETZ-Entitäten entfernen.

        Behalten:  § 117 Abs. 2 Satz 1 des Versicherungsvertragsgesetzes
        Verwerfen: § 8
        Verwerfen: § 10 Abs. 4
        Verwerfen: § 9 Satz 3

        Nicht-GESETZ-Entitäten werden nicht berührt.
        """
        result = []
        for e in entities:
            if e.label == "GESETZ" and self._NORMREF_ONLY.match(e.text.strip()):
                # Nur §-Nummer ohne Gesetzesname → verwerfen
                logger.debug("Normreferenz gefiltert", text=e.text)
            else:
                result.append(e)
        return result

    def _apply_blacklist(
        self,
        entities: list[NEREntity],
        blacklist: set[str],
    ) -> list[NEREntity]:
        """
        Ansatz 1: Entfernt Entitäten deren Text (lowercase) in der Blacklist steht.
        Groß-/Kleinschreibung wird ignoriert.
        """
        result = []
        for e in entities:
            if e.text.lower().strip() in blacklist:
                pass   # verworfen
            else:
                result.append(e)
        return result

    def _load_doc_extensions(self, doc_filename: str) -> dict:
        """
        Lädt register-spezifische NER-Erweiterungen aus docs.yaml.
        Returns: {"DATENOBJEKT": set(...), "ROLLE": set(...), ...}
        """
        if not doc_filename:
            return {}
        try:
            from pathlib import Path
            import yaml as _yaml
            docs_path = (self.config_path or Path(__file__).parents[3]).parent                         if self.config_path else Path(__file__).parents[3]
            # docs.yaml suchen
            candidates = [
                docs_path / "docs.yaml",
                docs_path.parent / "docs.yaml",
                Path(__file__).parents[3] / "docs.yaml",
            ]
            docs_yaml = next((p for p in candidates if p.exists()), None)
            if not docs_yaml:
                return {}
            docs = _yaml.safe_load(docs_yaml.read_text(encoding="utf-8")) or {}
            entry = docs.get(doc_filename, {})
            extensions = entry.get("ner_extensions", {})
            return {
                label: {v.lower() for v in values if v}
                for label, values in extensions.items()
            }
        except Exception as e:
            logger.debug("ner_extensions Laden fehlgeschlagen",
                         doc=doc_filename, error=str(e))
            return {}

    def _apply_zweischicht_ner(
        self,
        entities:      list,
        agnostische:   set,
        doc_extensions: dict,
    ) -> list:
        """
        Zweischicht-NER:
          Schicht 1: Agnostische Basis (nlp_config.yaml)
          Schicht 2: Register-spezifisch (docs.yaml)

        Priorität: Explizites Label > Agnostisch > Bestehendes Label
        """
        result = []
        for e in entities:
            text_lower = (e.text or "").lower()

            # Schicht 2: Register-spezifisch (höchste Priorität)
            matched = False
            import dataclasses as _dc
            for label, terms in doc_extensions.items():
                if text_lower in terms:
                    if e.label != label:
                        logger.debug(
                            "NER Register-Extension",
                            text=e.text,
                            alt=e.label,
                            neu=label,
                        )
                        e = _dc.replace(e, label=label)
                    matched = True
                    break

            # Schicht 1: Agnostisch (wenn nicht durch Schicht 2 erfasst)
            if not matched and text_lower in agnostische:
                if e.label not in ("DATENOBJEKT",):
                    logger.debug(
                        "NER Agnostisch-Klassifikation",
                        text=e.text,
                        alt=e.label,
                    )
                    e = _dc.replace(e, label="DATENOBJEKT")

            result.append(e)
        return result

    def _apply_label_corrections(
        self,
        entities: list[NEREntity],
        corrections: dict[str, str],
    ) -> list[NEREntity]:
        """
        Ansatz 2: Korrigiert das Label einer Entität wenn ihr Text
        in label_corrections gefunden wird.
        """
        for e in entities:
            new_label = corrections.get(e.text.lower().strip())
            if new_label:
                e.label = new_label
        return entities

    def _apply_context_validation(
        self,
        entities: list[NEREntity],
        full_text: str,
        ctx_cfg: dict,
    ) -> list[NEREntity]:
        """
        Ansatz 4: Prüft ob im Kontext (±window Zeichen) mindestens eines
        der required_context-Wörter vorkommt.
        Bei Fehlschlag: Label auf fallback_label setzen statt verwerfen.
        """
        window     = ctx_cfg.get("context_window", 80)
        label_cfgs = ctx_cfg

        result = []
        text_lower = full_text.lower()

        for e in entities:
            label_rule = label_cfgs.get(e.label, {})
            required   = label_rule.get("required_context", [])
            fallback   = label_rule.get("fallback_label", "SONSTIGE")

            # Keine Kontext-Prüfung für dieses Label
            if not required:
                result.append(e)
                continue

            # Kontext-Fenster um die Entität
            start    = max(0, e.start_char - window)
            end      = min(len(text_lower), e.end_char + window)
            context  = text_lower[start:end]

            # Mindestens ein Signalwort im Kontext?
            found = any(kw.lower() in context for kw in required)

            if found:
                result.append(e)
            else:
                # Nicht verwerfen – auf Fallback-Label herabstufen
                e.label = fallback
                result.append(e)

        return result

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
