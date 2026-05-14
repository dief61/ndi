# services/ingest/app/services/nlp/svo_extractor.py
#
# SVO-Extraktor: Extrahiert Subjekt-Verb-Objekt-Tripel aus spaCy-Docs
# und klassifiziert den Normtyp jedes Prädikats.
#
# Konfiguration via nlp_config.yaml (Sektion svo + normtypen).
# Alle Parameter können ohne Neustart geändert werden.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from app.services.nlp.nlp_processor import ChunkAnalysis, load_nlp_config


def _pos(token) -> str:
    """
    Gibt den POS-Tag zurück.
    Deutsches spaCy-Modell (de_core_news_lg) befüllt pos_ nicht immer –
    Fallback auf tag_ (feingranular) oder Heuristik via dep_.
    """
    if token.pos_ and token.pos_ != "":
        return token.pos_
    # tag_ → universales POS ableiten
    tag = token.tag_ or ""
    if tag.startswith("V"):
        return "VERB" if not tag.startswith("VMFIN") else "AUX"
    if tag.startswith("VM"):   # Modalverb
        return "AUX"
    if tag.startswith("N"):    return "NOUN"
    if tag.startswith("ART"):  return "DET"
    if tag.startswith("ADJ"):  return "ADJ"
    if tag.startswith("ADV"):  return "ADV"
    if tag.startswith("PRO"):  return "PRON"
    # Fallback via dep_
    if token.dep_ in ("sb","nsubj"):   return "NOUN"
    if token.dep_ in ("oa","obj"):     return "NOUN"
    return token.pos_ or "X"

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVOTripel:
    """Ein extrahiertes SVO-Tripel aus einem Satz."""
    subject:              Optional[str] = None
    subject_type:         str = "UNBEKANNT"
    predicate:            Optional[str] = None
    predicate_lemma:      Optional[str] = None
    object:               Optional[str] = None
    object_type:          str = "UNBEKANNT"
    context:              Optional[str] = None
    norm_type:            str = "UNKNOWN"
    norm_type_confidence: float = 0.0
    confidence:           float = 0.0
    sentence_text:        str = ""


# ─────────────────────────────────────────────────────────────────────────────
# SVOExtractor
# ─────────────────────────────────────────────────────────────────────────────

class SVOExtractor:
    """
    Extrahiert SVO-Tripel aus ChunkAnalysis-Objekten.

    Liest Konfiguration bei jedem Aufruf frisch (kein Caching) –
    Änderungen in nlp_config.yaml wirken sofort.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path

    # ─────────────────────────────────────────────────────────────────────────
    # Öffentliche API
    # ─────────────────────────────────────────────────────────────────────────

    def extract(self, analysis: ChunkAnalysis) -> list[SVOTripel]:
        """
        Extrahiert SVOs aus einer ChunkAnalysis.
        Gibt leere Liste zurück wenn kein Doc vorhanden.
        """
        if analysis.doc is None:
            return []

        cfg          = load_nlp_config(self.config_path)
        svo_cfg      = cfg.get("svo", {})
        norm_cfg     = cfg.get("normtypen", {})
        min_conf     = svo_cfg.get("min_confidence", 0.5)
        subject_deps = svo_cfg.get("subject_deps", ["sb","nsubj"])
        object_deps  = svo_cfg.get("object_deps",  ["oa","obj","obl"])

        # Stoppwort-Filter aus Config laden
        # None-Einträge aus YAML-Listen herausfiltern
        def _safe_set(lst):
            return set(w.lower() for w in (lst or []) if w is not None)
        stop_subjects = _safe_set(svo_cfg.get("stop_subjects", []))
        stop_objects  = _safe_set(svo_cfg.get("stop_objects",  []))
        min_subj_len  = svo_cfg.get("min_subject_length", 3)
        min_obj_len   = svo_cfg.get("min_object_length", 3)
        pronouns      = _safe_set(svo_cfg.get('pronouns', []))
        pron_penalty      = svo_cfg.get("pronoun_confidence_penalty", 0.3)
        # Stop-Prädikate: Substantive die fälschlich als Verb erkannt werden
        stop_predicates = _safe_set(svo_cfg.get('stop_predicates', []))

        tripel = []
        for sent in analysis.doc.sents:
            sent_tripel = self._extract_from_sentence(
                sent, svo_cfg, norm_cfg,
                subject_deps, object_deps,
            )
            for t in sent_tripel:
                if t.confidence < min_conf:
                    continue

                # ── Schicht 1: Stoppwort-Filter ───────────────────────────
                # Subjekt verwerfen wenn reines Stoppwort oder zu kurz
                if t.subject:
                    s_lower = (t.subject or '').strip().lower()
                    if s_lower in stop_subjects:
                        t.subject = None
                        t.subject_type = "UNBEKANNT"
                    elif len(s_lower) < min_subj_len:
                        t.subject = None
                        t.subject_type = "UNBEKANNT"
                    # Satz-/Absatz-Abkürzungen verwerfen: "S. 3", "Abs. 2"
                    elif re.match(r'^(s\.|abs\.|nr\.)\s*\d', s_lower):
                        t.subject = None
                        t.subject_type = "UNBEKANNT"
                    # Reine Zahlen verwerfen
                    elif re.match(r'^\d+\.?$', t.subject.strip()):
                        t.subject = None
                        t.subject_type = "UNBEKANNT"

                # Objekt verwerfen wenn reines Stoppwort oder zu kurz
                if t.object:
                    o_lower = (t.object or '').strip().lower()
                    if o_lower in stop_objects:
                        t.object = None
                        t.object_type = "UNBEKANNT"
                    elif len(o_lower) < min_obj_len:
                        t.object = None
                        t.object_type = "UNBEKANNT"
                    # Jahreszahlen und reine Zahlen verwerfen
                    elif re.match(r'^\d{1,4}$', t.object.strip()):
                        t.object = None
                        t.object_type = "UNBEKANNT"
                    # Eindeutige Adverbial-Fragmente und Ortsangaben verwerfen
                    elif re.match(
                        r'^(im übrigen|im wesentlichen|in der regel|'
                        r'im einzelfall|nach maßgabe|im rahmen|'
                        r'in niedersachsen|in deutschland|'
                        r'dafür|dazu|dabei|dagegen|daran|darauf|'
                        r'darüber|darunter|davon|dazu\.?)$',
                        o_lower,
                    ):
                        t.object = None
                        t.object_type = "UNBEKANNT"

                # ── Schicht 2: Pronomen-Konfidenz-Abzug ──────────────────
                # Pronomen-Subjekt behalten, aber Konfidenz reduzieren
                if t.subject and (t.subject or '').strip().lower() in pronouns:
                    t.subject_type = "PRONOMEN"
                    t.confidence   = max(0.0, t.confidence - pron_penalty)

                # ── Schicht 3: Stop-Prädikate ─────────────────────────
                # Substantive die fälschlich als Verb erkannt werden verwerfen
                pred_lower = (t.predicate or "").lower()
                if pred_lower and pred_lower in stop_predicates:
                    continue

                # SVO nur speichern wenn Prädikat vorhanden
                # (Subjekt oder Objekt darf None sein – Normtyp bleibt nützlich)
                if t.predicate:
                    tripel.append(t)

        return tripel

    # ─────────────────────────────────────────────────────────────────────────
    # Satz-Analyse
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_from_sentence(
        self,
        sent,
        svo_cfg:      dict,
        norm_cfg:     dict,
        subject_deps: list[str],
        object_deps:  list[str],
    ) -> list[SVOTripel]:
        """Extrahiert SVOs aus einem einzelnen Satz."""
        tripel = []

        # Root-Verb(en) finden – auch AUX (Modalverben wie "muss", "kann")
        roots = [t for t in sent if t.dep_ == "ROOT"]
        if not roots:
            return []

        for root in roots:
            # Vollständiges Verb-Cluster (Modalverb + Infinitiv)
            predicate, pred_lemma = self._get_verb_cluster(root, sent)

            # Subjekt suchen
            subject_token = self._find_dep(root, subject_deps)
            # Wenn Subjekt ein Verb ist (eingebetteter Satz wie "Wer ... hält"),
            # das echte Nomen-Subjekt eine Ebene tiefer suchen
            if subject_token is not None and _pos(subject_token) in ("VERB","AUX"):
                deeper = self._find_dep(subject_token, subject_deps)
                if deeper is not None:
                    subject_token = deeper
            subject_text = self._expand_noun_phrase(subject_token) \
                           if subject_token else None

            # Objekt suchen – bei AUX-ROOT auch beim oc-Kind suchen
            object_token = self._find_dep(root, object_deps)

            # oc/mo gefunden aber ist VERB → echtes Nomen-Objekt beim Infinitiv suchen
            if object_token is not None and _pos(object_token) in ("VERB","AUX"):
                deeper = self._find_dep(object_token, ["oa","obj","oc","pd","og"])
                if deeper is not None and _pos(deeper) not in ("VERB","AUX"):
                    object_token = deeper
                else:
                    object_token = None   # kein sinnvolles Objekt

            # ── Objekt aus Vollverb-Cluster holen ─────────────────────
            # Häufigster Fall: "muss nachweisen" – Objekt hängt am Infinitiv
            # Schritt 1: AUX-ROOT → alle VERB-Kinder durchsuchen
            if object_token is None and _pos(root) == "AUX":
                for child in root.children:
                    if _pos(child) in ("VERB", "AUX") and child.dep_ in (
                        "oc", "mo", "rc", "cj", "app"
                    ):
                        # Objekt am Vollverb suchen
                        deeper = self._find_dep(
                            child,
                            ["oa", "obj", "pd", "og", "op", "da", "obl"]
                        )
                        if deeper is not None and _pos(deeper) not in ("VERB","AUX"):
                            object_token = deeper
                            break
                        # Noch eine Ebene tiefer (z.B. eingebettete Infinitive)
                        for grandchild in child.children:
                            if _pos(grandchild) == "VERB":
                                deepest = self._find_dep(
                                    grandchild,
                                    ["oa", "obj", "pd", "og", "op", "da"]
                                )
                                if deepest is not None and                                         _pos(deepest) not in ("VERB","AUX"):
                                    object_token = deepest
                                    break
                        if object_token is not None:
                            break

            # Schritt 2: Normales VERB als ROOT – Objekt bei Hilfsverb-Kinder suchen
            if object_token is None and _pos(root) == "VERB":
                for child in root.children:
                    if _pos(child) == "AUX":
                        continue
                    if _pos(child) == "VERB" and child.dep_ in ("oc", "mo", "cj"):
                        deeper = self._find_dep(
                            child,
                            ["oa", "obj", "pd", "og", "op", "da"]
                        )
                        if deeper is not None and _pos(deeper) not in ("VERB","AUX"):
                            object_token = deeper
                            break

            # ── Passiv-Infinitiv: "ist durchzuführen", "ist anzumelden" ───
            # Beim Passiv ist das grammatische Subjekt das logische Objekt.
            # Wenn kein Objekt gefunden wurde und Prädikat ein Passiv-Infinitiv
            # ist (erkennbar an zu+Verb), Subjekt als Objekt übernehmen.
            if (object_token is None
                    and subject_token is not None
                    and predicate
                    and re.search(r'\bzu\w+en\b', predicate)):
                # Subjekt wird logisches Objekt – Subjekt auf None setzen
                object_token   = subject_token
                subject_token  = None

            object_text = self._expand_noun_phrase(object_token) \
                          if object_token else None

            # Kontext (Adverbiale, Präpositionalobjekte)
            context = self._extract_context(root, object_token, subject_token)

            # Normtyp klassifizieren
            # Erst auf Prädikat-Text (Verb-Cluster), dann auf Satz-Text
            # Das trifft gespaltene Verbkonstruktionen wie "ist ... durchzuführen"
            norm_type, norm_conf = self._classify_normtype(
                predicate + " " + sent.text, norm_cfg
            )

            # Typen bestimmen
            subject_type = self._classify_entity_type(
                subject_text,
                svo_cfg.get("subject_type_patterns", {}),
            )
            object_type = self._classify_entity_type(
                object_text,
                svo_cfg.get("object_type_patterns", {}),
            )

            # Konfidenz berechnen
            confidence = self._calc_confidence(
                subject_text, predicate, object_text, norm_conf,
            )

            tripel.append(SVOTripel(
                subject=subject_text,
                subject_type=subject_type,
                predicate=predicate,
                predicate_lemma=pred_lemma,
                object=object_text,
                object_type=object_type,
                context=context,
                norm_type=norm_type,
                norm_type_confidence=norm_conf,
                confidence=confidence,
                sentence_text=sent.text,
            ))

        return tripel

    # ─────────────────────────────────────────────────────────────────────────
    # Hilfsmethoden: Grammatik
    # ─────────────────────────────────────────────────────────────────────────

    def _find_dep(self, head, dep_labels: list[str]):
        """Findet direktes Kind mit gegebenen Dependency-Labels."""
        for child in head.children:
            if child.dep_ in dep_labels:
                return child
        return None

    def _expand_noun_phrase(self, token) -> Optional[str]:
        """
        Expandiert ein Token zur kompakten Nominalphrase.

        Regeln:
        1. Nur direkte Adjektiv/Nomen-Modifier mitnehmen (nk-Kinder)
        2. Genitivketten (og), Relativsätze und Verbphrasen abschneiden
        3. Komma und Satzzeichen stoppen die Phrase
        4. Artikel (DET/ART) am Anfang entfernen
        """
        if token is None:
            return None

        # Direkte Modifier des Kopf-Tokens sammeln (nur eine Ebene tief)
        # Erlaubte Dependency-Labels für Modifier
        ALLOWED_MOD_DEPS = {"nk", "amod", "attr", "det"}
        STOP_DEPS = {"og", "op", "sb", "oc", "mo", "rc", "relcl", "punct"}

        phrase_tokens = [token]
        for child in sorted(token.children, key=lambda x: x.i):
            # Stopp bei tiefen Abhängigkeiten oder Verben
            if child.dep_ in STOP_DEPS:
                continue
            if _pos(child) in ("VERB", "AUX"):
                continue
            if child.is_punct:
                continue
            # Nur einfache Modifier
            if child.dep_ in ALLOWED_MOD_DEPS or _pos(child) == "ADJ":
                phrase_tokens.append(child)
                # Eine Ebene tiefer: nur Adjektiv-Modifier
                for gc in child.children:
                    if _pos(gc) == "ADJ" and gc.dep_ in ALLOWED_MOD_DEPS:
                        phrase_tokens.append(gc)

        # Nach Position sortieren
        phrase_tokens = sorted(set(phrase_tokens), key=lambda t: t.i)

        # Artikel, Demonstrativpronomen (jede, alle, dieser) am Anfang entfernen
        while phrase_tokens and (
            _pos(phrase_tokens[0]) == "DET"
            or phrase_tokens[0].tag_ in ("ART", "PDAT", "PIAT", "PIS")
            or phrase_tokens[0].text.lower() in ("jede", "jeder", "jedes",
                "alle", "dieser", "diese", "dieses", "ein", "eine", "einen")
        ):
            phrase_tokens = phrase_tokens[1:]

        if not phrase_tokens:
            return token.text

        text = " ".join(t.text for t in phrase_tokens)
        return text.strip() if text.strip() else token.text

    def _get_verb_cluster(self, root, sent) -> tuple[str, str]:
        """
        Extrahiert vollständiges Verb-Cluster inkl. Modalverben.

        Deutsches spaCy-Modell:
          - Modalverb (AUX) ist oft ROOT: "muss"
          - Infinitiv hängt als oc-Kind: "besitzen"
          - Ergebnis: "muss besitzen"
        """
        verbs = [root]
        main_verb = root   # für Lemma

        # Bei AUX als ROOT: Infinitiv (oc) als Hauptverb suchen
        if _pos(root) in ("AUX", "VERB") and root.dep_ == "ROOT":
            pass  # immer weiter prüfen
        if _pos(root) == "AUX":
            for child in root.children:
                if child.dep_ in ("oc", "mo") and child.pos_ == "VERB":
                    verbs.append(child)
                    main_verb = child
                    break

        # Modalverb/Hilfsverb im übergeordneten Knoten suchen (für eingebettete Verben)
        if root.head != root and root.head.pos_ in ("VERB", "AUX"):
            verbs.insert(0, root.head)

        # Weitere Auxiliare als Kinder
        for child in root.children:
            if child.dep_ in ("mo", "aux", "auxpass") and child.pos_ == "AUX":
                if child not in verbs:
                    verbs.insert(0, child)

        verbs = sorted(set(verbs), key=lambda t: t.i)
        predicate  = " ".join(t.text for t in verbs)
        pred_lemma = main_verb.lemma_
        return predicate, pred_lemma

    def _extract_context(self, root, obj_token, subj_token) -> Optional[str]:
        """Extrahiert adverbiale Bestimmungen als Kontext."""
        context_parts = []
        for child in root.children:
            if (child.dep_ in ("mo", "op", "obl")
                    and child != obj_token
                    and child != subj_token
                    and child.pos_ not in ("PUNCT", "CCONJ")):
                phrase = " ".join(t.text for t in sorted(
                    child.subtree, key=lambda t: t.i
                ))
                if len(phrase) < 100:
                    context_parts.append(phrase)
        return "; ".join(context_parts) if context_parts else None

    # ─────────────────────────────────────────────────────────────────────────
    # Hilfsmethoden: Klassifikation
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_normtype(
        self, text: str, norm_cfg: dict
    ) -> tuple[str, float]:
        """
        Klassifiziert den Normtyp anhand der konfigurierten Regex-Muster.
        Liest Konfiguration direkt – Änderungen wirken sofort.

        Returns: (normtyp, confidence)
        """
        text_lower = text.lower()

        # Reihenfolge: MUST_NOT vor MUST prüfen (Überschneidungen!)
        priority_order = [
            "MUST_NOT", "DEF", "DEADLINE",
            "COMPETENCE", "EXCEPT", "MUST", "MAY",
        ]

        for normtyp in priority_order:
            if normtyp not in norm_cfg:
                continue
            cfg_entry = norm_cfg[normtyp]
            patterns  = cfg_entry.get("patterns", [])
            boost     = cfg_entry.get("confidence_boost", 0.0)

            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return normtyp, min(1.0, 0.85 + boost)

        return "UNKNOWN", 0.3

    def _classify_entity_type(self, text: Optional[str], patterns: dict) -> str:
        """Klassifiziert Entitätstyp via Suffix- und Exact-Matching."""
        if not text:
            return "UNBEKANNT"
        text_lower = text.lower()

        # Exact-Match zuerst
        exact = patterns.get("exact", {})
        for word, etype in exact.items():
            if word in text_lower:
                return etype

        # Suffix-Match
        suffixes = patterns.get("suffixes", {})
        for suffix, etype in suffixes.items():
            if text_lower.endswith(suffix):
                return etype

        return "UNBEKANNT"

    def _calc_confidence(
        self,
        subject: Optional[str],
        predicate: Optional[str],
        obj: Optional[str],
        norm_conf: float,
    ) -> float:
        """Berechnet Gesamt-Konfidenz des SVOs."""
        score = 0.0
        if subject:  score += 0.35
        if predicate: score += 0.30
        if obj:      score += 0.25
        score += norm_conf * 0.10
        return round(min(1.0, score), 3)
