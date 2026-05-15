# services/ingest/app/services/rag/query_transformer.py
#
# M3 – Schritt 1.1: Query-Analyse & -Transformation
#
# Aufgaben:
#   1. Query-Typ klassifizieren (NORM | ENTITY | IM | GENERAL)
#   2. Norm-Referenz extrahieren (§ 3 Abs. 1 NHundG → Direkt-Lookup)
#   3. Query transformieren (Direct / HyDE / Step-Back / Multi-Query)
#   4. Metadaten-Filter ableiten (doc_class, source_type, im_signals)
#
# Output: QueryBundle mit allen Suchvektoren und Filtern für den Retriever

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger()

_RAG_CFG_PATH = Path(__file__).parents[3] / "rag_config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Datenstrukturen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryVector:
    """Ein Suchvektor mit Strategie und Gewicht."""
    text:      str
    strategie: str    # direct | hyde | step_back | multi_query
    gewicht:   float  # 0.0 – 1.0


@dataclass
class QueryBundle:
    """
    Standardisierte Ausgabe des QueryTransformers.
    Eingabe für den Retriever (Schritt 1.2).
    """
    original_query:   str
    query_typ:        str                  # NORM | ENTITY | IM | GENERAL
    norm_reference:   Optional[str]        # "§ 3 Abs. 1 NHundG" oder None
    vektoren:         list[QueryVector]    # ein oder mehrere Suchvektoren
    metadata_filter:  dict                 # pgvector WHERE-Bedingungen
    im_filter:        bool                 # True = im_signals bevorzugen
    direktlookup:     bool                 # True = Vektorsuche überspringen
    debug:            dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Regex-Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Vollständige Norm-Referenz: "§ 3 Abs. 1 Satz 2 NHundG"
_NORM_REF_RE = re.compile(
    r'§\s*(?P<para>\d+[a-z]?)'
    r'(?:\s+Abs\.\s*(?P<abs>\d+))?'
    r'(?:\s+Satz\s*(?P<satz>\d+))?'
    r'(?:\s+Nr\.\s*(?P<nr>\d+))?'
    r'(?:\s+(?P<gesetz>[A-ZÄÖÜ][a-zA-ZÄÖÜäöü]{2,}G\b))?',
    re.IGNORECASE,
)

# Norm-Typ-Signale
_NORM_SIGNALE_RE = re.compile(
    r'((?<!\w)§\s*\d|abs\.\s*\d|absatz\s+\d|satz\s+\d|'
    r'\bmuss\b|\bdarf\b|\bsoll\b|\bbedarf\b|'
    r'\bist\s+zu\b|\bhat\s+zu\b|\bunverzüglich\b|\bbinnen\b|\bspätestens\b)',
    re.IGNORECASE,
)

# IM-Signale (Informationsmodell-Anfragen)
_IM_SIGNALE_RE = re.compile(
    r'\b(entit[aä]t|tabelle|attribut|spalte|datenbank|feld|'
    r'schema|datenbankfeld|persistier|datenmodell)\b',
    re.IGNORECASE,
)

# Entity/Zuständigkeits-Signale
_ENTITY_SIGNALE_RE = re.compile(
    r'\b(zust[äa]ndig|wer\s+ist|welche\s+beh[öo]rde|'
    r'welche\s+rolle|wer\s+darf|wer\s+muss)\b',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# QueryTransformer
# ─────────────────────────────────────────────────────────────────────────────

class QueryTransformer:
    """
    Transformiert einen Rohtext-Query in ein QueryBundle
    mit mehreren Suchvektoren und Metadaten-Filtern.

    Wird von der RAG-Pipeline als erster Schritt aufgerufen.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._cfg_path = config_path or _RAG_CFG_PATH
        self._gateway  = None
        self._suite    = None

    # ── Konfiguration ─────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            with open(self._cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("rag_config.yaml nicht gefunden – Defaults verwendet")
            return {}

    def _get_qt_cfg(self) -> dict:
        return self._load_config().get("query_transformation", {})

    # ── Lazy-Loading LLM-Komponenten ──────────────────────────────────────────

    def _get_gateway(self):
        if self._gateway is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[3]))
            from llm_gateway.gateway import LLMGateway
            self._gateway = LLMGateway()
        return self._gateway

    def _get_suite(self):
        if self._suite is None:
            from llm_gateway.prompt_suite import PromptSuite
            self._suite = PromptSuite()
        return self._suite

    # ── Aufgabe 1: Query-Typ klassifizieren ───────────────────────────────────

    def classify_query_typ(self, query: str) -> str:
        """
        Klassifiziert den Query-Typ anhand von Signal-Mustern.

        Kaskade (erste Bedingung gewinnt):
          NORM    → gesetzliche Signale (§, Modalverben, Fristen)
          IM      → Informationsmodell-Anfragen (Tabelle, Attribut, Entität)
          ENTITY  → Zuständigkeits- und Akteur-Anfragen
          GENERAL → Fallback
        """
        q = query.lower()

        # NORM: §-Zeichen oder juristische Modalverben
        if _NORM_SIGNALE_RE.search(q):
            return "NORM"

        # IM: Informationsmodell-Begriffe
        if _IM_SIGNALE_RE.search(q):
            return "IM"

        # ENTITY: Zuständigkeit / Akteure
        if _ENTITY_SIGNALE_RE.search(q):
            return "ENTITY"

        return "GENERAL"

    # ── Aufgabe 2: Norm-Referenz extrahieren ──────────────────────────────────

    def extract_norm_reference(self, query: str) -> Optional[str]:
        """
        Extrahiert eine kanonische Norm-Referenz aus dem Query.

        Beispiele:
          "§ 3 Abs. 1 NHundG"     → "§ 3 Abs. 1 NHundG"
          "§ 9 Satz 1"             → "§ 9 Satz 1"
          "Was muss ein Halter...?" → None

        Bei Fund: Direktlookup im Retriever möglich (kein Vektorvergleich).
        """
        m = _NORM_REF_RE.search(query)
        if not m:
            return None

        # Referenz rekonstruieren
        parts = [f"§ {m.group('para')}"]
        if m.group("abs"):
            parts.append(f"Abs. {m.group('abs')}")
        if m.group("satz"):
            parts.append(f"Satz {m.group('satz')}")
        if m.group("nr"):
            parts.append(f"Nr. {m.group('nr')}")
        if m.group("gesetz"):
            parts.append(m.group("gesetz"))

        ref = " ".join(parts)
        logger.debug("Norm-Referenz erkannt", ref=ref)
        return ref

    # ── Aufgabe 3a: Direct (immer aktiv) ─────────────────────────────────────

    def _vector_direct(self, query: str) -> QueryVector:
        return QueryVector(text=query, strategie="direct", gewicht=1.0)

    # ── Aufgabe 3b: HyDE ──────────────────────────────────────────────────────

    async def _vector_hyde(self, query: str, qt_cfg: dict) -> Optional[QueryVector]:
        """
        Generiert einen hypothetischen Normtext via LLM.
        Der Embedding-Vektor dieses Texts trifft echte Chunks besser
        als der kurze Query-Text.
        """
        hyde_cfg  = qt_cfg.get("hyde", {})
        if not hyde_cfg.get("enabled", True):
            return None

        prompt_key = hyde_cfg.get("prompt_key", "ps_hyde")
        suite      = self._get_suite()
        gw         = self._get_gateway()

        if not suite.exists(prompt_key):
            logger.warning("HyDE-Prompt nicht gefunden", key=prompt_key)
            return None

        try:
            system = suite.get_system(prompt_key)
            user   = suite.render_user(prompt_key, query=query)

            result = await gw.complete(
                system_prompt=system,
                user_prompt=user,
                json_mode=False,
            )

            if not result.erfolg or not result.content.strip():
                return None

            hypothese = result.content.strip()
            logger.debug("HyDE-Hypothese erzeugt",
                         query_start=query[:40],
                         hypothese_start=hypothese[:60])

            return QueryVector(
                text=hypothese,
                strategie="hyde",
                gewicht=hyde_cfg.get("gewicht", 0.8),
            )

        except Exception as e:
            logger.warning("HyDE-Fehler", error=str(e))
            return None

    # ── Aufgabe 3c: Step-Back ─────────────────────────────────────────────────

    async def _vector_step_back(
        self,
        query:    str,
        qt_cfg:   dict,
        query_typ: str,
    ) -> Optional[QueryVector]:
        """
        Abstrahiert den Query auf die übergeordnete Normenebene.
        Sinnvoll bei spezifischen §-Anfragen.
        """
        sb_cfg    = qt_cfg.get("step_back", {})
        if not sb_cfg.get("enabled", True):
            return None

        # Nur bei bestimmten Query-Typen
        nur_bei = sb_cfg.get("nur_bei_typen", ["NORM"])
        if query_typ not in nur_bei:
            return None

        prompt_key = sb_cfg.get("prompt_key", "ps_step_back")
        suite      = self._get_suite()
        gw         = self._get_gateway()

        if not suite.exists(prompt_key):
            logger.warning("Step-Back-Prompt nicht gefunden", key=prompt_key)
            return None

        try:
            result = await gw.complete(
                system_prompt=suite.get_system(prompt_key),
                user_prompt=suite.render_user(prompt_key, query=query),
                json_mode=False,
            )

            if not result.erfolg or not result.content.strip():
                return None

            abstrahiert = result.content.strip()
            logger.debug("Step-Back erzeugt",
                         original=query[:40],
                         abstrakt=abstrahiert[:60])

            return QueryVector(
                text=abstrahiert,
                strategie="step_back",
                gewicht=sb_cfg.get("gewicht", 0.6),
            )

        except Exception as e:
            logger.warning("Step-Back-Fehler", error=str(e))
            return None

    # ── Aufgabe 3d: Multi-Query ───────────────────────────────────────────────

    async def _vectors_multi_query(
        self,
        query:    str,
        qt_cfg:   dict,
        query_typ: str,
    ) -> list[QueryVector]:
        """
        Erzeugt mehrere Query-Varianten via LLM.
        Jede Variante deckt einen anderen Aspekt ab.
        """
        mq_cfg = qt_cfg.get("multi_query", {})
        if not mq_cfg.get("enabled", False):
            return []

        nur_bei = mq_cfg.get("nur_bei_typen", ["ENTITY", "GENERAL"])
        if query_typ not in nur_bei:
            return []

        prompt_key = mq_cfg.get("prompt_key", "ps_multi_query")
        count      = mq_cfg.get("count", 3)
        suite      = self._get_suite()
        gw         = self._get_gateway()

        if not suite.exists(prompt_key):
            return []

        try:
            result = await gw.complete(
                system_prompt=suite.get_system(prompt_key),
                user_prompt=suite.render_user(
                    prompt_key, query=query, count=count),
                json_mode=True,
            )

            if not result.erfolg or not result.parsed:
                return []

            varianten = result.parsed
            if not isinstance(varianten, list):
                return []

            return [
                QueryVector(
                    text=str(v),
                    strategie="multi_query",
                    gewicht=mq_cfg.get("gewicht", 0.7),
                )
                for v in varianten
                if isinstance(v, str) and v.strip()
            ]

        except Exception as e:
            logger.warning("Multi-Query-Fehler", error=str(e))
            return []

    # ── Aufgabe 4: Metadaten-Filter ───────────────────────────────────────────

    def build_metadata_filter(
        self,
        query_typ:      str,
        norm_reference: Optional[str],
    ) -> dict:
        """
        Leitet PostgreSQL-Vorfilter aus Query-Typ und Norm-Referenz ab.
        """
        if query_typ == "NORM":
            return {
                "doc_class":   ["A"],
                "source_type": ["gesetz", "verordnung", "standard"],
            }

        elif query_typ == "IM":
            return {
                "im_signals_exists": True,     # nur Chunks mit IM-Signalen
                "doc_class":         ["A", "B"],
            }

        elif query_typ == "ENTITY":
            return {
                "doc_class":  ["A", "B"],
                "norm_type":  ["COMPETENCE", "DEF", "MUST"],
            }

        else:   # GENERAL
            return {}   # kein Filter – alle Klassen durchsuchen

    # ── Öffentliche API ───────────────────────────────────────────────────────

    async def transform(
        self,
        query:    str,
        debug:    bool = False,
    ) -> QueryBundle:
        """
        Hauptmethode – transformiert einen Rohtext-Query in ein QueryBundle.

        Ablauf:
          1. Query-Typ klassifizieren
          2. Norm-Referenz extrahieren
          3. Vektoren parallel erzeugen (Direct + HyDE + Step-Back/Multi-Query)
          4. Metadaten-Filter ableiten

        Returns:
            QueryBundle mit allen Suchvektoren und Filtern für den Retriever.
        """
        query     = query.strip()
        qt_cfg    = self._get_qt_cfg()
        debug_log = {}

        # ── Schritt 1: Query-Typ ──────────────────────────────────────────────
        query_typ = self.classify_query_typ(query)
        logger.info("Query-Typ", typ=query_typ, query=query[:60])
        if debug:
            debug_log["query_typ_signale"] = {
                "norm":   bool(_NORM_SIGNALE_RE.search(query.lower())),
                "im":     bool(_IM_SIGNALE_RE.search(query.lower())),
                "entity": bool(_ENTITY_SIGNALE_RE.search(query.lower())),
            }

        # ── Schritt 2: Norm-Referenz ──────────────────────────────────────────
        norm_ref   = self.extract_norm_reference(query)
        direktlookup = norm_ref is not None
        hyde_cfg   = qt_cfg.get("hyde", {})
        skip_hyde  = direktlookup and hyde_cfg.get("skip_if_direktlookup", True)

        # ── Schritt 3: Vektoren parallel erzeugen ────────────────────────────
        vektoren: list[QueryVector] = [self._vector_direct(query)]

        llm_tasks = []

        if not skip_hyde:
            llm_tasks.append(self._vector_hyde(query, qt_cfg))
        else:
            async def _noop(): return None
            llm_tasks.append(_noop())

        llm_tasks.append(
            self._vector_step_back(query, qt_cfg, query_typ)
        )
        llm_tasks.append(
            self._vectors_multi_query(query, qt_cfg, query_typ)
        )

        results = await asyncio.gather(*llm_tasks, return_exceptions=True)

        # HyDE
        if not isinstance(results[0], Exception) and results[0] is not None:
            vektoren.append(results[0])

        # Step-Back
        if not isinstance(results[1], Exception) and results[1] is not None:
            vektoren.append(results[1])

        # Multi-Query
        if not isinstance(results[2], Exception):
            vektoren.extend(results[2] or [])

        # ── Schritt 4: Metadaten-Filter ───────────────────────────────────────
        metadata_filter = self.build_metadata_filter(query_typ, norm_ref)

        logger.info(
            "QueryBundle erzeugt",
            typ=query_typ,
            vektoren=len(vektoren),
            direktlookup=direktlookup,
            norm_ref=norm_ref,
        )

        if debug:
            debug_log.update({
                "vektoren": [
                    {"strategie": v.strategie,
                     "gewicht":   v.gewicht,
                     "text":      v.text[:80]}
                    for v in vektoren
                ],
                "metadata_filter": metadata_filter,
            })

        return QueryBundle(
            original_query  = query,
            query_typ       = query_typ,
            norm_reference  = norm_ref,
            vektoren        = vektoren,
            metadata_filter = metadata_filter,
            im_filter       = query_typ == "IM",
            direktlookup    = direktlookup,
            debug           = debug_log,
        )
