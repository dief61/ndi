# services/ingest/app/services/knowledge_graph.py
#
# Knowledge Graph Service – exportiert SVO-Tripel und NER-Entitäten
# aus PostgreSQL nach Apache Jena Fuseki als RDF/Turtle.
#
# Named Graphs:
#   https://mnr.nds.de/graph/svo      SVO-Tripel
#   https://mnr.nds.de/graph/ner      NER-Entitäten
#   https://mnr.nds.de/graph/norms    Norm-Metadaten
#
# SPARQL-Endpoint: http://localhost:3030/mnr/sparql
# Update-Endpoint: http://localhost:3030/mnr/update

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Namespaces
# ─────────────────────────────────────────────────────────────────────────────

NS_MNR    = "https://mnr.nds.de/ontology#"
NS_NORM   = "https://mnr.nds.de/norm/"
NS_CHUNK  = "https://mnr.nds.de/chunk/"
NS_ENTITY = "https://mnr.nds.de/entity/"
NS_SVO    = "https://mnr.nds.de/svo/"

GRAPH_SVO   = "https://mnr.nds.de/graph/svo"
GRAPH_NER   = "https://mnr.nds.de/graph/ner"
GRAPH_NORMS = "https://mnr.nds.de/graph/norms"

PREFIXES = f"""PREFIX mnr:    <{NS_MNR}>
PREFIX norm:   <{NS_NORM}>
PREFIX chunk:  <{NS_CHUNK}>
PREFIX ent:    <{NS_ENTITY}>
PREFIX svo:    <{NS_SVO}>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX dcterms:<http://purl.org/dc/terms/>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _safe_uri(base: str, value: str) -> str:
    """Erzeugt eine sichere URI aus Basis + Wert."""
    safe = re.sub(r'[^\w\-.]', '_', str(value))
    return f"<{base}{safe}>"


def _literal(value: str, datatype: str = None, lang: str = None) -> str:
    """Erzeugt ein RDF-Literal."""
    escaped = str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    if datatype:
        return f'"{escaped}"^^<{datatype}>'
    if lang:
        return f'"{escaped}"@{lang}'
    return f'"{escaped}"'


def _float_literal(value: float) -> str:
    return f'"{value:.4f}"^^<http://www.w3.org/2001/XMLSchema#decimal>'


# ─────────────────────────────────────────────────────────────────────────────
# Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExportResult:
    """Ergebnis eines KG-Export-Laufs."""
    docs_processed:    int = 0
    chunks_processed:  int = 0
    triples_svo:       int = 0
    triples_ner:       int = 0
    triples_norms:     int = 0
    errors:            list[str] = field(default_factory=list)

    @property
    def total_triples(self) -> int:
        return self.triples_svo + self.triples_ner + self.triples_norms


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph Service
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphService:
    """
    Exportiert SVO-Tripel und NER-Entitäten nach Apache Jena Fuseki.

    Workflow:
    1. SVO-Tripel aus svo_extractions → Graph svo
    2. NER-Entitäten aus ner_entities → Graph ner
    3. Norm-Metadaten aus norm_documents → Graph norms
    4. Sync-Status in kg_sync_log protokollieren
    """

    def __init__(self):
        fuseki_url = getattr(settings, "fuseki_url",
                             "http://localhost:3030")
        fuseki_pwd = getattr(settings, "fuseki_admin_password",
                             "mnr_fuseki_password")
        self.sparql_endpoint = f"{fuseki_url}/mnr/sparql"
        self.update_endpoint = f"{fuseki_url}/mnr/update"
        self.data_endpoint   = f"{fuseki_url}/mnr/data"
        self.auth = ("admin", fuseki_pwd)
        self.timeout = httpx.Timeout(60.0)

    # ── Verbindung testen ─────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """
        Prüft ob Fuseki erreichbar ist.
        Nutzt SPARQL ASK – kein Admin-Auth nötig (lesender Zugriff).
        """
        test_url = f"{self.sparql_endpoint}?query=ASK%7B%7D"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    test_url,
                    headers={"Accept": "application/sparql-results+json"},
                )
                # 200 = OK, boolean=true → Fuseki läuft
                return r.status_code == 200
        except Exception:
            return False

    # ── SPARQL-Abfragen ───────────────────────────────────────────────────────

    async def query(self, sparql: str) -> dict:
        """Führt eine SPARQL SELECT-Abfrage aus."""
        async with httpx.AsyncClient(
            auth=self.auth, timeout=self.timeout
        ) as client:
            r = await client.post(
                self.sparql_endpoint,
                data={"query": PREFIXES + "\n" + sparql},
                headers={"Accept": "application/sparql-results+json"},
            )
            r.raise_for_status()
            return r.json()

    async def update(self, sparql: str) -> bool:
        """Führt ein SPARQL UPDATE aus."""
        async with httpx.AsyncClient(
            auth=self.auth, timeout=self.timeout
        ) as client:
            r = await client.post(
                self.update_endpoint,
                data={"update": PREFIXES + "\n" + sparql},
            )
            r.raise_for_status()
            return True

    async def upload_turtle(self, turtle: str, graph_uri: str) -> bool:
        """Lädt Turtle-RDF in einen Named Graph hoch."""
        async with httpx.AsyncClient(
            auth=self.auth, timeout=self.timeout
        ) as client:
            r = await client.post(
                f"{self.data_endpoint}?graph={quote(graph_uri)}",
                content=turtle.encode("utf-8"),
                headers={"Content-Type": "text/turtle; charset=utf-8"},
            )
            r.raise_for_status()
            return True

    # ── Norm-Metadaten exportieren ────────────────────────────────────────────

    async def export_norm_metadata(self, conn) -> int:
        """Exportiert Norm-Metadaten aus norm_documents nach Fuseki."""
        docs = await conn.fetch("""
            SELECT id::text, doc_id, title, source_type,
                   jurisdiction, version, valid_from, valid_to,
                   ingest_ts
            FROM norm_documents
            ORDER BY ingest_ts
        """)

        if not docs:
            return 0

        triples = [PREFIXES]
        triples.append(f"\n# Norm-Metadaten – {len(docs)} Dokumente\n")

        for doc in docs:
            norm_uri = _safe_uri(NS_NORM, doc["doc_id"])
            triples.append(f"{norm_uri}")
            triples.append(f"    a mnr:NormDocument ;")
            triples.append(f"    dcterms:title {_literal(doc['title'], lang='de')} ;")
            triples.append(f"    mnr:sourceType {_literal(doc['source_type'])} ;")
            triples.append(f"    mnr:docId {_literal(doc['doc_id'])} ;")
            if doc["jurisdiction"]:
                triples.append(f"    mnr:jurisdiction {_literal(doc['jurisdiction'])} ;")
            if doc["version"]:
                triples.append(f"    mnr:version {_literal(doc['version'])} ;")
            if doc["valid_from"]:
                triples.append(f"    mnr:validFrom {_literal(str(doc['valid_from']), 'http://www.w3.org/2001/XMLSchema#date')} ;")
            triples.append(f"    .")

        turtle = "\n".join(triples)

        # Alten Graph ersetzen (CLEAR kann bei einigen Images fehlschlagen)
        await self._safe_clear_graph(GRAPH_NORMS)
        await self.upload_turtle(turtle, GRAPH_NORMS)

        count = len(docs) * 4  # ca. 4 Tripel pro Dokument
        logger.info("Norm-Metadaten exportiert", docs=len(docs), triples=count)
        return count

    # ── SVO-Tripel exportieren ────────────────────────────────────────────────

    async def export_svo(
        self,
        conn,
        doc_id:      Optional[str] = None,
        incremental: bool          = True,
    ) -> int:
        """
        Exportiert SVO-Tripel aus svo_extractions nach Fuseki.

        Args:
            doc_id:      Optional – nur für dieses Dokument exportieren
            incremental: True = nur neue/geänderte Chunks exportieren
        """
        where_parts = ["s.subject IS NOT NULL"]
        params      = []

        if doc_id:
            params.append(doc_id)
            where_parts.append(f"nd.doc_id = ${len(params)}")

        if incremental:
            where_parts.append("""
                nc.id NOT IN (
                    SELECT chunk_id FROM kg_sync_log
                    WHERE status = 'exported'
                )
            """)

        where = "WHERE " + " AND ".join(where_parts)

        rows = await conn.fetch(f"""
            SELECT
                s.id::text           AS svo_id,
                s.chunk_id::text     AS chunk_id,
                nc.doc_id::text      AS doc_id,
                nd.doc_id            AS norm_doc_id,
                nd.title             AS doc_title,
                nc.norm_reference,
                s.subject,
                s.subject_type,
                s.predicate,
                s.predicate_lemma,
                s.object,
                s.object_type,
                s.norm_type,
                s.context,
                s.confidence,
                s.norm_type_confidence
            FROM svo_extractions s
            JOIN norm_chunks nc   ON s.chunk_id = nc.id
            JOIN norm_documents nd ON nc.doc_id = nd.id
            {where}
            ORDER BY nd.doc_id, s.chunk_id
        """, *params)

        if not rows:
            logger.info("Keine SVO-Tripel zum Exportieren")
            return 0

        triples  = [PREFIXES]
        triples.append(f"\n# SVO-Tripel – {len(rows)} Einträge\n")
        chunk_ids = set()

        for row in rows:
            svo_uri   = _safe_uri(NS_SVO,    row["svo_id"])
            chunk_uri = _safe_uri(NS_CHUNK,  row["chunk_id"])
            norm_uri  = _safe_uri(NS_NORM,   row["norm_doc_id"])

            triples.append(f"{svo_uri}")
            triples.append(f"    a mnr:SVOTriple ;")
            triples.append(f"    mnr:subject   {_literal(row['subject'], lang='de')} ;")
            triples.append(f"    mnr:predicate {_literal(row['predicate'], lang='de')} ;")

            if row["predicate_lemma"]:
                triples.append(f"    mnr:predicateLemma {_literal(row['predicate_lemma'], lang='de')} ;")

            if row["object"]:
                triples.append(f"    mnr:object    {_literal(row['object'], lang='de')} ;")
                if row["object_type"]:
                    triples.append(f"    mnr:objectType {_literal(row['object_type'])} ;")

            if row["subject_type"]:
                triples.append(f"    mnr:subjectType {_literal(row['subject_type'])} ;")

            triples.append(f"    mnr:normType  {_literal(row['norm_type'] or 'UNKNOWN')} ;")
            triples.append(f"    mnr:confidence {_float_literal(float(row['confidence'] or 0))} ;")
            triples.append(f"    mnr:fromChunk {chunk_uri} ;")
            triples.append(f"    mnr:fromDocument {norm_uri} ;")

            if row["norm_reference"]:
                triples.append(f"    mnr:normReference {_literal(row['norm_reference'])} ;")

            if row["context"]:
                triples.append(f"    mnr:context {_literal(row['context'], lang='de')} ;")

            triples.append(f"    .")
            chunk_ids.add(row["chunk_id"])

        turtle = "\n".join(triples)

        # In Fuseki hochladen
        if not incremental:
            await self._safe_clear_graph(GRAPH_SVO)
        await self.upload_turtle(turtle, GRAPH_SVO)

        # Sync-Log aktualisieren
        await self._update_sync_log(conn, chunk_ids, "exported",
                                    len(rows), GRAPH_SVO)

        logger.info("SVO-Tripel exportiert",
                    rows=len(rows), chunks=len(chunk_ids))
        return len(rows) * 8  # ca. 8 Tripel pro SVO

    # ── NER-Entitäten exportieren ─────────────────────────────────────────────

    async def export_ner(
        self,
        conn,
        doc_id:      Optional[str] = None,
        incremental: bool          = True,
    ) -> int:
        """Exportiert NER-Entitäten aus ner_entities nach Fuseki."""
        where_parts = ["e.confidence > 0.7"]
        params      = []

        if doc_id:
            params.append(doc_id)
            where_parts.append(f"nd.doc_id = ${len(params)}")

        where = "WHERE " + " AND ".join(where_parts)

        rows = await conn.fetch(f"""
            SELECT
                e.id::text          AS ner_id,
                e.chunk_id::text    AS chunk_id,
                nd.doc_id           AS norm_doc_id,
                e.text,
                e.label,
                e.source,
                e.confidence,
                nc.norm_reference
            FROM ner_entities e
            JOIN norm_chunks nc    ON e.chunk_id = nc.id
            JOIN norm_documents nd ON nc.doc_id = nd.id
            {where}
            ORDER BY e.label, e.text
        """, *params)

        if not rows:
            return 0

        triples = [PREFIXES]
        triples.append(f"\n# NER-Entitäten – {len(rows)} Einträge\n")

        # Entitäten deduplizieren (gleicher Text + Label → eine URI)
        seen_entities: set[str] = set()

        for row in rows:
            ent_key  = f"{row['label']}_{row['text']}"
            ent_uri  = _safe_uri(NS_ENTITY, ent_key)
            chunk_uri = _safe_uri(NS_CHUNK, row["chunk_id"])
            norm_uri  = _safe_uri(NS_NORM,  row["norm_doc_id"])

            # Entitäts-Typ-Definition (einmalig)
            if ent_key not in seen_entities:
                seen_entities.add(ent_key)
                rdf_class = self._label_to_rdf_class(row["label"])
                triples.append(f"{ent_uri}")
                triples.append(f"    a mnr:{rdf_class}, mnr:NamedEntity ;")
                triples.append(f"    rdfs:label {_literal(row['text'], lang='de')} ;")
                triples.append(f"    mnr:nerLabel {_literal(row['label'])} ;")
                triples.append(f"    .")

            # Vorkommen: Entität ↔ Chunk
            occ_uri = _safe_uri(NS_ENTITY, f"occ_{row['ner_id']}")
            triples.append(f"{occ_uri}")
            triples.append(f"    a mnr:EntityOccurrence ;")
            triples.append(f"    mnr:entity    {ent_uri} ;")
            triples.append(f"    mnr:inChunk   {chunk_uri} ;")
            triples.append(f"    mnr:inDocument {norm_uri} ;")
            triples.append(f"    mnr:confidence {_float_literal(float(row['confidence']))} ;")
            triples.append(f"    mnr:nerSource  {_literal(row['source'])} ;")
            if row["norm_reference"]:
                triples.append(f"    mnr:normReference {_literal(row['norm_reference'])} ;")
            triples.append(f"    .")

        turtle = "\n".join(triples)

        if not incremental:
            await self._safe_clear_graph(GRAPH_NER)
        await self.upload_turtle(turtle, GRAPH_NER)

        logger.info("NER-Entitäten exportiert",
                    rows=len(rows), unique=len(seen_entities))
        return len(rows) * 5

    # ── Vollständiger Export ──────────────────────────────────────────────────

    async def export_all(
        self,
        conn,
        doc_id:      Optional[str] = None,
        incremental: bool          = True,
    ) -> ExportResult:
        """Exportiert alle drei Graphen."""
        result = ExportResult()

        try:
            result.triples_norms = await self.export_norm_metadata(conn)
        except Exception as e:
            result.errors.append(f"Norm-Metadaten: {e}")
            logger.error("Norm-Export fehlgeschlagen", error=str(e))

        try:
            result.triples_svo = await self.export_svo(
                conn, doc_id=doc_id, incremental=incremental
            )
        except Exception as e:
            result.errors.append(f"SVO: {e}")
            logger.error("SVO-Export fehlgeschlagen", error=str(e))

        try:
            result.triples_ner = await self.export_ner(
                conn, doc_id=doc_id, incremental=incremental
            )
        except Exception as e:
            result.errors.append(f"NER: {e}")
            logger.error("NER-Export fehlgeschlagen", error=str(e))

        logger.info(
            "KG-Export abgeschlossen",
            tripel_svo=result.triples_svo,
            tripel_ner=result.triples_ner,
            tripel_norms=result.triples_norms,
            gesamt=result.total_triples,
            fehler=len(result.errors),
        )
        return result

    # ── Vordefinierte SPARQL-Abfragen ─────────────────────────────────────────

    async def query_svo_by_subject(self, subject: str) -> list[dict]:
        """Alle SVOs für ein bestimmtes Subjekt (Akteur)."""
        sparql = f"""
        SELECT ?subjekt ?praedikat ?objekt ?normtyp ?normref ?konfidenz
        WHERE {{
            GRAPH <{GRAPH_SVO}> {{
                ?svo a mnr:SVOTriple ;
                     mnr:subject   ?subjekt ;
                     mnr:predicate ?praedikat ;
                     mnr:normType  ?normtyp ;
                     mnr:confidence ?konfidenz .
                OPTIONAL {{ ?svo mnr:object        ?objekt }}
                OPTIONAL {{ ?svo mnr:normReference ?normref }}
                FILTER (LCASE(STR(?subjekt)) = LCASE("{subject}"))
            }}
        }}
        ORDER BY DESC(?konfidenz)
        LIMIT 50
        """
        result = await self.query(sparql)
        return self._bindings_to_list(result)

    async def query_entities_by_label(self, label: str) -> list[dict]:
        """Alle NER-Entitäten eines bestimmten Labels."""
        sparql = f"""
        SELECT ?text ?label ?normref (COUNT(?occ) AS ?haeufigkeit)
        WHERE {{
            GRAPH <{GRAPH_NER}> {{
                ?ent   a mnr:NamedEntity ;
                       rdfs:label ?text ;
                       mnr:nerLabel "{label}" .
                ?occ   mnr:entity ?ent .
                OPTIONAL {{ ?occ mnr:normReference ?normref }}
            }}
        }}
        GROUP BY ?text ?label ?normref
        ORDER BY DESC(?haeufigkeit)
        LIMIT 30
        """
        result = await self.query(sparql)
        return self._bindings_to_list(result)

    async def query_norm_dependencies(self) -> list[dict]:
        """Welche Normen werden von anderen Normen referenziert?"""
        sparql = f"""
        SELECT ?quelle ?ziel (COUNT(?svo) AS ?verweise)
        WHERE {{
            GRAPH <{GRAPH_SVO}> {{
                ?svo mnr:fromDocument ?quelle ;
                     mnr:normReference ?normref .
            }}
            GRAPH <{GRAPH_NORMS}> {{
                ?zielDoc mnr:normReference ?normref .
                BIND(STR(?zielDoc) AS ?ziel)
            }}
        }}
        GROUP BY ?quelle ?ziel
        ORDER BY DESC(?verweise)
        LIMIT 50
        """
        result = await self.query(sparql)
        return self._bindings_to_list(result)

    async def query_impact_analysis(self, norm_doc_id: str) -> list[dict]:
        """
        Impact-Analyse: Welche Chunks/SVOs sind von einem Dokument abhängig?
        Wichtig für Gesetzesänderungen.
        """
        norm_uri = f"{NS_NORM}{norm_doc_id}"
        sparql = f"""
        SELECT ?chunk ?subjekt ?praedikat ?objekt ?normref
        WHERE {{
            GRAPH <{GRAPH_SVO}> {{
                ?svo mnr:fromDocument <{norm_uri}> ;
                     mnr:subject   ?subjekt ;
                     mnr:predicate ?praedikat ;
                     mnr:fromChunk ?chunk .
                OPTIONAL {{ ?svo mnr:object        ?objekt }}
                OPTIONAL {{ ?svo mnr:normReference ?normref }}
            }}
        }}
        ORDER BY ?normref
        LIMIT 100
        """
        result = await self.query(sparql)
        return self._bindings_to_list(result)

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    async def _safe_clear_graph(self, graph_uri: str) -> None:
        """
        Löscht einen Named Graph sicher.
        Versucht zuerst SPARQL UPDATE CLEAR, dann HTTP DELETE.
        """
        # Methode 1: HTTP DELETE auf den Graph-Endpoint
        try:
            async with httpx.AsyncClient(
                auth=self.auth, timeout=self.timeout
            ) as client:
                r = await client.delete(
                    f"{self.data_endpoint}?graph={quote(graph_uri)}"
                )
                if r.status_code in (200, 204, 404):
                    return
        except Exception:
            pass

        # Methode 2: SPARQL UPDATE
        try:
            await self.update(f"DROP SILENT GRAPH <{graph_uri}>")
        except Exception:
            pass  # Graph existiert noch nicht – kein Problem

    def _label_to_rdf_class(self, label: str) -> str:
        """NER-Label → RDF-Klasse."""
        mapping = {
            "GESETZ":      "Norm",
            "BEHÖRDE":     "Institution",
            "ROLLE":       "Role",
            "ORT":         "Location",
            "DATENOBJEKT": "DataObject",
            "SONSTIGE":    "Entity",
        }
        return mapping.get(label, "Entity")

    async def _update_sync_log(
        self,
        conn,
        chunk_ids: set[str],
        status:    str,
        count:     int,
        graph_uri: str,
    ) -> None:
        """Aktualisiert den KG-Sync-Log für verarbeitete Chunks."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        for cid in chunk_ids:
            # Prüfen ob Eintrag existiert
            existing = await conn.fetchval(
                "SELECT id FROM kg_sync_log WHERE chunk_id = $1::uuid",
                cid,
            )
            if existing:
                await conn.execute(
                    """UPDATE kg_sync_log
                       SET status=$2, triple_count=$3,
                           graph_uri=$4, exported_at=$5, updated_at=now()
                       WHERE chunk_id=$1::uuid""",
                    cid, status, count // len(chunk_ids), graph_uri, now,
                )
            else:
                await conn.execute(
                    """INSERT INTO kg_sync_log
                           (chunk_id, doc_id, status, triple_count, graph_uri, exported_at)
                       SELECT $1::uuid, nc.doc_id, $2, $3, $4, $5
                       FROM norm_chunks nc WHERE nc.id = $1::uuid""",
                    cid, status, count // len(chunk_ids), graph_uri, now,
                )

    def _bindings_to_list(self, sparql_result: dict) -> list[dict]:
        """Konvertiert SPARQL-JSON-Ergebnis in eine einfache Liste."""
        bindings = sparql_result.get("results", {}).get("bindings", [])
        return [
            {k: v.get("value") for k, v in row.items()}
            for row in bindings
        ]
