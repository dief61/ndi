# services/ingest/app/services/rag/retriever.py
#
# M3 – Schritt 1.2 + 1.3 + 1.4: Klassen-sensitiver Hybrid-Retriever
#
# Schritt 1.2: Klassen-sensitiver Retriever
#   - Klasse A: Direkt-Lookup (norm_reference) + Vektor-Cosine + FTS → RRF
#   - Klasse B: Direkt-Lookup (requirement_id) + Vektor-Cosine + FTS → RRF
#   - Klasse C: nur Vektor-Cosine (erhöhter Threshold 0.72)
#
# Schritt 1.3: Parent-Child-Expansion
#   - Klasse A: §-Parent immer nachladen
#   - Klasse B: Kapitel-Parent selektiv + Tabellen-Chunks
#   - Klasse C: kein Parent-Fetch
#
# Schritt 1.4: Cross-Reference-Expansion (nur Klasse A)
#   - cross_references-Feld → referenzierte Chunks nachladen (max. Tiefe 2)

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import asyncpg
import structlog
import yaml

from app.services.rag.query_transformer import QueryBundle, QueryVector

logger = structlog.get_logger()

_RAG_CFG_PATH = Path(__file__).parents[3] / "rag_config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Datenstrukturen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """Ein gefundener Chunk mit Score und Metadaten."""
    chunk_id:          str
    doc_id:            str
    doc_class:         str             # A | B | C
    norm_reference:    Optional[str]
    section_path:      Optional[str]
    heading_breadcrumb: Optional[str]
    requirement_id:    Optional[str]
    chunk_type:        str
    hierarchy_level:   int
    parent_id:         Optional[str]
    cross_references:  list[str]
    confidence_weight: float
    content:           str
    token_count:       int
    im_signals:        Optional[dict]
    score:             float           # kombinierter Retrieval-Score
    retrieval_source:  str             # direktlookup | vektor | fts | rrf


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Klassen-sensitiver Hybrid-Retriever für MNR.

    Kombiniert:
      - Direkt-Lookup (norm_reference / requirement_id)
      - Vektorsearch (pgvector cosine)
      - Volltextsuche (PostgreSQL FTS)
      - RRF-Fusion (Reciprocal Rank Fusion)

    Ergänzt durch:
      - Parent-Child-Expansion (1.3)
      - Cross-Reference-Expansion (1.4)
    """

    def __init__(
        self,
        pool:        asyncpg.Pool,
        embedder,                     # Embedder-Instanz aus M1
        config_path: Optional[Path] = None,
    ):
        self._pool    = pool
        self._embedder = embedder
        self._cfg_path = config_path or _RAG_CFG_PATH

    # ── Konfiguration ─────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            with open(self._cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _row_to_chunk(self, row: asyncpg.Record, score: float,
                      source: str) -> RetrievedChunk:
        """asyncpg-Record → RetrievedChunk."""
        return RetrievedChunk(
            chunk_id           = str(row["id"]),
            doc_id             = str(row["doc_id"]),
            doc_class          = row["doc_class"] or "A",
            norm_reference     = row.get("norm_reference"),
            section_path       = row.get("section_path"),
            heading_breadcrumb = row.get("heading_breadcrumb"),
            requirement_id     = row.get("requirement_id"),
            chunk_type         = row.get("chunk_type") or "tatbestand",
            hierarchy_level    = row.get("hierarchy_level") or 1,
            parent_id          = str(row["parent_id"]) if row.get("parent_id") else None,
            cross_references   = list(row.get("cross_references") or []),
            confidence_weight  = float(row.get("confidence_weight") or 1.0),
            content            = row["content"] or "",
            token_count        = row.get("token_count") or 0,
            im_signals         = dict(row["im_signals"]) if row.get("im_signals") else None,
            score              = score,
            retrieval_source   = source,
        )

    def _build_where(
        self,
        meta_filter:    dict,
        doc_class:      Optional[str] = None,
        extra_where:    str = "",
    ) -> tuple[str, list]:
        """
        Baut WHERE-Klausel und Parameter-Liste aus Metadaten-Filter.
        """
        conditions = []
        params     = []
        idx        = 1   # PostgreSQL $1, $2, ...

        # Gültigkeitsprüfung (zeitlich)
        conditions.append(
            f"(nc.valid_to IS NULL OR nc.valid_to >= CURRENT_DATE)"
        )

        # Klassen-Filter
        if doc_class:
            conditions.append(f"nc.doc_class = ${idx}")
            params.append(doc_class)
            idx += 1
        elif "doc_class" in meta_filter:
            classes = meta_filter["doc_class"]
            ph      = ", ".join(f"${idx + i}" for i in range(len(classes)))
            conditions.append(f"nc.doc_class IN ({ph})")
            params.extend(classes)
            idx    += len(classes)

        # source_type-Filter
        if "source_type" in meta_filter:
            types = meta_filter["source_type"]
            ph    = ", ".join(f"${idx + i}" for i in range(len(types)))
            conditions.append(
                f"nd.source_type IN ({ph})"
            )
            params.extend(types)
            idx  += len(types)

        # norm_type-Filter
        if "norm_type" in meta_filter:
            # Chunks über SVOs filtern
            types = meta_filter["norm_type"]
            ph    = ", ".join(f"${idx + i}" for i in range(len(types)))
            conditions.append(
                f"EXISTS (SELECT 1 FROM svo_extractions s "
                f"WHERE s.chunk_id = nc.id AND s.norm_type IN ({ph}))"
            )
            params.extend(types)
            idx  += len(types)

        # im_signals-Filter
        if meta_filter.get("im_signals_exists"):
            conditions.append("nc.im_signals IS NOT NULL")

        # Zusätzliche WHERE-Bedingung
        if extra_where:
            conditions.append(extra_where)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        return where, params

    # ── Schritt 1.2a: Direkt-Lookup ───────────────────────────────────────────

    async def _direktlookup(
        self,
        norm_reference: str,
        meta_filter:    dict,
        top_k:          int,
    ) -> list[RetrievedChunk]:
        """
        Direkt-Lookup über norm_reference (Klasse A) oder
        requirement_id (Klasse B).
        Umgeht den Vektorvergleich vollständig.
        """
        async with self._pool.acquire() as conn:
            # Klasse A: norm_reference
            rows_a = await conn.fetch("""
                SELECT nc.*, nd.source_type
                FROM norm_chunks nc
                JOIN norm_documents nd ON nc.doc_id = nd.id
                WHERE nc.norm_reference ILIKE $1
                  AND nc.doc_class = 'A'
                  AND (nc.valid_to IS NULL OR nc.valid_to >= CURRENT_DATE)
                ORDER BY nc.hierarchy_level ASC
                LIMIT $2
            """, f"{norm_reference}%", top_k)

            chunks = [
                self._row_to_chunk(r, 1.0, "direktlookup")
                for r in rows_a
            ]

            if chunks:
                logger.debug("Direkt-Lookup Klasse A",
                             ref=norm_reference, treffer=len(chunks))
                return chunks

            # Fallback: Fuzzy-Suche wenn exakt nichts gefunden
            para = norm_reference.split()[0:2]   # z.B. ["§", "3"]
            rows_f = await conn.fetch("""
                SELECT nc.*, nd.source_type
                FROM norm_chunks nc
                JOIN norm_documents nd ON nc.doc_id = nd.id
                WHERE nc.norm_reference ILIKE $1
                  AND nc.doc_class = 'A'
                  AND (nc.valid_to IS NULL OR nc.valid_to >= CURRENT_DATE)
                ORDER BY nc.hierarchy_level ASC
                LIMIT $2
            """, f"{'%'.join(para)}%", top_k)

            return [
                self._row_to_chunk(r, 0.9, "direktlookup_fuzzy")
                for r in rows_f
            ]

    # ── Schritt 1.2b: Vektor-Suche ────────────────────────────────────────────

    async def _vektor_search(
        self,
        vektor:      QueryVector,
        doc_class:   str,
        threshold:   float,
        top_k:       int,
        meta_filter: dict,
    ) -> list[tuple[RetrievedChunk, int]]:
        """
        Vektorsuche für eine bestimmte Dokumentklasse.
        Returns: [(chunk, rank)]
        """
        # Embedding erzeugen
        embedding = self._embedder.embed_query(vektor.text)
        if embedding is None:
            return []

        # pgvector erwartet String-Format '[0.1, 0.2, ...]'
        if isinstance(embedding, (list, tuple)):
            embedding = "[" + ",".join(str(x) for x in embedding) + "]"
        elif hasattr(embedding, "tolist"):          # numpy array
            embedding = "[" + ",".join(str(x) for x in embedding.tolist()) + "]" 

        where, params = self._build_where(meta_filter, doc_class=doc_class)
        # Embedding als letzter Parameter
        params.append(embedding)
        emb_idx = len(params)
        params.append(top_k * 2)    # mehr Kandidaten für RRF
        limit_idx = len(params)
        params.append(threshold)
        thr_idx = len(params)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT nc.*,
                       nd.source_type,
                       1 - (nc.embedding <=> ${emb_idx}::vector) AS cosine_sim
                FROM norm_chunks nc
                JOIN norm_documents nd ON nc.doc_id = nd.id
                {where}
                  AND nc.embedding IS NOT NULL
                  AND 1 - (nc.embedding <=> ${emb_idx}::vector) >= ${thr_idx}
                ORDER BY nc.embedding <=> ${emb_idx}::vector
                LIMIT ${limit_idx}
            """, *params)

        return [
            (self._row_to_chunk(r, float(r["cosine_sim"]), "vektor"), rank + 1)
            for rank, r in enumerate(rows)
        ]

    # ── Schritt 1.2c: FTS-Suche ───────────────────────────────────────────────

    async def _fts_search(
        self,
        query_text:  str,
        doc_class:   str,
        top_k:       int,
        meta_filter: dict,
        language:    str = "german",
    ) -> list[tuple[RetrievedChunk, int]]:
        """
        PostgreSQL Volltextsuche für eine bestimmte Dokumentklasse.
        Returns: [(chunk, rank)]
        """
        where, params = self._build_where(meta_filter, doc_class=doc_class)
        params.append(query_text)
        q_idx = len(params)
        params.append(top_k * 2)
        limit_idx = len(params)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT nc.*,
                       nd.source_type,
                       ts_rank_cd(
                         to_tsvector('{language}', nc.content),
                         plainto_tsquery('{language}', ${q_idx})
                       ) AS fts_score
                FROM norm_chunks nc
                JOIN norm_documents nd ON nc.doc_id = nd.id
                {where}
                  AND to_tsvector('{language}', nc.content)
                      @@ plainto_tsquery('{language}', ${q_idx})
                ORDER BY fts_score DESC
                LIMIT ${limit_idx}
            """, *params)

        return [
            (self._row_to_chunk(r, float(r["fts_score"]), "fts"), rank + 1)
            for rank, r in enumerate(rows)
        ]

    # ── RRF-Fusion ────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        vektor_ranked: list[tuple[RetrievedChunk, int]],
        fts_ranked:    list[tuple[RetrievedChunk, int]],
        k:             int = 60,
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion: kombiniert Vektor- und FTS-Ranking.

        RRF-Score = Σ 1 / (k + rank_i)

        Chunks die in beiden Listen auftauchen werden bevorzugt.
        """
        scores: dict[str, float]          = {}
        chunks: dict[str, RetrievedChunk] = {}

        for chunk, rank in vektor_ranked:
            cid          = chunk.chunk_id
            scores[cid]  = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks[cid]  = chunk

        for chunk, rank in fts_ranked:
            cid          = chunk.chunk_id
            scores[cid]  = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunks:
                chunks[cid] = chunk

        # Sortieren nach RRF-Score absteigend
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result     = []
        for cid in sorted_ids:
            c       = chunks[cid]
            c.score = scores[cid]
            c.retrieval_source = "rrf"
            result.append(c)

        return result

    # ── Schritt 1.3: Parent-Child-Expansion ───────────────────────────────────

    async def _expand_parents(
        self,
        chunks: list[RetrievedChunk],
        cfg:    dict,
    ) -> list[RetrievedChunk]:
        """
        Lädt Parent-Chunks nach:
          - Klasse A: immer (§-Paragraph als Kontext-Anker)
          - Klasse B: selektiv (wenn token_count > class_b_token_grenze)
          - Klasse C: nie
        """
        class_a_immer    = cfg.get("class_a_immer", True)
        class_b_grenze   = cfg.get("class_b_token_grenze", 450)
        tabellen_exp     = cfg.get("tabellen_expansion", True)

        parent_ids = set()
        for c in chunks:
            if not c.parent_id:
                continue
            if c.doc_class == "A" and class_a_immer:
                parent_ids.add(c.parent_id)
            elif c.doc_class == "B":
                if c.token_count > class_b_grenze:
                    parent_ids.add(c.parent_id)
                if tabellen_exp and c.chunk_type == "tabelle":
                    parent_ids.add(c.parent_id)

        if not parent_ids:
            return chunks

        # Parents laden
        already = {c.chunk_id for c in chunks}
        new_ids = parent_ids - already

        if not new_ids:
            return chunks

        placeholders = ", ".join(f"${i+1}" for i in range(len(new_ids)))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT nc.*, nd.source_type
                FROM norm_chunks nc
                JOIN norm_documents nd ON nc.doc_id = nd.id
                WHERE nc.id IN ({placeholders})
            """, *[p for p in new_ids])

        parents = [
            self._row_to_chunk(r, 0.0, "parent_expansion")
            for r in rows
        ]
        logger.debug("Parent-Expansion", parents=len(parents))
        return chunks + parents

    # ── Schritt 1.4: Cross-Reference-Expansion (Klasse A) ─────────────────────

    async def _expand_crossrefs(
        self,
        chunks:   list[RetrievedChunk],
        cfg:      dict,
        tiefe:    int = 0,
    ) -> list[RetrievedChunk]:
        """
        Lädt cross_references rekursiv nach (nur Klasse A).
        Verhindert Endlosschleifen durch max_tiefe.
        """
        if not cfg.get("enabled", True):
            return chunks

        max_tiefe = cfg.get("max_tiefe", 2)
        if tiefe >= max_tiefe:
            return chunks

        already  = {c.chunk_id for c in chunks}
        ref_ids: set[str] = set()

        for c in chunks:
            if c.doc_class == "A" and c.cross_references:
                for ref in c.cross_references:
                    if str(ref) not in already:
                        ref_ids.add(str(ref))

        if not ref_ids:
            return chunks

        placeholders = ", ".join(f"${i+1}" for i in range(len(ref_ids)))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT nc.*, nd.source_type
                FROM norm_chunks nc
                JOIN norm_documents nd ON nc.doc_id = nd.id
                WHERE nc.id IN ({placeholders})
                  AND nc.doc_class = 'A'
            """, *[p for p in ref_ids])

        new_chunks = [
            self._row_to_chunk(r, 0.0, f"crossref_tiefe_{tiefe+1}")
            for r in rows
        ]

        if new_chunks:
            logger.debug("Cross-Reference-Expansion",
                         tiefe=tiefe+1, neue=len(new_chunks))
            # Rekursiv weitere Querverweise verfolgen
            all_chunks = chunks + new_chunks
            return await self._expand_crossrefs(
                all_chunks, cfg, tiefe + 1)

        return chunks

    # ── Öffentliche API ───────────────────────────────────────────────────────

    async def retrieve(
        self,
        bundle: QueryBundle,
    ) -> list[RetrievedChunk]:
        """
        Hauptmethode – führt den vollständigen Retrieval-Prozess aus.

        Ablauf:
          1.2a Direkt-Lookup (wenn norm_reference erkannt)
          1.2b Vektor-Suche je Klasse und Vektor
          1.2c FTS-Suche je Klasse
          RRF  Fusion von Vektor und FTS
          1.3  Parent-Child-Expansion
          1.4  Cross-Reference-Expansion (Klasse A)

        Returns:
            Liste gefundener Chunks – unsortiert (Sortierung durch Re-Ranker 1.5)
        """
        cfg         = self._load_config()
        ret_cfg     = cfg.get("retrieval", {})
        pc_cfg      = cfg.get("parent_child", {})
        cr_cfg      = cfg.get("cross_reference", {})
        thresholds  = ret_cfg.get("score_thresholds", {})
        top_k       = ret_cfg.get("top_k_pro_klasse", 6)
        rrf_k       = ret_cfg.get("rrf_k", 60)
        fts_enabled = ret_cfg.get("fts_enabled", True)
        meta_filter = bundle.metadata_filter

        all_chunks: list[RetrievedChunk] = []
        seen_ids:   set[str] = set()

        # ── 1.2a: Direkt-Lookup ───────────────────────────────────────────────
        if bundle.direktlookup and bundle.norm_reference:
            dl_chunks = await self._direktlookup(
                bundle.norm_reference, meta_filter, top_k
            )
            for c in dl_chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)

        # ── 1.2b + 1.2c: Vektor + FTS pro Klasse ─────────────────────────────
        # Klassen bestimmen aus Filter (oder alle wenn kein Filter)
        filter_classes = meta_filter.get("doc_class") or ["A", "B", "C"]
        threshold_map  = {
            "A": thresholds.get("class_a", 0.60),
            "B": thresholds.get("class_b", 0.60),
            "C": thresholds.get("class_c", 0.72),
        }

        for doc_class in filter_classes:
            threshold = threshold_map.get(doc_class, 0.60)

            for vektor in bundle.vektoren:
                # Vektor-Suche
                v_ranked = await self._vektor_search(
                    vektor, doc_class, threshold, top_k, meta_filter
                )

                # FTS-Suche (parallel für bessere Performance)
                f_ranked = []
                if fts_enabled:
                    f_ranked = await self._fts_search(
                        bundle.original_query, doc_class,
                        top_k, meta_filter,
                        language=ret_cfg.get("fts_language", "german"),
                    )

                # RRF-Fusion
                fused = self._rrf_fusion(v_ranked, f_ranked, k=rrf_k)

                # Klassen-Konfidenz einrechnen
                kw = {"A": 1.0, "B": 0.85, "C": 0.65}.get(doc_class, 1.0)
                for c in fused:
                    c.score *= kw
                    if c.chunk_id not in seen_ids:
                        all_chunks.append(c)
                        seen_ids.add(c.chunk_id)

        # ── 1.3: Parent-Child-Expansion ───────────────────────────────────────
        all_chunks = await self._expand_parents(all_chunks, pc_cfg)

        # ── 1.4: Cross-Reference-Expansion (Klasse A) ─────────────────────────
        all_chunks = await self._expand_crossrefs(all_chunks, cr_cfg)

        logger.info(
            "Retrieval abgeschlossen",
            chunks_gesamt    = len(all_chunks),
            direktlookup     = bundle.direktlookup,
            query_typ        = bundle.query_typ,
        )

        return all_chunks
