# services/ingest/app/services/storage.py
#
# DocumentStorage: Persistiert Dokumente, Chunks und Embeddings.
#
# Verantwortlichkeiten:
#   store_raw_document()    → Rohdatei in MinIO ablegen
#   create_document_record() → Metadaten in norm_documents (PostgreSQL)
#   store_chunks()          → Chunks + Embeddings in norm_chunks (PostgreSQL)

from __future__ import annotations

import io
import json
from typing import Optional

import asyncpg
import structlog
from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.services.chunker import Chunk

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-Typ-Mapping
#
# Der Chunker erzeugt Normtypen (must, may, definition, ...).
# Die Datenbank erlaubt nur: tatbestand | rechtsfolge | definition |
#   ausnahme | anforderung | tabelle | verweis | zustaendigkeit
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_TYPE_MAP = {
    # Normtyp-Codes → DB-erlaubte Werte
    "must":        "tatbestand",
    "may":         "tatbestand",
    "must_not":    "tatbestand",
    "def":         "definition",
    "definition":  "definition",
    "except":      "ausnahme",
    "ausnahme":    "ausnahme",
    "deadline":    "tatbestand",
    "competence":  "zustaendigkeit",
    "zustaendigkeit": "zustaendigkeit",
    "anforderung": "anforderung",
    "tabelle":     "tabelle",
    "verweis":     "verweis",
    "rechtsfolge": "rechtsfolge",
    "tatbestand":  "tatbestand",
}

VALID_CHUNK_TYPES = {
    "tatbestand", "rechtsfolge", "definition", "ausnahme",
    "anforderung", "tabelle", "verweis", "zustaendigkeit"
}


def normalize_chunk_type(chunk_type: Optional[str]) -> str:
    """Mappt Chunker-Normtypen auf DB-erlaubte Werte. Fallback: tatbestand."""
    if not chunk_type:
        return "tatbestand"
    mapped = CHUNK_TYPE_MAP.get(chunk_type.lower(), chunk_type.lower())
    return mapped if mapped in VALID_CHUNK_TYPES else "tatbestand"


# ─────────────────────────────────────────────────────────────────────────────
# DocumentStorage
# ─────────────────────────────────────────────────────────────────────────────

class DocumentStorage:
    """
    Persistiert Dokumente, Chunks und Embeddings in PostgreSQL und MinIO.

    PostgreSQL-Verbindung: asyncpg (async)
    MinIO-Verbindung: minio Python-Client (sync, im Threadpool ausgeführt)
    """

    def __init__(self):
        self._pg_pool: Optional[asyncpg.Pool] = None
        self._minio_client: Optional[Minio] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Verbindungs-Management
    # ─────────────────────────────────────────────────────────────────────────

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazy-Init des PostgreSQL Connection Pools."""
        if self._pg_pool is None:
            self._pg_pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            logger.info("PostgreSQL Connection Pool initialisiert")
        return self._pg_pool

    def _get_minio(self) -> Minio:
        """Lazy-Init des MinIO Clients."""
        if self._minio_client is None:
            self._minio_client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_root_user,
                secret_key=settings.minio_root_password,
                secure=False,   # Lokal kein TLS
            )
            logger.info("MinIO Client initialisiert", endpoint=settings.minio_endpoint)
        return self._minio_client

    async def close(self):
        """Connection Pool schließen (beim Service-Shutdown aufrufen)."""
        if self._pg_pool:
            await self._pg_pool.close()
            self._pg_pool = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Rohdokument in MinIO
    # ─────────────────────────────────────────────────────────────────────────

    async def store_raw_document(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """
        Speichert das Rohdokument in MinIO (Bucket: mnr-dokumente).

        Returns:
            minio_path: "mnr-dokumente/{doc_id}/{filename}"
        """
        bucket   = "mnr-dokumente"
        obj_path = f"{doc_id}/{filename}"

        minio = self._get_minio()

        # Bucket anlegen falls nicht vorhanden
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)
            logger.info("MinIO Bucket angelegt", bucket=bucket)

        # Datei hochladen
        minio.put_object(
            bucket_name=bucket,
            object_name=obj_path,
            data=io.BytesIO(content),
            length=len(content),
        )

        minio_path = f"{bucket}/{obj_path}"
        logger.info(
            "Rohdokument in MinIO gespeichert",
            minio_path=minio_path,
            size_bytes=len(content),
        )
        return minio_path

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Dokument-Metadaten in PostgreSQL
    # ─────────────────────────────────────────────────────────────────────────

    async def create_document_record(
        self,
        doc_id: str,
        metadata,
        minio_path: str,
    ) -> str:
        """
        Legt einen Datensatz in norm_documents an.

        Args:
            doc_id:     Externe ID (UUID-String)
            metadata:   DocumentMetadata aus dem Ingest-Request
            minio_path: Pfad in MinIO

        Returns:
            Interne PostgreSQL-UUID (wird für Chunk-Verknüpfung benötigt)
        """
        pool = await self._get_pool()

        # Metadaten als JSONB
        meta_json = json.dumps({
            "minio_path":     minio_path,
            "register_scope": getattr(metadata, "register_scope", None),
        })

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO norm_documents (
                    doc_id, source_type, title, jurisdiction,
                    valid_from, valid_to, norm_reference,
                    version, language, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                ON CONFLICT (doc_id) DO UPDATE
                    SET title        = EXCLUDED.title,
                        metadata     = EXCLUDED.metadata,
                        ingest_ts    = now()
                RETURNING id
                """,
                doc_id,
                getattr(metadata, "source_type", "gesetz"),
                getattr(metadata, "title",       "Unbekannt"),
                getattr(metadata, "jurisdiction", None),
                getattr(metadata, "valid_from",  None),
                getattr(metadata, "valid_to",    None),
                getattr(metadata, "norm_reference", None),
                getattr(metadata, "version",     None),
                getattr(metadata, "language",    "de"),
                meta_json,
            )

        internal_id = str(row["id"])
        logger.info(
            "Dokument-Datensatz angelegt",
            doc_id=doc_id,
            internal_id=internal_id,
        )
        return internal_id

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Chunks + Embeddings in PostgreSQL
    # ─────────────────────────────────────────────────────────────────────────

    async def store_chunks(self, chunks: list[Chunk]) -> int:
        """
        Speichert Chunks und Embeddings als Batch in norm_chunks.

        Die interne norm_documents.id wird per doc_id-Lookup aufgelöst.
        Chunks ohne Embedding werden mit NULL-Vektor gespeichert.

        Returns:
            Anzahl gespeicherter Chunks
        """
        if not chunks:
            return 0

        pool = await self._get_pool()

        # Interne UUIDs für alle doc_ids nachschlagen
        doc_ids = list({c.doc_id for c in chunks})
        doc_id_map: dict[str, str] = {}   # externe doc_id → interne UUID

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, doc_id FROM norm_documents WHERE doc_id = ANY($1)",
                doc_ids,
            )
            for row in rows:
                doc_id_map[row["doc_id"]] = str(row["id"])

        missing = [d for d in doc_ids if d not in doc_id_map]
        if missing:
            raise ValueError(
                f"Dokument-Datensätze nicht gefunden für doc_ids: {missing}. "
                f"Bitte zuerst create_document_record() aufrufen."
            )

        # Chunk-ID-Map: externe chunk_id → interne UUID
        # Wird für parent_id-Auflösung benötigt
        chunk_id_map: dict[str, str] = {}

        # Parent-Chunks zuerst einfügen (Hierarchie-Reihenfolge)
        parents = [c for c in chunks if c.parent_chunk_id is None]
        children = [c for c in chunks if c.parent_chunk_id is not None]
        ordered = parents + children

        async with pool.acquire() as conn:
            inserted = 0

            for chunk in ordered:
                internal_doc_id  = doc_id_map[chunk.doc_id]
                internal_parent  = chunk_id_map.get(chunk.parent_chunk_id) \
                                   if chunk.parent_chunk_id else None
                chunk_type_db    = normalize_chunk_type(chunk.chunk_type)

                # Embedding: Liste → pgvector-String "[x, y, z, ...]"
                embedding_val = None
                if chunk.embedding:
                    embedding_val = "[" + ",".join(
                        str(round(v, 8)) for v in chunk.embedding
                    ) + "]"

                row = await conn.fetchrow(
                    """
                    INSERT INTO norm_chunks (
                        doc_id, doc_class,
                        norm_reference, section_path,
                        heading_breadcrumb, requirement_id,
                        chunk_type, hierarchy_level,
                        parent_id, overlap_with_prev,
                        confidence_weight, content,
                        token_count, metadata,
                        embedding, version, valid_from
                    )
                    VALUES (
                        $1::uuid, $2,
                        $3, $4,
                        $5, $6,
                        $7, $8,
                        $9::uuid, $10,
                        $11, $12,
                        $13, $14::jsonb,
                        $15::vector, $16, $17
                    )
                    ON CONFLICT DO NOTHING
                    RETURNING id
                    """,
                    internal_doc_id,
                    chunk.doc_class,
                    chunk.norm_reference,
                    chunk.section_path,
                    chunk.heading_breadcrumb,
                    chunk.requirement_id,
                    chunk_type_db,
                    chunk.hierarchy_level,
                    internal_parent,
                    chunk.overlap_with_prev or 0.0,
                    chunk.confidence_weight,
                    chunk.text,
                    chunk.token_count,
                    json.dumps({"chunk_id": chunk.chunk_id}),
                    embedding_val,
                    chunk.version,
                    chunk.valid_from,
                )

                if row:
                    chunk_id_map[chunk.chunk_id] = str(row["id"])
                    inserted += 1

        logger.info(
            "Chunks gespeichert",
            total=len(chunks),
            inserted=inserted,
            skipped=len(chunks) - inserted,
        )
        return inserted

    # ─────────────────────────────────────────────────────────────────────────
    # Hilfsmethoden
    # ─────────────────────────────────────────────────────────────────────────

    async def get_document_chunk_count(self, doc_id: str) -> int:
        """Gibt die Anzahl der gespeicherten Chunks für ein Dokument zurück."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) as cnt
                FROM norm_chunks nc
                JOIN norm_documents nd ON nc.doc_id = nd.id
                WHERE nd.doc_id = $1
                """,
                doc_id,
            )
        return row["cnt"] if row else 0

    async def document_exists(self, doc_id: str) -> bool:
        """Prüft ob ein Dokument bereits in der Datenbank existiert."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM norm_documents WHERE doc_id = $1",
                doc_id,
            )
        return row is not None
