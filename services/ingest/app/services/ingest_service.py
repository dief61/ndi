# services/ingest/app/services/ingest_service.py
#
# Orchestriert die vollständige M1-Pipeline:
# Parsing → Chunking → Embedding → Speicherung
#
# Wird aufgerufen von:
#   - FastAPI Background-Task (über ingest.py Route)
#   - ingest_cli.py (Kommandozeile)
#
# Job-Status wird in ingest_jobs (PostgreSQL) geschrieben.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import asyncpg
import structlog

from app.core.config import settings
from app.services.parser import TikaParser
from app.services.chunker import ChunkingRouter
from app.services.embedder import Embedder
from app.services.storage import DocumentStorage
from app.services.abbrev_normalizer import AbbrevNormalizer

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-Ergebnis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Ergebnis eines vollständigen Pipeline-Durchlaufs."""
    job_id:      str
    doc_id:      str
    filename:    str
    status:      str            # done | error
    doc_class:   Optional[str]  # A | B | C
    chunk_count: int
    error:       Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# IngestService
# ─────────────────────────────────────────────────────────────────────────────

class IngestService:
    """
    Orchestriert die M1-Ingest-Pipeline.

    Singleton-Nutzung empfohlen (Embedder lädt Modell einmalig).
    FastAPI: als App-State gespeichert (main.py lifespan).
    CLI:     direkt instanziiert in ingest_cli.py.
    """

    def __init__(self):
        self.parser     = TikaParser()
        self.normalizer = AbbrevNormalizer()
        self.chunker    = ChunkingRouter()
        self.embedder   = Embedder()
        self.storage    = DocumentStorage()
        self._pg_pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazy-Init des PostgreSQL Connection Pools für Job-Status."""
        if self._pg_pool is None:
            self._pg_pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                min_size=1,
                max_size=5,
            )
        return self._pg_pool

    async def close(self):
        """Verbindungen schließen (Shutdown-Hook in main.py)."""
        await self.storage.close()
        if self._pg_pool:
            await self._pg_pool.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Job-Status Hilfsmethoden
    # ─────────────────────────────────────────────────────────────────────────

    async def _create_job(self, job_id: str, doc_id: str, filename: str):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_jobs (job_id, doc_id, filename, status)
                VALUES ($1, $2, $3, 'queued')
                ON CONFLICT (job_id) DO NOTHING
                """,
                job_id, doc_id, filename,
            )

    async def _update_job(
        self,
        job_id: str,
        status: str,
        doc_class: Optional[str] = None,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ):
        pool = await self._get_pool()
        finished = "now()" if status in ("done", "error") else "NULL"
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE ingest_jobs SET
                    status        = $1,
                    doc_class     = COALESCE($2, doc_class),
                    chunk_count   = COALESCE($3, chunk_count),
                    error_message = COALESCE($4, error_message),
                    updated_at    = now(),
                    finished_at   = {finished}
                WHERE job_id = $5
                """,
                status, doc_class, chunk_count, error_message, job_id,
            )

    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """Gibt den aktuellen Status eines Jobs zurück."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT job_id, doc_id, filename, status,
                       doc_class, chunk_count, error_message,
                       started_at, updated_at, finished_at
                FROM ingest_jobs WHERE job_id = $1
                """,
                job_id,
            )
        if not row:
            return None
        return {
            "job_id":        row["job_id"],
            "doc_id":        row["doc_id"],
            "filename":      row["filename"],
            "status":        row["status"],
            "doc_class":     row["doc_class"],
            "chunk_count":   row["chunk_count"],
            "error_message": row["error_message"],
            "started_at":    row["started_at"].isoformat() if row["started_at"] else None,
            "updated_at":    row["updated_at"].isoformat() if row["updated_at"] else None,
            "finished_at":   row["finished_at"].isoformat() if row["finished_at"] else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    async def run_pipeline(
        self,
        doc_id: str,
        job_id: str,
        file_content: bytes,
        filename: str,
        metadata,
        doc_class_override: Optional[str] = None,
        chunk_limit: Optional[int] = None,
    ) -> PipelineResult:
        """
        Vollständige M1-Pipeline.

        Args:
            doc_id:             Externe Dokument-UUID
            job_id:             Job-UUID für Status-Tracking
            file_content:       Rohe Datei-Bytes
            filename:           Originalname
            metadata:           DocumentMetadata
            doc_class_override: Dokumentklasse erzwingen (A/B/C)
            chunk_limit:        Nur N Chunks verarbeiten (für Tests)

        Returns:
            PipelineResult mit Status und Statistiken
        """
        log = logger.bind(doc_id=doc_id, job_id=job_id, filename=filename)
        log.info("Pipeline gestartet")

        await self._create_job(job_id, doc_id, filename)

        try:
            # ── Schritt 1: Rohdokument → MinIO ───────────────────────────────
            await self._update_job(job_id, "parsing")
            log.info("Schritt 1: Dokument in MinIO speichern")
            minio_path = await self.storage.store_raw_document(
                doc_id=doc_id,
                filename=filename,
                content=file_content,
            )

            # ── Schritt 2: Dokument-Metadaten → PostgreSQL ───────────────────
            log.info("Schritt 2: Dokument-Metadaten speichern")
            await self.storage.create_document_record(
                doc_id=doc_id,
                metadata=metadata,
                minio_path=minio_path,
            )

            # ── Schritt 3: Tika-Parsing ──────────────────────────────────────
            log.info("Schritt 3: Tika-Parsing")
            parsed = await self.parser.parse(
                content=file_content,
                filename=filename,
            )
            log.info("Parsing abgeschlossen",
                     doc_class=parsed.doc_class_hint,
                     char_count=len(parsed.text))

            # ── Schritt 3b: Abkürzungsauflösung ─────────────────────────────
            log.info("Schritt 3b: Abkürzungsauflösung")
            norm_result = self.normalizer.normalize(parsed.text)
            log.info(
                "Abkürzungen aufgelöst",
                count=len(norm_result.replacements),
                abbrevs=list({r.abbrev for r in norm_result.replacements}),
            )

            await self._update_job(job_id, "chunking",
                                   doc_class=parsed.doc_class_hint)

            # ── Schritt 4: Chunking (auf aufgelöstem Text) ───────────────────
            log.info("Schritt 4: Chunking")
            chunks = self.chunker.route_and_chunk(
                text=norm_result.resolved_text,   # aufgelöster Text
                structure=parsed.structure,
                doc_id=doc_id,
                metadata=metadata,
                doc_class_override=doc_class_override,
            )

            # Originaltext und abbrev_map in jeden Chunk eintragen
            import json
            abbrev_map_json = norm_result.abbrev_map
            for chunk in chunks:
                # Originaltext-Ausschnitt für diesen Chunk bestimmen
                chunk.content_original = norm_result.get_original_snippet(
                    0, len(norm_result.original_text)
                ) if not abbrev_map_json else None
                chunk.abbrev_map = abbrev_map_json if abbrev_map_json else None
            log.info("Chunking abgeschlossen", chunk_count=len(chunks))

            # Optionaler Chunk-Limit (für Tests)
            if chunk_limit and chunk_limit > 0:
                chunks = chunks[:chunk_limit]
                log.info("Chunk-Limit aktiv", limit=chunk_limit)

            await self._update_job(job_id, "embedding",
                                   chunk_count=len(chunks))

            # ── Schritt 5: Embeddings ────────────────────────────────────────
            log.info("Schritt 5: Embeddings erzeugen")
            chunks_with_embeddings = await self.embedder.embed_chunks(chunks)

            # ── Schritt 6: PostgreSQL speichern ──────────────────────────────
            await self._update_job(job_id, "storing")
            log.info("Schritt 6: Chunks in PostgreSQL speichern")
            inserted = await self.storage.store_chunks(chunks_with_embeddings)

            # ── Fertig ───────────────────────────────────────────────────────
            await self._update_job(job_id, "done", chunk_count=inserted)
            log.info("Pipeline abgeschlossen", chunk_count=inserted)

            return PipelineResult(
                job_id=job_id,
                doc_id=doc_id,
                filename=filename,
                status="done",
                doc_class=parsed.doc_class_hint,
                chunk_count=inserted,
            )

        except Exception as e:
            error_msg = str(e)
            log.error("Pipeline fehlgeschlagen", error=error_msg)
            await self._update_job(job_id, "error", error_message=error_msg)
            return PipelineResult(
                job_id=job_id,
                doc_id=doc_id,
                filename=filename,
                status="error",
                doc_class=None,
                chunk_count=0,
                error=error_msg,
            )
