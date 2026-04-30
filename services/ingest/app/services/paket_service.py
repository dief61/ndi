# services/ingest/app/services/paket_service.py
#
# PaketService: Verwaltet Pakete und lädt Dokumente aus MinIO
# für die Wiederverarbeitung.

from __future__ import annotations

import io
from typing import Optional, Tuple

import asyncpg
import structlog

from app.core.config import settings
from app.services.ingest_service import IngestService
from app.services.storage import DocumentStorage

logger = structlog.get_logger()


class PaketService:
    """
    Verwaltet den Lebenszyklus von Ingest-Paketen.

    Aufgaben:
      - Paket-Datensatz in PostgreSQL anlegen
      - Dokumente aus MinIO laden (via doc_id)
      - Pipeline für jedes Dokument delegieren
      - Paket-Status aggregieren
    """

    def __init__(self, ingest_service: IngestService):
        self.ingest_service = ingest_service
        self.storage        = DocumentStorage()
        self._pg_pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
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
        if self._pg_pool:
            await self._pg_pool.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Paket anlegen
    # ─────────────────────────────────────────────────────────────────────────

    async def create_paket(self, paket) -> None:
        """Legt den Paket-Datensatz in ingest_pakete an."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_pakete (
                    paket_id, version_id, paket_name, version,
                    manifest_hash, dokument_ids, status, total_docs
                )
                VALUES ($1, $2, $3, $4, $5, $6, 'queued', $7)
                ON CONFLICT (paket_id) DO UPDATE
                    SET updated_at = now()
                """,
                paket.paket_id,
                paket.version_id,
                paket.paket_name,
                paket.version,
                paket.manifest_hash,
                paket.dokument_ids,
                len(paket.dokument_ids),
            )
        logger.info("Paket angelegt",
                    paket_id=paket.paket_id,
                    total_docs=len(paket.dokument_ids))

    async def register_job(
        self,
        paket_id: str,
        job_id:   str,
        doc_id:   str,
    ) -> None:
        """Verknüpft einen Job mit einem Paket."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_paket_jobs (paket_id, job_id, doc_id)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                """,
                paket_id, job_id, doc_id,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Dokument aus MinIO laden
    # ─────────────────────────────────────────────────────────────────────────

    async def load_from_minio(
        self,
        doc_id: str,
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Lädt das Rohdokument für eine doc_id aus MinIO.

        Sucht im Bucket mnr-dokumente unter dem Pfad {doc_id}/{filename}.
        Gibt (file_content, filename) zurück oder (None, None) wenn nicht gefunden.
        """
        minio = self.storage._get_minio()
        bucket = "mnr-dokumente"

        try:
            # Alle Objekte unter dem doc_id-Prefix auflisten
            objects = list(minio.list_objects(bucket, prefix=f"{doc_id}/"))

            if not objects:
                logger.warning(
                    "Kein Objekt in MinIO gefunden",
                    doc_id=doc_id,
                    bucket=bucket,
                )
                return None, None

            # Erstes Objekt nehmen (pro doc_id gibt es genau eine Datei)
            obj = objects[0]
            filename = obj.object_name.split("/")[-1]

            # Datei herunterladen
            response = minio.get_object(bucket, obj.object_name)
            file_content = response.read()
            response.close()
            response.release_conn()

            logger.info(
                "Dokument aus MinIO geladen",
                doc_id=doc_id,
                filename=filename,
                size_bytes=len(file_content),
            )
            return file_content, filename

        except Exception as e:
            logger.error(
                "Fehler beim Laden aus MinIO",
                doc_id=doc_id,
                error=str(e),
            )
            return None, None

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline-Delegierung
    # ─────────────────────────────────────────────────────────────────────────

    async def run_doc_pipeline(
        self,
        paket_id:           str,
        job_id:             str,
        doc_id:             str,
        file_content:       bytes,
        filename:           str,
        metadata,
        doc_class_override: Optional[str] = None,
        chunk_limit:        Optional[int]  = None,
    ) -> None:
        """
        Führt die Pipeline für ein einzelnes Dokument aus dem Paket aus
        und aktualisiert anschließend den Paket-Status.
        """
        # Paket auf 'processing' setzen (einmalig)
        await self._set_paket_processing(paket_id)

        # Pipeline ausführen (delegiert an IngestService)
        # Hinweis: _create_job wurde bereits im Endpoint aufgerufen
        result = await self.ingest_service.run_pipeline(
            doc_id=doc_id,
            job_id=job_id,
            file_content=file_content,
            filename=filename,
            metadata=metadata,
            doc_class_override=doc_class_override,
            chunk_limit=chunk_limit,
        )

        # Paket-Statistik aktualisieren
        await self._update_paket_stats(paket_id, result.status)

    async def _set_paket_processing(self, paket_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingest_pakete
                SET status = 'processing', updated_at = now()
                WHERE paket_id = $1 AND status = 'queued'
                """,
                paket_id,
            )

    async def _update_paket_stats(self, paket_id: str, job_status: str) -> None:
        """Aktualisiert done_docs / error_docs und leitet Gesamtstatus ab."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if job_status == "done":
                await conn.execute(
                    """
                    UPDATE ingest_pakete
                    SET done_docs = done_docs + 1, updated_at = now()
                    WHERE paket_id = $1
                    """,
                    paket_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE ingest_pakete
                    SET error_docs = error_docs + 1, updated_at = now()
                    WHERE paket_id = $1
                    """,
                    paket_id,
                )

            # Gesamtstatus berechnen
            row = await conn.fetchrow(
                """
                SELECT total_docs, done_docs, error_docs
                FROM ingest_pakete WHERE paket_id = $1
                """,
                paket_id,
            )

            if row:
                total   = row["total_docs"]
                done    = row["done_docs"]
                errors  = row["error_docs"]
                finished = (done + errors) >= total

                if finished:
                    if errors == 0:
                        new_status = "done"
                    elif done == 0:
                        new_status = "error"
                    else:
                        new_status = "partial"

                    await conn.execute(
                        """
                        UPDATE ingest_pakete
                        SET status = $1, finished_at = now(), updated_at = now()
                        WHERE paket_id = $2
                        """,
                        new_status, paket_id,
                    )
                    logger.info(
                        "Paket abgeschlossen",
                        paket_id=paket_id,
                        status=new_status,
                        done=done,
                        errors=errors,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # Status abfragen
    # ─────────────────────────────────────────────────────────────────────────

    async def get_paket_status(self, paket_id: str) -> Optional[dict]:
        """Gibt den aggregierten Paket-Status zurück."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            paket = await conn.fetchrow(
                """
                SELECT paket_id, paket_name, version, status,
                       total_docs, done_docs, error_docs
                FROM ingest_pakete WHERE paket_id = $1
                """,
                paket_id,
            )
            if not paket:
                return None

            # Jobs des Pakets laden
            jobs = await conn.fetch(
                """
                SELECT pj.job_id, pj.doc_id,
                       ij.status, ij.doc_class,
                       ij.chunk_count, ij.error_message
                FROM ingest_paket_jobs pj
                LEFT JOIN ingest_jobs ij ON pj.job_id = ij.job_id
                WHERE pj.paket_id = $1
                ORDER BY pj.created_at
                """,
                paket_id,
            )

        pending = (
            paket["total_docs"]
            - paket["done_docs"]
            - paket["error_docs"]
        )

        return {
            "paket_id":    paket["paket_id"],
            "paket_name":  paket["paket_name"],
            "version":     paket["version"],
            "status":      paket["status"],
            "total_docs":  paket["total_docs"],
            "done_docs":   paket["done_docs"],
            "error_docs":  paket["error_docs"],
            "pending_docs": max(0, pending),
            "jobs": [
                {
                    "job_id":      row["job_id"],
                    "doc_id":      row["doc_id"],
                    "status":      row["status"] or "queued",
                    "doc_class":   row["doc_class"],
                    "chunk_count": row["chunk_count"],
                    "error":       row["error_message"],
                }
                for row in jobs
            ],
        }
