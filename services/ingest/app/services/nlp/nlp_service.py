# services/ingest/app/services/nlp/nlp_service.py
#
# NLPService: Orchestriert die vollständige NLP-Pipeline (M2).
# Option A: Post-Ingest – läuft asynchron nach dem Dokument-Ingest.
#
# Kann jederzeit für einzelne Dokumente oder alle Chunks neu gestartet
# werden. Alte Ergebnisse werden bei overwrite_existing=true gelöscht.

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import asyncpg
import structlog


def _to_uuid(v):
    """Konvertiert String oder asyncpg-UUID zu uuid.UUID."""
    if isinstance(v, str):
        return uuid.UUID(v)
    return v  # asyncpg gibt bereits UUID-Objekte zurück

from app.core.config import settings
from app.services.nlp.nlp_processor import NLPProcessor, load_nlp_config
from app.services.nlp.svo_extractor import SVOExtractor
from app.services.nlp.ner_extractor import NERExtractor
from app.services.question_filter import QuestionFilter, load_question_config

logger = structlog.get_logger()


class NLPService:
    """
    Orchestriert die M2-NLP-Pipeline:
      1. Chunks aus PostgreSQL laden
      2. spaCy-Analyse
      3. SVO-Extraktion
      4. NER-Extraktion
      5. Ergebnisse in svo_extractions + ner_entities speichern

    Jederzeit neu ausführbar – alte Ergebnisse werden bei Bedarf gelöscht.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.processor   = NLPProcessor(config_path)
        self.svo         = SVOExtractor(config_path)
        self.ner         = NERExtractor(config_path)
        self.qfilter     = QuestionFilter(config_path)
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=settings.postgres_host, port=settings.postgres_port,
                user=settings.postgres_user, password=settings.postgres_password,
                database=settings.postgres_db, min_size=2, max_size=10,
            )
        return self._pool

    async def close(self):
        if self._pool:
            await self._pool.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Öffentliche API
    # ─────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        doc_id: Optional[str] = None,   # None = alle Dokumente
        job_id: Optional[str] = None,   # None = neuer Job-ID
    ) -> dict:
        """
        Startet einen NLP-Job.

        Args:
            doc_id: Optional – nur dieses Dokument verarbeiten
            job_id: Optional – vorhandenen Job-ID verwenden (für Restart)

        Returns:
            Job-Statusdict
        """
        cfg      = load_nlp_config(self.config_path)
        job_id   = job_id or str(uuid.uuid4())
        log      = logger.bind(job_id=job_id, doc_id=doc_id or "alle")

        log.info("NLP-Job gestartet")

        # Job anlegen
        await self._create_job(job_id, doc_id, cfg)

        try:
            # Chunks laden
            chunks = await self._load_chunks(doc_id, cfg)
            log.info("Chunks geladen", count=len(chunks))

            if not chunks:
                await self._update_job(job_id, "done", chunks_total=0)
                return await self.get_job_status(job_id)

            await self._update_job(job_id, "running", chunks_total=len(chunks))

            # Alte Ergebnisse löschen wenn overwrite_existing=true
            if cfg.get("worker", {}).get("overwrite_existing", True):
                deleted = await self._delete_existing(
                    doc_id=doc_id,
                    chunk_ids=[c["id"] for c in chunks],
                )
                log.info("Alte Ergebnisse gelöscht", count=deleted)

            # Batch-Größe aus Config
            batch_size = cfg.get("worker", {}).get("chunk_batch_size", 50)
            show_prog  = cfg.get("worker", {}).get("show_progress", True)

            svo_total = 0
            ner_total = 0

            # Chunks in Batches verarbeiten
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                if show_prog:
                    print(
                        f"\r  NLP: {i}/{len(chunks)} Chunks "
                        f"| SVO: {svo_total} | NER: {ner_total}",
                        end="", flush=True,
                    )

                # NLP-Filter: Frage-Chunks überspringen wenn nlp.enabled=true
                nlp_qcfg = load_question_config(self.config_path, scope="nlp")
                nlp_q_enabled  = nlp_qcfg.get("enabled", False)
                nlp_q_classes  = nlp_qcfg.get("apply_to_classes", ["B","C"])
                nlp_q_action   = nlp_qcfg.get("action", "skip")

                filtered_batch = []
                skipped_batch  = []
                for c in batch:
                    if (nlp_q_enabled
                            and c.get("doc_class") in nlp_q_classes
                            and nlp_q_action == "skip"):
                        # Temporären Chunk-Stub bauen für QuestionFilter
                        from app.services.chunker import Chunk
                        stub = Chunk(
                            chunk_id=str(c["id"]),
                            doc_id=str(c["doc_id"]),
                            doc_class=c["doc_class"],
                            text=c["content"],
                        )
                        result = self.qfilter.filter(
                            chunks=[stub],
                            doc_class=c["doc_class"],
                            scope="nlp",
                        )
                        if result.filtered_chunks:
                            skipped_batch.append(c)
                            continue
                    filtered_batch.append(c)

                if skipped_batch:
                    logger.debug(
                        "NLP-Filter: Frage-Chunks übersprungen",
                        count=len(skipped_batch),
                    )

                # spaCy-Analyse (Batch) – nur nicht-gefilterte Chunks
                analyses = self.processor.analyze_batch(
                    [(str(c["id"]), c["content"]) for c in filtered_batch]
                )

                # SVO + NER für jeden Chunk
                svo_rows = []
                ner_rows = []

                for chunk, analysis in zip(filtered_batch, analyses):
                    chunk_id = str(chunk["id"])
                    doc_uuid = str(chunk["doc_id"])

                    # SVO
                    svos = self.svo.extract(analysis)
                    for s in svos:
                        svo_rows.append((
                            chunk_id, doc_uuid, job_id,
                            s.subject, s.subject_type,
                            s.predicate, s.predicate_lemma,
                            s.object, s.object_type,
                            s.context, s.norm_type,
                            s.norm_type_confidence,
                            s.confidence, s.sentence_text,
                        ))

                    # NER
                    entities = self.ner.extract(analysis)
                    for e in entities:
                        ner_rows.append((
                            chunk_id, doc_uuid, job_id,
                            e.text, e.label,
                            e.start_char, e.end_char,
                            e.confidence, e.source,
                        ))

                # Batch in DB schreiben
                s_count, n_count = await self._insert_batch(svo_rows, ner_rows)
                svo_total += s_count
                ner_total += n_count

                # Job-Fortschritt aktualisieren
                await self._update_job(
                    job_id, "running",
                    chunks_done=i + len(batch),
                    svo_count=svo_total,
                    ner_count=ner_total,
                )

            if show_prog:
                print()  # Newline nach Progress-Anzeige

            await self._update_job(
                job_id, "done",
                chunks_done=len(chunks),
                svo_count=svo_total,
                ner_count=ner_total,
            )
            log.info("NLP-Job abgeschlossen",
                     svo=svo_total, ner=ner_total, chunks=len(chunks))

        except Exception as e:
            log.error("NLP-Job fehlgeschlagen", error=str(e))
            await self._update_job(job_id, "error", error_message=str(e))

        return await self.get_job_status(job_id)

    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """Gibt aktuellen Job-Status zurück."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT job_id, doc_id, status, chunks_total, chunks_done,
                       svo_count, ner_count, error_message,
                       started_at, updated_at, finished_at
                FROM nlp_jobs WHERE job_id = $1
                """, job_id,
            )
        if not row:
            return None
        return {
            "job_id":        row["job_id"],
            "doc_id":        row["doc_id"] or "alle",
            "status":        row["status"],
            "chunks_total":  row["chunks_total"],
            "chunks_done":   row["chunks_done"],
            "svo_count":     row["svo_count"],
            "ner_count":     row["ner_count"],
            "error_message": row["error_message"],
            "started_at":    row["started_at"].isoformat() if row["started_at"] else None,
            "updated_at":    row["updated_at"].isoformat() if row["updated_at"] else None,
            "finished_at":   row["finished_at"].isoformat() if row["finished_at"] else None,
        }

    async def list_jobs(self, limit: int = 20) -> list[dict]:
        """Listet letzte NLP-Jobs auf."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT job_id, doc_id, status, chunks_total, chunks_done,
                       svo_count, ner_count, started_at, finished_at
                FROM nlp_jobs ORDER BY started_at DESC LIMIT $1
                """, limit,
            )
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────────
    # DB-Hilfsmethoden
    # ─────────────────────────────────────────────────────────────────────────

    async def _load_chunks(self, doc_id: Optional[str], cfg: dict) -> list:
        """Lädt Chunks aus PostgreSQL gemäß Konfiguration."""
        pool       = await self._get_pool()
        classes    = cfg.get("worker", {}).get("process_classes", ["A","B","C"])
        min_tokens = cfg.get("worker", {}).get("min_token_count", 10)

        async with pool.acquire() as conn:
            if doc_id:
                rows = await conn.fetch(
                    """
                    SELECT nc.id, nc.doc_id, nc.content, nc.doc_class,
                           nc.norm_reference, nc.chunk_type
                    FROM norm_chunks nc
                    JOIN norm_documents nd ON nc.doc_id = nd.id
                    WHERE nd.doc_id = $1
                      AND nc.doc_class = ANY($2::text[])
                      AND nc.token_count >= $3
                    ORDER BY nc.created_at
                    """,
                    doc_id, classes, min_tokens,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, doc_id, content, doc_class,
                           norm_reference, chunk_type
                    FROM norm_chunks
                    WHERE doc_class = ANY($1::text[])
                      AND token_count >= $2
                    ORDER BY created_at
                    """,
                    classes, min_tokens,
                )
        return [dict(r) for r in rows]

    async def _delete_existing(
        self,
        doc_id: Optional[str],
        chunk_ids: list[str],
    ) -> int:
        """Löscht alte SVO/NER-Ergebnisse für die gegebenen Chunks."""
        pool = await self._get_pool()
        ids  = [_to_uuid(cid) for cid in chunk_ids]
        async with pool.acquire() as conn:
            r1 = await conn.execute(
                "DELETE FROM svo_extractions WHERE chunk_id = ANY($1)", ids
            )
            r2 = await conn.execute(
                "DELETE FROM ner_entities WHERE chunk_id = ANY($1)", ids
            )
        # Anzahl gelöschter Zeilen
        return int(r1.split()[-1]) + int(r2.split()[-1])

    async def _insert_batch(
        self,
        svo_rows: list[tuple],
        ner_rows: list[tuple],
    ) -> tuple[int, int]:
        """Schreibt SVO- und NER-Ergebnisse als Batch in die DB."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if svo_rows:
                await conn.executemany(
                    """
                    INSERT INTO svo_extractions (
                        chunk_id, doc_id, nlp_job_id,
                        subject, subject_type,
                        predicate, predicate_lemma,
                        object, object_type, context,
                        norm_type, norm_type_confidence,
                        confidence, sentence_text
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    """,
                    [(
                        _to_uuid(r[0]), _to_uuid(r[1]), r[2],
                        r[3], r[4], r[5], r[6], r[7], r[8],
                        r[9], r[10], r[11], r[12], r[13],
                    ) for r in svo_rows],
                )
            if ner_rows:
                await conn.executemany(
                    """
                    INSERT INTO ner_entities (
                        chunk_id, doc_id, nlp_job_id,
                        text, label, start_char, end_char,
                        confidence, source
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    """,
                    [(
                        _to_uuid(r[0]), _to_uuid(r[1]), r[2],
                        r[3], r[4], r[5], r[6], r[7], r[8],
                    ) for r in ner_rows],
                )
        return len(svo_rows), len(ner_rows)

    async def _create_job(
        self, job_id: str, doc_id: Optional[str], cfg: dict
    ):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO nlp_jobs (job_id, doc_id, status, config_snapshot)
                VALUES ($1, $2, 'queued', $3::jsonb)
                ON CONFLICT (job_id) DO NOTHING
                """,
                job_id, doc_id, json.dumps(cfg),
            )

    async def _update_job(
        self, job_id: str, status: str, **kwargs
    ):
        pool     = await self._get_pool()
        finished = "now()" if status in ("done","error") else "NULL"
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE nlp_jobs SET
                    status          = $1,
                    chunks_total    = COALESCE($2, chunks_total),
                    chunks_done     = COALESCE($3, chunks_done),
                    svo_count       = COALESCE($4, svo_count),
                    ner_count       = COALESCE($5, ner_count),
                    error_message   = COALESCE($6, error_message),
                    updated_at      = now(),
                    finished_at     = {finished}
                WHERE job_id = $7
                """,
                status,
                kwargs.get("chunks_total"),
                kwargs.get("chunks_done"),
                kwargs.get("svo_count"),
                kwargs.get("ner_count"),
                kwargs.get("error_message"),
                job_id,
            )
