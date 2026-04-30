# services/ingest/app/api/routes/paket.py
#
# Endpoints für den Paket-basierten Ingest.
#
# Ablauf:
#   1. POST /api/v1/ingest/paket
#      → Paket-JSON entgegennehmen
#      → Dokumente aus MinIO laden (via dokument_ids)
#      → Für jedes Dokument einen Ingest-Job starten
#      → Paket-ID + Job-IDs zurückgeben
#
#   2. GET /api/v1/ingest/paket/{paket_id}
#      → Aggregierten Status aller Jobs im Paket zurückgeben

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import structlog

from app.services.ingest_service import IngestService
from app.services.paket_service import PaketService

logger = structlog.get_logger()
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PaketRequest(BaseModel):
    """
    Eingehendes Paket-JSON.
    Entspricht exakt dem vereinbarten Format.
    """
    version_id:    str = Field(..., description="UUID der Version")
    paket_id:      str = Field(..., description="UUID des Pakets")
    paket_name:    str = Field(..., description="Lesbarer Name des Pakets")
    version:       str = Field(..., description="Versionsnummer, z.B. 1.0.3")
    manifest_hash: Optional[str] = Field(None, description="Hash des Manifests")
    dokument_ids:  list[str] = Field(..., description="Liste der Dokument-IDs in MinIO")

    # Optionale Pipeline-Steuerung
    force_class:   Optional[str] = Field(None, description="Dokumentklasse erzwingen: A, B oder C")
    chunk_limit:   Optional[int] = Field(None, description="Max. Chunks pro Dokument (0=alle)")
    source_type:   str = Field("gesetz", description="Dokumenttyp für alle Dokumente im Paket")


class JobStatus(BaseModel):
    job_id:      str
    doc_id:      str
    status:      str
    doc_class:   Optional[str]
    chunk_count: Optional[int]
    error:       Optional[str]


class PaketResponse(BaseModel):
    paket_id:    str
    paket_name:  str
    version:     str
    status:      str
    total_docs:  int
    jobs:        list[JobStatus]
    message:     str


class PaketStatusResponse(BaseModel):
    paket_id:    str
    paket_name:  str
    version:     str
    status:      str
    total_docs:  int
    done_docs:   int
    error_docs:  int
    pending_docs: int
    jobs:        list[JobStatus]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/paket", response_model=PaketResponse)
async def ingest_paket(
    paket:            PaketRequest,
    request:          Request,
    background_tasks: BackgroundTasks,
):
    """
    Paket entgegennehmen und alle Dokumente in die Pipeline einreihen.

    Für jede dokument_id in der Liste:
      1. Dokument aus MinIO laden (Bucket: mnr-dokumente)
      2. Neuen Ingest-Job anlegen
      3. Pipeline als Background-Task starten

    Antwortet sofort mit allen job_ids – Status per GET /paket/{paket_id}.
    """
    if paket.force_class and paket.force_class not in ("A", "B", "C"):
        raise HTTPException(
            status_code=422,
            detail="force_class muss A, B oder C sein."
        )

    if not paket.dokument_ids:
        raise HTTPException(
            status_code=422,
            detail="dokument_ids darf nicht leer sein."
        )

    ingest_service: IngestService = request.app.state.ingest_service
    paket_service:  PaketService  = request.app.state.paket_service

    logger.info(
        "Paket empfangen",
        paket_id=paket.paket_id,
        paket_name=paket.paket_name,
        doc_count=len(paket.dokument_ids),
    )

    # Paket-Datensatz anlegen
    await paket_service.create_paket(paket)

    # Für jedes Dokument: MinIO laden + Job starten
    jobs: list[JobStatus] = []

    for doc_id in paket.dokument_ids:
        job_id = str(uuid.uuid4())

        # Prüfen ob Dokument in MinIO existiert
        file_content, filename = await paket_service.load_from_minio(doc_id)

        if file_content is None:
            # Dokument nicht gefunden → Job mit Fehler anlegen
            logger.warning("Dokument nicht in MinIO gefunden", doc_id=doc_id)
            # REIHENFOLGE: erst ingest_jobs, dann ingest_paket_jobs (FK!)
            await ingest_service._create_job(job_id, doc_id, doc_id)
            await ingest_service._update_job(
                job_id, "error",
                error_message=f"Dokument {doc_id} nicht in MinIO gefunden."
            )
            await paket_service.register_job(
                paket_id=paket.paket_id,
                job_id=job_id,
                doc_id=doc_id,
            )
            jobs.append(JobStatus(
                job_id=job_id, doc_id=doc_id,
                status="error", doc_class=None, chunk_count=None,
                error=f"Dokument {doc_id} nicht in MinIO gefunden.",
            ))
            continue

        # REIHENFOLGE: erst ingest_jobs anlegen, dann ingest_paket_jobs (FK!)
        await ingest_service._create_job(job_id, doc_id, filename)
        await paket_service.register_job(
            paket_id=paket.paket_id,
            job_id=job_id,
            doc_id=doc_id,
        )

        # Metadaten für Pipeline
        class PaketMetadata:
            source_type    = paket.source_type
            title          = filename
            jurisdiction   = None
            valid_from     = None
            valid_to       = None
            norm_reference = None
            version        = paket.version
            language       = "de"
            register_scope = None

        # Pipeline als Background-Task starten
        background_tasks.add_task(
            paket_service.run_doc_pipeline,
            paket_id=paket.paket_id,
            job_id=job_id,
            doc_id=doc_id,
            file_content=file_content,
            filename=filename,
            metadata=PaketMetadata(),
            doc_class_override=paket.force_class,
            chunk_limit=paket.chunk_limit,
        )

        jobs.append(JobStatus(
            job_id=job_id, doc_id=doc_id,
            status="queued", doc_class=None, chunk_count=None, error=None,
        ))

        logger.info("Job eingereiht", job_id=job_id, doc_id=doc_id)

    return PaketResponse(
        paket_id=paket.paket_id,
        paket_name=paket.paket_name,
        version=paket.version,
        status="queued",
        total_docs=len(paket.dokument_ids),
        jobs=jobs,
        message=(
            f"{len(jobs)} Jobs eingereiht. "
            f"Status: GET /api/v1/ingest/paket/{paket.paket_id}"
        ),
    )


@router.get("/paket/{paket_id}", response_model=PaketStatusResponse)
async def get_paket_status(paket_id: str, request: Request):
    """
    Aggregierten Status aller Jobs eines Pakets abfragen.

    Status-Werte des Pakets:
      queued     → noch kein Job gestartet
      processing → mindestens ein Job läuft noch
      done       → alle Jobs erfolgreich
      partial    → einige Jobs fehlgeschlagen
      error      → alle Jobs fehlgeschlagen
    """
    paket_service: PaketService = request.app.state.paket_service
    status = await paket_service.get_paket_status(paket_id)

    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Paket '{paket_id}' nicht gefunden."
        )
    return status
