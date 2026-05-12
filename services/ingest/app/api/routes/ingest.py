# services/ingest/app/api/routes/ingest.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Optional
from datetime import date
import uuid
import structlog

from app.services.ingest_service import IngestService

logger = structlog.get_logger()
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    source_type:    str = "gesetz"
    title:          str = "Unbekanntes Dokument"
    jurisdiction:   Optional[str] = None
    valid_from:     Optional[date] = None
    valid_to:       Optional[date] = None
    norm_reference: Optional[str] = None
    version:        Optional[str] = None
    language:       str = "de"
    register_scope: Optional[list[str]] = None


class IngestResponse(BaseModel):
    job_id:   str
    doc_id:   str
    status:   str
    message:  str


# ─────────────────────────────────────────────────────────────────────────────
# Erlaubte Dateiformate
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {
    # ── Textdokumente ──────────────────────────────────────────────────────
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/msword",                                                         # doc
    "application/rtf",
    "text/plain",
    "text/html",
    # ── OpenDocument (LibreOffice) ─────────────────────────────────────────
    "application/vnd.oasis.opendocument.text",           # odt
    "application/vnd.oasis.opendocument.presentation",   # odp
    "application/vnd.oasis.opendocument.spreadsheet",    # ods
    # ── Microsoft Office ──────────────────────────────────────────────────
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",       # xlsx
    "application/vnd.ms-excel",                                                  # xls
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # pptx → Klasse C
    "application/vnd.ms-powerpoint",                                              # ppt  → Klasse C
    # ── Strukturierte Daten / XÖV ──────────────────────────────────────────
    "application/xml",
    "text/xml",
    "application/json",
    "text/csv",
    "text/tab-separated-values",                         # tsv
    # ── E-Books ────────────────────────────────────────────────────────────
    "application/epub+zip",
    # ── E-Mail ─────────────────────────────────────────────────────────────
    "message/rfc822",                                    # eml
    "application/vnd.ms-outlook",                        # msg → Klasse C
}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/document", response_model=IngestResponse)
async def ingest_document(
    request:          Request,
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    source_type:      str = Form("gesetz"),
    title:            str = Form("Unbekanntes Dokument"),
    jurisdiction:     Optional[str] = Form(None),
    norm_reference:   Optional[str] = Form(None),
    version:          Optional[str] = Form(None),
    language:         str = Form("de"),
    force_class:           Optional[str] = Form(None),     # A | B | C
    source_type_explicit:  str           = Form("false"),   # Prio 1: CLI/API
    source_type_from_yaml: str           = Form("false"),   # Prio 2: docs.yaml
):
    """
    Dokument hochladen und Ingest-Pipeline starten.

    Die Pipeline läuft als Background-Task.
    Der Endpoint antwortet sofort mit job_id und doc_id.
    Status abfragen: GET /api/v1/ingest/status/{job_id}
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Dateiformat '{file.content_type}' nicht unterstützt. "
                f"Erlaubt: PDF, DOCX, DOC, RTF, TXT, HTML, "
                f"ODT, ODS, ODP, XLSX, XLS, PPTX, PPT, "
                f"XML, JSON, CSV, TSV, EPUB, EML, MSG"
            )
        )

    if force_class and force_class not in ("A", "B", "C"):
        raise HTTPException(
            status_code=422,
            detail="force_class muss A, B oder C sein."
        )

    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    metadata = DocumentMetadata(
        source_type=source_type,
        title=title,
        jurisdiction=jurisdiction,
        norm_reference=norm_reference,
        version=version,
        language=language,
    )
    # Prioritäts-Flags setzen
    # Prio 1: explizit per CLI/API übergeben
    metadata._source_type_explicit  = (
        source_type_explicit.lower() == "true"
        or source_type not in ("gesetz", "text", "")
    )
    # Prio 2: aus docs.yaml
    metadata._source_type_from_yaml = source_type_from_yaml.lower() == "true"

    # Dateiinhalt vor dem Background-Task lesen
    file_content = await file.read()
    filename     = file.filename

    logger.info("Ingest-Request empfangen",
                doc_id=doc_id, job_id=job_id,
                filename=filename, source_type=source_type)

    # IngestService vom App-State holen (Singleton)
    service: IngestService = request.app.state.ingest_service

    background_tasks.add_task(
        service.run_pipeline,
        doc_id=doc_id,
        job_id=job_id,
        file_content=file_content,
        filename=filename,
        metadata=metadata,
        doc_class_override=force_class,
    )

    return IngestResponse(
        job_id=job_id,
        doc_id=doc_id,
        status="queued",
        message=(
            f"Dokument '{filename}' in die Pipeline eingereiht. "
            f"Status: GET /api/v1/ingest/status/{job_id}"
        ),
    )


@router.get("/status/{job_id}")
async def get_ingest_status(job_id: str, request: Request):
    """
    Verarbeitungsstatus eines Ingest-Jobs abfragen.

    Status-Werte:
      queued    → in der Warteschlange
      parsing   → Tika-Parsing läuft
      chunking  → Chunking läuft
      embedding → Embedding-Berechnung läuft
      storing   → Speicherung in PostgreSQL
      done      → erfolgreich abgeschlossen
      error     → fehlgeschlagen (error_message enthält Details)
    """
    service: IngestService = request.app.state.ingest_service
    status = await service.get_job_status(job_id)

    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' nicht gefunden."
        )
    return status


@router.get("/jobs")
async def list_recent_jobs(request: Request, limit: int = 20):
    """
    Letzte Ingest-Jobs auflisten (neueste zuerst).
    """
    service: IngestService = request.app.state.ingest_service
    pool = await service._get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT job_id, doc_id, filename, status,
                   doc_class, chunk_count, started_at, finished_at
            FROM ingest_jobs
            ORDER BY started_at DESC
            LIMIT $1
            """,
            limit,
        )

    return [
        {
            "job_id":      row["job_id"],
            "doc_id":      row["doc_id"],
            "filename":    row["filename"],
            "status":      row["status"],
            "doc_class":   row["doc_class"],
            "chunk_count": row["chunk_count"],
            "started_at":  row["started_at"].isoformat() if row["started_at"] else None,
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        }
        for row in rows
    ]
