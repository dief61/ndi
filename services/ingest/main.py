# services/ingest/main.py
# NDI Ingest-Service – FastAPI Hauptapplikation

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.api.routes import ingest, health, paket
from app.core.config import settings
from app.services.ingest_service import IngestService
from app.services.paket_service import PaketService

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: Startup / Shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NDI Ingest-Service startet", version="0.1.0")

    # IngestService – Singleton (Embedder lädt Modell einmalig)
    ingest_service = IngestService()
    app.state.ingest_service = ingest_service

    # PaketService – nutzt denselben IngestService
    paket_service = PaketService(ingest_service=ingest_service)
    app.state.paket_service = paket_service

    logger.info(
        "Services initialisiert",
        embedding_model=ingest_service.embedder.active_name,
        device=ingest_service.embedder.device,
    )

    yield  # ← Service läuft

    logger.info("NDI Ingest-Service wird beendet")
    await ingest_service.close()
    await paket_service.close()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NDI Ingest-Service",
    description=(
        "Dokument-Ingestion, Chunking und Embedding "
        "für das Meta-Normen-Register (MNR)"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router einbinden
app.include_router(health.router, prefix="/health",        tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
app.include_router(paket.router,  prefix="/api/v1/ingest", tags=["Paket"])
