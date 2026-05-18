# services/ingest/app/api/routes/rag_router.py
#
# M3 – RAG-Engine FastAPI-Endpunkte.
#
# Aktueller Stand: Gerüst mit Mock-Antworten.
# Die eigentliche Logik wird schrittweise in:
#   app/services/rag/query_transformer.py  (Schritt 1.1)
#   app/services/rag/retriever.py          (Schritt 1.2-1.4)
#   app/services/rag/reranker.py           (Schritt 1.5)
#   app/services/rag/context_assembler.py  (Schritt 1.6)
#   app/services/rag/rag_pipeline.py       (Orchestrierung)
# implementiert.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

_RAG_CFG_PATH = Path(__file__).parents[3] / "rag_config.yaml"


# ── Request / Response Modelle ────────────────────────────────────────────────

class RAGRequest(BaseModel):
    query:          str
    doc_filter:     Optional[str]  = None   # Dokument-Filter (z.B. "NHundG")
    top_k:          Optional[int]  = None   # Überschreibt rag_config.yaml
    json_mode:      bool           = False  # Strukturierte JSON-Antwort
    debug:          bool           = False  # QueryBundle in Antwort einschließen


class RAGChunk(BaseModel):
    chunk_id:        str
    doc_class:       str
    norm_reference:  Optional[str]
    chunk_type:      str
    content:         str
    score:           float
    confidence_weight: float
    praembel:        str            # Kontext-Präambel für LLM


class RAGResponse(BaseModel):
    query:           str
    query_typ:       str            # NORM | ENTITY | IM | GENERAL
    chunks:          list[RAGChunk]
    kontext:         str            # Assemblierter Kontext-String für LLM
    traceability:    list[dict]     # Quellenverweise je Kontext-Quelle
    direktlookup:    bool           # True wenn §-Referenz direkt gefunden
    debug_info:      Optional[dict] # QueryBundle (nur wenn debug=True)


# ── Endpunkte ─────────────────────────────────────────────────────────────────

@router.get("/status", summary="RAG-Engine Status")
async def rag_status():
    """
    Gibt den Status der RAG-Engine zurück.
    Zeigt ob rag_config.yaml geladen werden kann.
    """
    try:
        cfg = yaml.safe_load(_RAG_CFG_PATH.read_text(encoding="utf-8"))
        return {
            "status":        "bereit",
            "config_geladen": True,
            "top_k_final":   cfg.get("retrieval", {}).get("top_k_final", 8),
            "hyde_enabled":  cfg.get("query_transformation", {})
                               .get("hyde", {}).get("enabled", False),
            "reranker":      cfg.get("reranker", {}).get("modell", "–"),
            "hinweis":       "RAG-Pipeline in Entwicklung (M3 Schritt 1)",
        }
    except FileNotFoundError:
        raise HTTPException(
            404,
            "rag_config.yaml nicht gefunden – Datei in services/ingest/ ablegen"
        )
    except Exception as e:
        raise HTTPException(500, f"Config-Fehler: {str(e)}")


@router.get("/config", summary="RAG-Konfiguration lesen")
async def rag_config():
    """Gibt die aktuelle rag_config.yaml als JSON zurück."""
    try:
        cfg = yaml.safe_load(_RAG_CFG_PATH.read_text(encoding="utf-8"))
        return cfg
    except FileNotFoundError:
        raise HTTPException(404, "rag_config.yaml nicht gefunden")


@router.post("/query", summary="RAG-Abfrage ausführen", response_model=RAGResponse)
async def rag_query(req: RAGRequest, request: Request):
    """
    Führt eine RAG-Abfrage aus.

    Ablauf (nach vollständiger Implementierung):
      1. Query-Transformation (1.1): Typ, HyDE, Step-Back, Filter
      2. Retrieval (1.2-1.4):        Vektor + FTS, Parent-Child, Cross-Ref
      3. Re-Ranking (1.5):           Cross-Encoder + Klassen-Gewichtung
      4. Context Assembly (1.6):     Präambeln + IM-Signale

    Aktuell: Mock-Antwort (Gerüst-Phase)
    """
    # ── RAG-Pipeline (Schritte 1.1–1.6) ─────────────────────────────────────
    from app.services.rag.rag_pipeline import RAGPipeline

    ingest_service = request.app.state.ingest_service
    pool           = await ingest_service._get_pool()

    pipeline = RAGPipeline(
        pool     = pool,
        embedder = ingest_service.embedder,
    )
    result = await pipeline.run(
        query = req.query,
        debug = req.debug,
    )

    rag_chunks = [
        RAGChunk(
            chunk_id          = c.chunk_id,
            doc_class         = c.doc_class,
            norm_reference    = c.norm_reference,
            chunk_type        = c.chunk_type,
            content           = c.content,
            score             = round(c.score, 4),
            confidence_weight = c.confidence_weight,
            praembel          = _build_praembel(c),
        )
        for c in result.chunks
    ]

    return RAGResponse(
        query        = result.original_query,
        query_typ    = result.query_typ,
        chunks       = rag_chunks,
        kontext      = result.kontext,
        traceability = result.traceability,
        direktlookup = result.direktlookup,
        debug_info   = {
            **result.stats,
            **(result.debug or {}),
        } if req.debug else None,
    )


@router.post("/reload", summary="RAG-Konfiguration neu laden")
async def rag_reload():
    """
    Lädt rag_config.yaml neu.
    Nach Konfigurationsänderungen aufrufen.
    """
    try:
        yaml.safe_load(_RAG_CFG_PATH.read_text(encoding="utf-8"))
        return {
            "status":  "ok",
            "message": "rag_config.yaml neu geladen",
        }
    except Exception as e:
        raise HTTPException(500, f"Reload fehlgeschlagen: {str(e)}")

def _build_praembel(c) -> str:
    label_map = {"A": "RECHTSNORM", "B": "FACHKONZEPT", "C": "AUSLEGUNG"}
    label     = label_map.get(c.doc_class, "NORM")
    parts     = [label]
    if c.norm_reference:
        parts.append(c.norm_reference)
    elif c.heading_breadcrumb:
        parts.append(c.heading_breadcrumb)
    if c.chunk_type and c.chunk_type not in ("tatbestand", ""):
        parts.append(c.chunk_type.upper())
    return "[" + " | ".join(parts) + "]"
