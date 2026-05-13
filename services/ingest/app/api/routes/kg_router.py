# services/ingest/app/api/routes/kg_router.py
#
# FastAPI-Endpoints für den Knowledge Graph.

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import asyncpg
from app.core.config import settings
from app.services.knowledge_graph import KnowledgeGraphService

router = APIRouter()


async def _conn():
    return await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )


@router.get("/ping", summary="Fuseki-Verbindung prüfen")
async def kg_ping():
    kg = KnowledgeGraphService()
    ok = await kg.ping()
    if not ok:
        raise HTTPException(503, "Fuseki nicht erreichbar – läuft der Container?")
    return {"status": "ok", "endpoint": kg.sparql_endpoint}


@router.post("/export", summary="Vollständigen KG-Export starten")
async def kg_export(
    doc_id:      Optional[str] = Query(None, description="Nur dieses Dokument"),
    incremental: bool          = Query(True,  description="Nur neue Chunks"),
    full:        bool          = Query(False,  description="Kompletter Re-Export"),
):
    conn = await _conn()
    kg   = KnowledgeGraphService()
    try:
        result = await kg.export_all(
            conn,
            doc_id=doc_id,
            incremental=not full,
        )
        return {
            "status":         "ok",
            "triples_svo":    result.triples_svo,
            "triples_ner":    result.triples_ner,
            "triples_norms":  result.triples_norms,
            "total_triples":  result.total_triples,
            "errors":         result.errors,
        }
    finally:
        await conn.close()


@router.get("/stats", summary="KG-Export-Statistik")
async def kg_stats():
    conn = await _conn()
    try:
        rows = await conn.fetch(
            "SELECT * FROM kg_sync_stats ORDER BY title"
        )
        return {"stats": [dict(r) for r in rows]}
    finally:
        await conn.close()


@router.get("/query/svo", summary="SVOs für ein Subjekt")
async def kg_query_svo(
    subject: str = Query(..., description="Subjekt / Akteur"),
):
    kg = KnowledgeGraphService()
    try:
        return {"results": await kg.query_svo_by_subject(subject)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/query/entities", summary="NER-Entitäten nach Label")
async def kg_query_entities(
    label: str = Query(..., description="GESETZ | BEHÖRDE | ROLLE | ORT"),
):
    kg = KnowledgeGraphService()
    try:
        return {"results": await kg.query_entities_by_label(label)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/query/impact", summary="Impact-Analyse für ein Dokument")
async def kg_impact(
    doc_id: str = Query(..., description="doc_id aus norm_documents"),
):
    kg = KnowledgeGraphService()
    try:
        return {"results": await kg.query_impact_analysis(doc_id)}
    except Exception as e:
        raise HTTPException(500, str(e))
