# services/ingest/app/api/routes/reports_router.py
#
# Report-Endpoints – direkte DB-Abfragen, strukturiertes JSON.
# Das Frontend rendert die Daten als Tabellen.

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Query
import asyncpg
from app.core.config import settings

router = APIRouter()


async def _conn():
    return await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )


# ── Fragen-Report ─────────────────────────────────────────────────────────────

@router.get("/question")
async def report_question(
    doc:       Optional[str] = Query(None),
    show_all:  bool          = Query(False),
):
    conn = await _conn()
    try:
        where_parts = []
        params = []
        if not show_all:
            where_parts.append("reviewed = FALSE")
        if doc:
            params.append(f"%{doc}%")
            where_parts.append(f"doc_title ILIKE ${len(params)}")
        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        rows = await conn.fetch(f"""
            SELECT id::text, doc_title, doc_class, question_type,
                   question_text, context_before, context_after,
                   section_path, chunk_position,
                   reviewed, decision, reviewer_note,
                   created_at
            FROM filtered_questions
            {where}
            ORDER BY doc_title, chunk_position
            LIMIT 200
        """, *params)

        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) AS gesamt,
                SUM(CASE WHEN reviewed=FALSE THEN 1 END) AS offen,
                SUM(CASE WHEN decision='behalten'  THEN 1 END) AS behalten,
                SUM(CASE WHEN decision='verwerfen' THEN 1 END) AS verworfen,
                SUM(CASE WHEN decision='unklar'    THEN 1 END) AS unklar
            FROM filtered_questions
        """)

        typen = await conn.fetch("""
            SELECT question_type, COUNT(*) AS cnt
            FROM filtered_questions
            GROUP BY question_type ORDER BY cnt DESC
        """)

        return {
            "stats": {
                "gesamt":    stats["gesamt"] or 0,
                "offen":     stats["offen"]  or 0,
                "behalten":  stats["behalten"]  or 0,
                "verworfen": stats["verworfen"] or 0,
                "unklar":    stats["unklar"]    or 0,
            },
            "typen": [{"typ": r["question_type"], "anzahl": r["cnt"]} for r in typen],
            "fragen": [
                {
                    "id":            r["id"],
                    "dokument":      r["doc_title"],
                    "klasse":        r["doc_class"],
                    "typ":           r["question_type"],
                    "frage":         r["question_text"],
                    "kontext_vor":   (r["context_before"] or "")[-80:],
                    "kontext_nach":  (r["context_after"]  or "")[:80],
                    "abschnitt":     r["section_path"] or "",
                    "reviewed":      r["reviewed"],
                    "entscheidung":  r["decision"] or "",
                    "notiz":         r["reviewer_note"] or "",
                }
                for r in rows
            ],
        }
    finally:
        await conn.close()


# ── Ingest-Report ─────────────────────────────────────────────────────────────

@router.get("/ingest")
async def report_ingest(doc: Optional[str] = Query(None)):
    conn = await _conn()
    try:
        where = "WHERE nd.title ILIKE $1" if doc else ""
        params = [f"%{doc}%"] if doc else []

        jobs = await conn.fetch(f"""
            SELECT ij.filename, ij.status, ij.doc_class,
                   ij.chunk_count, ij.error_message,
                   ij.started_at, ij.finished_at,
                   EXTRACT(EPOCH FROM (ij.finished_at - ij.started_at))::INT AS dauer
            FROM ingest_jobs ij
            {('JOIN norm_documents nd ON ij.doc_id = nd.doc_id ' + where) if doc else ''}
            ORDER BY ij.started_at DESC LIMIT 50
        """, *params)

        docs = await conn.fetch(f"""
            SELECT nd.title, nd.source_type,
                   COUNT(nc.id)                                              AS chunks,
                   SUM(CASE WHEN nc.embedding IS NOT NULL THEN 1 ELSE 0 END) AS mit_emb,
                   -- Anzahl aufgelöster Abkürzungen je Dokument
                   COALESCE((
                       SELECT SUM(sub.laenge) FROM (
                           SELECT min(jsonb_array_length(nc2.abbrev_map)) AS laenge
                           FROM norm_chunks nc2, norm_documents nd2
                           WHERE nc2.doc_id = nd2.id
                             AND nd2.id = nd.id
                             AND nc2.abbrev_map IS NOT NULL
                           GROUP BY nd2.title, nc2.abbrev_map
                       ) sub
                   ), 0)                                                      AS abbrev_anzahl,
                   MAX(nc.doc_class)                                          AS klasse,
                   SUM(nc.token_count)                                        AS tokens
            FROM norm_documents nd
            LEFT JOIN norm_chunks nc ON nc.doc_id = nd.id
            {where}
            GROUP BY nd.title, nd.source_type, nd.id
            ORDER BY nd.title
        """, *params)

        chunk_typen = await conn.fetch(f"""
            SELECT nc.doc_class, nc.chunk_type,
                   COUNT(*) AS anzahl,
                   AVG(nc.token_count)::INT AS avg_tokens
            FROM norm_chunks nc
            JOIN norm_documents nd ON nc.doc_id = nd.id
            {where}
            GROUP BY nc.doc_class, nc.chunk_type
            ORDER BY nc.doc_class, anzahl DESC
        """, *params)

        abbrevs = await conn.fetch(f"""
            SELECT elem->>'abbrev' AS abkuerzung,
                   elem->>'label'  AS label,
                   COUNT(*)        AS in_chunks
            FROM norm_chunks nc
            JOIN norm_documents nd ON nc.doc_id = nd.id
            {(',' if not doc else ',') }
            jsonb_array_elements(nc.abbrev_map) AS elem
            {('WHERE nd.title ILIKE $1 AND' if doc else 'WHERE')} nc.abbrev_map IS NOT NULL
            GROUP BY abkuerzung, label ORDER BY in_chunks DESC LIMIT 20
        """, *params) if True else []

        return {
            "jobs":        [dict(r) for r in jobs],
            "dokumente":   [dict(r) for r in docs],
            "chunk_typen": [dict(r) for r in chunk_typen],
            "abkuerzungen":[dict(r) for r in abbrevs],
        }
    finally:
        await conn.close()


# ── NLP-Qualitätsbericht ──────────────────────────────────────────────────────

@router.get("/nlp_quality")
async def report_nlp_quality(doc: Optional[str] = Query(None)):
    conn = await _conn()
    try:
        where = "WHERE nd.title ILIKE $1" if doc else ""
        params = [f"%{doc}%"] if doc else []
        join = f"JOIN norm_chunks nc ON s.chunk_id = nc.id JOIN norm_documents nd ON nc.doc_id = nd.id {where}"

        normtypen = await conn.fetch(f"""
            SELECT s.norm_type,
                   COUNT(*) AS anzahl,
                   ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),1) AS prozent,
                   ROUND(AVG(s.confidence)::NUMERIC,3) AS avg_conf
            FROM svo_extractions s {join}
            GROUP BY s.norm_type ORDER BY anzahl DESC
        """, *params)

        svo_stats = await conn.fetchrow(f"""
            SELECT COUNT(*) AS gesamt,
                   SUM(CASE WHEN s.subject IS NOT NULL THEN 1 ELSE 0 END) AS mit_subj,
                   SUM(CASE WHEN s.object  IS NOT NULL THEN 1 ELSE 0 END) AS mit_obj,
                   SUM(CASE WHEN s.subject IS NOT NULL
                            AND s.object IS NOT NULL THEN 1 ELSE 0 END) AS vollstaendig,
                   SUM(CASE WHEN s.subject_type='PRONOMEN' THEN 1 ELSE 0 END) AS pronomen,
                   ROUND(AVG(s.confidence)::NUMERIC,3) AS avg_conf
            FROM svo_extractions s {join}
        """, *params)

        top_subj = await conn.fetch(f"""
            SELECT s.subject, s.subject_type, COUNT(*) AS cnt
            FROM svo_extractions s {join}
            WHERE s.subject IS NOT NULL
            GROUP BY s.subject, s.subject_type
            ORDER BY cnt DESC LIMIT 15
        """, *params)

        top_obj = await conn.fetch(f"""
            SELECT s.object, s.object_type, COUNT(*) AS cnt
            FROM svo_extractions s {join}
            WHERE s.object IS NOT NULL
            GROUP BY s.object, s.object_type
            ORDER BY cnt DESC LIMIT 15
        """, *params)

        ner_stats = await conn.fetch(f"""
            SELECT e.label, e.source,
                   COUNT(*) AS anzahl,
                   ROUND(AVG(e.confidence)::NUMERIC,3) AS avg_conf
            FROM ner_entities e
            JOIN norm_chunks nc ON e.chunk_id = nc.id
            JOIN norm_documents nd ON nc.doc_id = nd.id
            {where}
            GROUP BY e.label, e.source ORDER BY e.label, e.source
        """, *params)

        top_ner = await conn.fetch(f"""
            SELECT e.text, e.label, e.source, COUNT(*) AS cnt,
                   ROUND(AVG(e.confidence)::NUMERIC,3) AS avg_conf
            FROM ner_entities e
            JOIN norm_chunks nc ON e.chunk_id = nc.id
            JOIN norm_documents nd ON nc.doc_id = nd.id
            {where}
            WHERE e.confidence > 0.7
            GROUP BY e.text, e.label, e.source
            ORDER BY cnt DESC LIMIT 15
        """, *params)

        g = svo_stats["gesamt"] or 1
        return {
            "normtypen":  [dict(r) for r in normtypen],
            "svo": {
                "gesamt":       svo_stats["gesamt"]       or 0,
                "mit_subj":     svo_stats["mit_subj"]     or 0,
                "mit_obj":      svo_stats["mit_obj"]      or 0,
                "vollstaendig": svo_stats["vollstaendig"] or 0,
                "pronomen":     svo_stats["pronomen"]     or 0,
                "avg_conf":     float(svo_stats["avg_conf"] or 0),
                "subj_pct":     round((svo_stats["mit_subj"] or 0)/g*100,1),
                "obj_pct":      round((svo_stats["mit_obj"]  or 0)/g*100,1),
                "full_pct":     round((svo_stats["vollstaendig"] or 0)/g*100,1),
            },
            "top_subjekte": [dict(r) for r in top_subj],
            "top_objekte":  [dict(r) for r in top_obj],
            "ner_stats":    [dict(r) for r in ner_stats],
            "top_ner":      [dict(r) for r in top_ner],
        }
    finally:
        await conn.close()


# ── NLP-Monitor ───────────────────────────────────────────────────────────────

@router.get("/nlp_monitor")
async def report_nlp_monitor():
    conn = await _conn()
    try:
        jobs = await conn.fetch("""
            SELECT job_id, doc_id, status,
                   chunks_total, chunks_done, svo_count, ner_count,
                   error_message, started_at, updated_at, finished_at,
                   EXTRACT(EPOCH FROM (
                       COALESCE(finished_at, now()) - started_at
                   ))::INT AS laufzeit_sek
            FROM nlp_jobs
            ORDER BY started_at DESC LIMIT 10
        """)

        letzter = dict(jobs[0]) if jobs else None
        if letzter:
            for k in ["started_at","updated_at","finished_at"]:
                if letzter[k]:
                    letzter[k] = letzter[k].isoformat()

        return {
            "letzter_job": letzter,
            "alle_jobs": [
                {
                    "job_id":      r["job_id"][:8] + "...",
                    "status":      r["status"],
                    "chunks":      f"{r['chunks_done'] or 0}/{r['chunks_total'] or 0}",
                    "svo":         r["svo_count"] or 0,
                    "ner":         r["ner_count"] or 0,
                    "laufzeit":    f"{r['laufzeit_sek'] or 0}s",
                    "gestartet":   r["started_at"].strftime("%d.%m %H:%M") if r["started_at"] else "",
                }
                for r in jobs
            ],
        }
    finally:
        await conn.close()
