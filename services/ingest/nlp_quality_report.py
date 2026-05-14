#!/usr/bin/env python3
# services/ingest/nlp_quality_report.py
#
# Vollständiger Qualitätsbericht für die M2 NLP-Pipeline.
# Ermittelt alle relevanten Kennzahlen und gibt Handlungsempfehlungen.
#
# Aufruf:
#   python nlp_quality_report.py              # alle Dokumente
#   python nlp_quality_report.py --doc "NHundG"  # ein Dokument
#   python nlp_quality_report.py --export     # CSV-Export

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

import asyncpg
import time as _time
from datetime import timezone as _tz, timedelta as _td

def _local(dt):
    """Konvertiert UTC-Timestamp aus DB in lokale Zeit."""
    if dt is None: return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz.utc)
    offset = -_time.timezone if _time.daylight == 0 else -_time.altzone
    return dt + _td(seconds=offset)

sys.path.insert(0, str(Path(__file__).parent))
from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Argumente
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NDI NLP Qualitätsbericht")
    parser.add_argument("--doc",    type=str, default=None,
                        help="Filter nach Dokumenttitel")
    parser.add_argument("--export", action="store_true",
                        help="CSV-Export der Detaildaten")
    parser.add_argument("--out",    type=str, default=None,
                        help="CSV-Ausgabepfad")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Ausgabe-Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def sep(title: str, width: int = 65):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def sub(title: str):
    print(f"\n  {'─'*55}")
    print(f"  {title}")
    print(f"  {'─'*55}")


def kpi(label: str, value, target=None, unit: str = "",
        higher_is_better: bool = True):
    """Zeigt eine KPI mit Zielwert und Bewertung."""
    icon = ""
    if target is not None:
        try:
            v = float(str(value).replace("%", "").replace(",", "."))
            t = float(str(target).replace("%", "").replace(",", "."))
            ok = v >= t if higher_is_better else v <= t
            icon = "  ✅" if ok else "  ⚠️"
        except (ValueError, TypeError):
            icon = ""

    target_str = f"  (Ziel: {target}{unit})" if target is not None else ""
    print(f"  {'─'*2} {label:<40} {value}{unit}{target_str}{icon}")


def bar(value: float, width: int = 25) -> str:
    """Einfacher Fortschrittsbalken 0-100%."""
    filled = int(width * min(value, 100) / 100)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Datenbankabfragen
# ─────────────────────────────────────────────────────────────────────────────

async def get_conn():
    return await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )


async def report_overview(conn, doc_filter):
    """Gesamtüberblick: Chunks, SVOs, NER."""
    sep("1. Überblick")

    where = "WHERE nd.title ILIKE $1" if doc_filter else ""
    params = [f"%{doc_filter}%"] if doc_filter else []

    # Basis-Counts
    chunks = await conn.fetchval(f"""
        SELECT COUNT(*) FROM norm_chunks nc
        JOIN norm_documents nd ON nc.doc_id = nd.id {where}
    """, *params)

    svo = await conn.fetchval("""
        SELECT COUNT(*) FROM svo_extractions
    """ if not doc_filter else """
        SELECT COUNT(*) FROM svo_extractions s
        JOIN norm_chunks nc ON s.chunk_id = nc.id
        JOIN norm_documents nd ON nc.doc_id = nd.id
        WHERE nd.title ILIKE $1
    """, *params)

    ner = await conn.fetchval("""
        SELECT COUNT(*) FROM ner_entities
    """ if not doc_filter else """
        SELECT COUNT(*) FROM ner_entities e
        JOIN norm_chunks nc ON e.chunk_id = nc.id
        JOIN norm_documents nd ON nc.doc_id = nd.id
        WHERE nd.title ILIKE $1
    """, *params)

    nlp_jobs = await conn.fetchrow("""
        SELECT COUNT(*) AS jobs,
               MAX(finished_at) AS last_run,
               SUM(svo_count) AS total_svo,
               SUM(ner_count) AS total_ner
        FROM nlp_jobs WHERE status = 'done'
    """)

    print(f"\n  Chunks gesamt:        {chunks:>8,}")
    print(f"  SVO-Tripel:           {svo:>8,}")
    print(f"  NER-Entitäten:        {ner:>8,}")
    print(f"  NLP-Jobs (done):      {nlp_jobs['jobs']:>8,}")
    if nlp_jobs['last_run']:
        from datetime import timezone as _tz
        last_run = nlp_jobs['last_run']
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=_tz.utc)
        # astimezone() nutzt die Systemzeitzone inkl. aktuellem DST-Status
        local_time = last_run.astimezone()
        print(f"  Letzter NLP-Lauf:     {local_time.strftime('%d.%m.%Y %H:%M')} (Ortszeit)")

    if chunks > 0:
        svo_rate = svo / chunks * 100
        ner_rate = ner / chunks * 100
        print(f"\n  SVOs pro Chunk:       {svo/chunks:.1f}")
        print(f"  NER pro Chunk:        {ner/chunks:.1f}")


async def report_normtypen(conn, doc_filter):
    """SVO Normtyp-Verteilung und UNKNOWN-Rate."""
    sep("2. SVO – Normtyp-Qualität")

    params = [f"%{doc_filter}%"] if doc_filter else []
    join = "JOIN norm_chunks nc ON s.chunk_id = nc.id JOIN norm_documents nd ON nc.doc_id = nd.id WHERE nd.title ILIKE $1" if doc_filter else ""

    rows = await conn.fetch(f"""
        SELECT
            s.norm_type,
            COUNT(*) AS anzahl,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS prozent,
            ROUND(AVG(s.confidence)::NUMERIC, 3) AS avg_conf,
            ROUND(MIN(s.confidence)::NUMERIC, 3) AS min_conf
        FROM svo_extractions s
        {join}
        GROUP BY s.norm_type
        ORDER BY anzahl DESC
    """, *params)

    total = sum(r["anzahl"] for r in rows)

    print(f"\n  {'Normtyp':<14} {'Anzahl':>7} {'Anteil':>7} {'ø Konf':>8}  Verteilung")
    print(f"  {'─'*70}")

    unknown_pct = 0.0
    for r in rows:
        b = bar(float(r["prozent"]), 20)
        icon = ""
        if r["norm_type"] == "UNKNOWN":
            unknown_pct = float(r["prozent"])
            icon = "  ⚠️" if unknown_pct > 20 else "  ✅"
        print(f"  {r['norm_type']:<14} {r['anzahl']:>7,} {r['prozent']:>6.1f}%"
              f" {r['avg_conf']:>8}  {b}{icon}")

    print(f"\n  Gesamt: {total:,} SVOs")
    print()
    kpi("UNKNOWN-Rate", f"{unknown_pct:.1f}", "< 20", "%",
        higher_is_better=False)
    if unknown_pct > 20:
        print("  → Empfehlung: Normtyp-Patterns in nlp_config.yaml erweitern")


async def report_svo_qualitaet(conn, doc_filter):
    """SVO-Vollständigkeit: Subject, Object, Confidence."""
    sep("3. SVO – Vollständigkeit")

    params = [f"%{doc_filter}%"] if doc_filter else []
    join = "JOIN norm_chunks nc ON s.chunk_id = nc.id JOIN norm_documents nd ON nc.doc_id = nd.id WHERE nd.title ILIKE $1" if doc_filter else ""

    stats = await conn.fetchrow(f"""
        SELECT
            COUNT(*) AS gesamt,
            SUM(CASE WHEN s.subject IS NOT NULL THEN 1 ELSE 0 END) AS mit_subj,
            SUM(CASE WHEN s.object  IS NOT NULL THEN 1 ELSE 0 END) AS mit_obj,
            SUM(CASE WHEN s.subject IS NOT NULL
                     AND s.object  IS NOT NULL THEN 1 ELSE 0 END) AS vollstaendig,
            SUM(CASE WHEN s.subject_type = 'PRONOMEN' THEN 1 ELSE 0 END) AS pronomen,
            ROUND(AVG(s.confidence)::NUMERIC, 3) AS avg_conf,
            ROUND(AVG(s.norm_type_confidence)::NUMERIC, 3) AS avg_norm_conf
        FROM svo_extractions s
        {join}
    """, *params)

    g = stats["gesamt"] or 1
    subj_pct  = stats["mit_subj"]    / g * 100
    obj_pct   = stats["mit_obj"]     / g * 100
    full_pct  = stats["vollstaendig"]/ g * 100
    pron_pct  = stats["pronomen"]    / g * 100

    kpi("Mit Subjekt",             f"{subj_pct:.1f}", "> 80", "%")
    kpi("Mit Objekt",              f"{obj_pct:.1f}",  "> 50", "%")  # dt. Rechtstexte: > 50% realistisch
    kpi("Vollständig (S+V+O)",     f"{full_pct:.1f}", "> 35", "%")  # dt. Rechtstexte: > 35% realistisch
    kpi("Pronomen als Subjekt",    f"{pron_pct:.1f}", "< 15", "%",
        higher_is_better=False)
    kpi("Ø SVO-Konfidenz",         stats["avg_conf"],  "> 0.65")
    kpi("Ø Normtyp-Konfidenz",     stats["avg_norm_conf"], "> 0.75")

    # Top-Subjekte
    sub("Top-10 Subjekte (Akteure)")
    top_subj = await conn.fetch(f"""
        SELECT s.subject, s.subject_type, COUNT(*) AS cnt
        FROM svo_extractions s
        {join}
        WHERE s.subject IS NOT NULL
        GROUP BY s.subject, s.subject_type
        ORDER BY cnt DESC LIMIT 10
    """, *params)

    for r in top_subj:
        print(f"  {r['cnt']:>5}×  {r['subject'][:35]:<35} [{r['subject_type']}]")

    # Top-Objekte
    sub("Top-10 Objekte (Datenobjekte/Anforderungen)")
    top_obj = await conn.fetch(f"""
        SELECT s.object, s.object_type, COUNT(*) AS cnt
        FROM svo_extractions s
        {join}
        WHERE s.object IS NOT NULL
        GROUP BY s.object, s.object_type
        ORDER BY cnt DESC LIMIT 10
    """, *params)

    for r in top_obj:
        print(f"  {r['cnt']:>5}×  {r['object'][:35]:<35} [{r['object_type']}]")


async def report_ner_qualitaet(conn, doc_filter):
    """NER-Qualität nach Label und Quelle."""
    sep("4. NER – Qualität")

    params = [f"%{doc_filter}%"] if doc_filter else []
    join = "JOIN norm_chunks nc ON e.chunk_id = nc.id JOIN norm_documents nd ON nc.doc_id = nd.id WHERE nd.title ILIKE $1" if doc_filter else ""

    # Verteilung nach Label + Quelle
    rows = await conn.fetch(f"""
        SELECT
            e.label, e.source,
            COUNT(*) AS anzahl,
            ROUND(AVG(e.confidence)::NUMERIC, 3) AS avg_conf
        FROM ner_entities e
        {join}
        GROUP BY e.label, e.source
        ORDER BY e.label, e.source
    """, *params)

    print(f"\n  {'Label':<14} {'Quelle':<8} {'Anzahl':>7} {'ø Konf':>8}")
    print(f"  {'─'*45}")

    prev_label = None
    for r in rows:
        if r["label"] != prev_label:
            prev_label = r["label"]
            print()
        src_icon = "🤖" if r["source"] == "flair" else "📏"
        print(f"  {r['label']:<14} {src_icon} {r['source']:<6} "
              f"{r['anzahl']:>7,}  {r['avg_conf']:>8}")

    # Flair vs Regelbasiert
    flair_count = sum(r["anzahl"] for r in rows if r["source"] == "flair")
    rule_count  = sum(r["anzahl"] for r in rows if r["source"] == "rule")
    total_ner   = flair_count + rule_count or 1

    print(f"\n  Flair-Anteil:  {flair_count/total_ner*100:.1f}%  ({flair_count:,})")
    print(f"  Regel-Anteil:  {rule_count/total_ner*100:.1f}%  ({rule_count:,})")
    print()
    kpi("Flair-Anteil", f"{flair_count/total_ner*100:.1f}", "> 40", "%",
        higher_is_better=True)

    # Top erkannte Entitäten
    sub("Top-10 erkannte Entitäten")
    top_ent = await conn.fetch(f"""
        SELECT e.text, e.label, e.source,
               COUNT(*) AS cnt,
               ROUND(AVG(e.confidence)::NUMERIC, 3) AS avg_conf
        FROM ner_entities e
        {join}
        WHERE e.confidence > 0.7
        GROUP BY e.text, e.label, e.source
        ORDER BY cnt DESC LIMIT 10
    """, *params)

    for r in top_ent:
        icon = "🤖" if r["source"] == "flair" else "📏"
        print(f"  {r['cnt']:>5}×  {icon} {r['text'][:30]:<30} [{r['label']}]  "
              f"ø{r['avg_conf']}")


async def report_abbrev_qualitaet(conn, doc_filter):
    """Abkürzungs-Auflösungsqualität."""
    sep("5. Abkürzungen & Synonyme")

    params = [f"%{doc_filter}%"] if doc_filter else []
    where_nc = "JOIN norm_documents nd ON nc.doc_id = nd.id WHERE nd.title ILIKE $1" if doc_filter else ""

    # Chunks mit Abkürzungen
    with_abbrev = await conn.fetchval(f"""
        SELECT COUNT(*) FROM norm_chunks nc
        {where_nc}
        {'AND' if doc_filter else 'WHERE'} nc.abbrev_map IS NOT NULL
          AND jsonb_array_length(nc.abbrev_map) > 0
    """, *params)

    total_chunks = await conn.fetchval(f"""
        SELECT COUNT(*) FROM norm_chunks nc
        {where_nc}
    """, *params)

    # Häufigste aufgelöste Abkürzungen
    rows = await conn.fetch(f"""
        SELECT
            elem->>'abbrev'   AS abkuerzung,
            elem->>'resolved' AS aufloesung,
            elem->>'label'    AS label,
            COUNT(*)          AS in_chunks
        FROM norm_chunks nc
        {where_nc},
        jsonb_array_elements(nc.abbrev_map) AS elem
        {'AND' if doc_filter else 'WHERE'} nc.abbrev_map IS NOT NULL
        GROUP BY abkuerzung, aufloesung, label
        ORDER BY in_chunks DESC
        LIMIT 15
    """, *params)

    pct = with_abbrev / total_chunks * 100 if total_chunks else 0
    print(f"\n  Chunks mit aufgelösten Abkürzungen: {with_abbrev:,} "
          f"von {total_chunks:,} ({pct:.1f}%)")

    if rows:
        print(f"\n  {'Abkürzung':<15} {'Label':<12} {'Häufigkeit':>10}  Auflösung")
        print(f"  {'─'*70}")
        for r in rows:
            print(f"  {r['abkuerzung']:<15} {r['label']:<12} {r['in_chunks']:>10}  "
                  f"{r['aufloesung'][:35]}")
    else:
        print("\n  ⚠️  Keine Abkürzungen aufgelöst – abbrev_dict.yaml prüfen")
        print("  → Tipp: nlp_worker.py --run nach Befüllung des Wörterbuchs")


async def report_chunks_ohne_nlp(conn, doc_filter):
    """Chunks die noch keine NLP-Verarbeitung haben."""
    sep("6. NLP-Abdeckung")

    params = [f"%{doc_filter}%"] if doc_filter else []
    join_nd = "JOIN norm_documents nd ON nc.doc_id = nd.id" + (
        " WHERE nd.title ILIKE $1" if doc_filter else "")

    total = await conn.fetchval(f"""
        SELECT COUNT(*) FROM norm_chunks nc {join_nd}
    """, *params)

    mit_svo = await conn.fetchval(f"""
        SELECT COUNT(DISTINCT s.chunk_id)
        FROM svo_extractions s
        JOIN norm_chunks nc ON s.chunk_id = nc.id
        {join_nd}
    """, *params)

    mit_ner = await conn.fetchval(f"""
        SELECT COUNT(DISTINCT e.chunk_id)
        FROM ner_entities e
        JOIN norm_chunks nc ON e.chunk_id = nc.id
        {join_nd}
    """, *params)

    svo_pct = mit_svo / total * 100 if total else 0
    ner_pct = mit_ner / total * 100 if total else 0

    print()
    kpi("Chunks mit SVO-Extraktion",  f"{svo_pct:.1f}", "> 60", "%")
    kpi("Chunks mit NER-Extraktion",  f"{ner_pct:.1f}", "> 70", "%")

    print(f"\n  Chunks ohne SVO: {total - mit_svo:,}")
    print(f"  Chunks ohne NER: {total - mit_ner:,}")

    if svo_pct < 60:
        print("\n  → Empfehlung: nlp_worker.py --run erneut ausführen")
    if ner_pct < 70:
        print("  → Empfehlung: NER-Confidence-Threshold in nlp_config.yaml prüfen")


async def report_handlungsempfehlungen(conn, doc_filter):
    """Automatische Handlungsempfehlungen basierend auf den Kennzahlen."""
    sep("7. Handlungsempfehlungen")

    params = [f"%{doc_filter}%"] if doc_filter else []
    join = "JOIN norm_chunks nc ON s.chunk_id = nc.id JOIN norm_documents nd ON nc.doc_id = nd.id WHERE nd.title ILIKE $1" if doc_filter else ""

    issues = []

    # UNKNOWN-Rate
    unknown = await conn.fetchval(f"""
        SELECT ROUND(
            SUM(CASE WHEN s.norm_type = 'UNKNOWN' THEN 1 ELSE 0 END) * 100.0
            / NULLIF(COUNT(*), 0), 1
        )
        FROM svo_extractions s {join}
    """, *params)

    if unknown and float(unknown) > 20:
        issues.append((
            "⚠️  UNKNOWN-Rate hoch",
            f"{unknown}% der SVOs haben keinen Normtyp",
            "nlp_config.yaml → normtypen: Regex-Patterns für Pflichten/Ermessen ergänzen",
        ))

    # Fehlende Objekte
    no_obj = await conn.fetchval(f"""
        SELECT ROUND(
            SUM(CASE WHEN s.object IS NULL THEN 1 ELSE 0 END) * 100.0
            / NULLIF(COUNT(*), 0), 1
        )
        FROM svo_extractions s {join}
    """, *params)

    if no_obj and float(no_obj) > 50:
        issues.append((
            "⚠️  Viele SVOs ohne Objekt",
            f"{no_obj}% der SVOs haben kein Objekt",
            "Bei deutschen Rechtstexten ist > 50% Objekt-Abdeckung realistisch.\n"
            "     Viele Normsätze sind intransitiv (Passiv, Modalverb+Infinitiv).\n"
            "     svo_extractor.py erweitern für tiefere Infinitiv-Cluster-Suche.",
        ))

    # Abkürzungen
    abbrev_count = await conn.fetchval("""
        SELECT COUNT(*) FROM norm_chunks
        WHERE abbrev_map IS NOT NULL
          AND jsonb_array_length(abbrev_map) > 0
    """)

    if not abbrev_count or abbrev_count == 0:
        issues.append((
            "⚠️  Keine Abkürzungen aufgelöst",
            "abbrev_dict.yaml enthält keine passenden Einträge für diese Dokumente",
            "abbrev_dict.yaml befüllen und Ingest wiederholen",
        ))

    if not issues:
        print("\n  ✅ Keine kritischen Probleme gefunden.")
        print("  Das NLP-System arbeitet innerhalb der Zielwerte.")
    else:
        for icon_title, desc, action in issues:
            print(f"\n  {icon_title}")
            print(f"     Problem:   {desc}")
            print(f"     Maßnahme:  {action}")


async def export_csv(conn, doc_filter, out_path):
    """CSV-Export der SVO-Daten für externe Analyse."""
    params = [f"%{doc_filter}%"] if doc_filter else []
    join = """JOIN norm_chunks nc ON s.chunk_id = nc.id
              JOIN norm_documents nd ON nc.doc_id = nd.id
              WHERE nd.title ILIKE $1""" if doc_filter else ""

    rows = await conn.fetch(f"""
        SELECT
            nd.title AS dokument,
            nc.doc_class AS klasse,
            nc.norm_reference,
            s.norm_type,
            s.subject, s.subject_type,
            s.predicate, s.predicate_lemma,
            s.object, s.object_type,
            s.context,
            s.confidence,
            s.norm_type_confidence,
            s.sentence_text
        FROM svo_extractions s
        JOIN norm_chunks nc ON s.chunk_id = nc.id
        JOIN norm_documents nd ON nc.doc_id = nd.id
        {join}
        ORDER BY nd.title, nc.norm_reference, s.norm_type
    """, *params)

    out = out_path or f"nlp_quality_{datetime.now().strftime('%m%d%H%M%S')}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, delimiter=";", fieldnames=[
            "dokument", "klasse", "norm_reference", "norm_type",
            "subject", "subject_type", "predicate", "predicate_lemma",
            "object", "object_type", "context",
            "confidence", "norm_type_confidence", "sentence_text",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))

    print(f"\n  ✅ {len(rows)} SVOs exportiert → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Hauptprogramm
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    print(f"\n{'='*65}")
    print(f"  NDI / MNR – NLP Qualitätsbericht")
    print(f"  {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    if args.doc:
        print(f"  Filter: '{args.doc}'")
    print(f"{'='*65}")

    conn = await get_conn()
    try:
        await report_overview(conn,               args.doc)
        await report_normtypen(conn,              args.doc)
        await report_svo_qualitaet(conn,          args.doc)
        await report_ner_qualitaet(conn,          args.doc)
        await report_abbrev_qualitaet(conn,       args.doc)
        await report_chunks_ohne_nlp(conn,        args.doc)
        await report_handlungsempfehlungen(conn,  args.doc)

        if args.export:
            sep("CSV-Export")
            await export_csv(conn, args.doc, args.out)

    finally:
        await conn.close()

    print(f"\n{'='*65}")
    print(f"  Bericht abgeschlossen")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
