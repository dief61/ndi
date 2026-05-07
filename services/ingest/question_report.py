#!/usr/bin/env python3
# services/ingest/question_report.py
#
# Review-Script für gefilterte Fragen nach dem Ingest.
#
# Aufruf:
#   python question_report.py                    # alle unreviewed Fragen
#   python question_report.py --doc "FAQ1"       # nur ein Dokument
#   python question_report.py --all              # inkl. bereits reviewed
#   python question_report.py --export           # CSV-Export
#   python question_report.py --decide <id> behalten|verwerfen|unklar

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent))
from app.core.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="NDI Fragen-Review")
    parser.add_argument("--doc",    type=str, default=None,
                        help="Filter nach Dokumenttitel")
    parser.add_argument("--all",    action="store_true",
                        help="Auch bereits reviewte Fragen anzeigen")
    parser.add_argument("--export", action="store_true",
                        help="CSV-Export")
    parser.add_argument("--out",    type=str, default=None,
                        help="CSV-Ausgabepfad")
    parser.add_argument("--decide", nargs=2, metavar=("ID", "ENTSCHEIDUNG"),
                        help="Entscheidung setzen: --decide <uuid> behalten|verwerfen|unklar")
    parser.add_argument("--note",   type=str, default=None,
                        help="Notiz zur Entscheidung (mit --decide)")
    return parser.parse_args()


def sep(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


async def main():
    args = parse_args()

    conn = await asyncpg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db,
    )

    # ── Entscheidung setzen ───────────────────────────────────────────────
    if args.decide:
        fid, decision = args.decide
        if decision not in ("behalten", "verwerfen", "unklar"):
            print(f"\n  Ungültige Entscheidung: '{decision}'")
            print(f"  Erlaubt: behalten | verwerfen | unklar")
            await conn.close()
            return

        result = await conn.execute(
            """
            UPDATE filtered_questions
            SET reviewed=TRUE, decision=$1, reviewer_note=$2
            WHERE id::text LIKE $3
            """,
            decision, args.note, f"{fid}%",
        )
        print(f"\n  ✅ Entscheidung gesetzt: {decision}")
        if args.note:
            print(f"  Notiz: {args.note}")
        await conn.close()
        return

    # ── Statistik ─────────────────────────────────────────────────────────
    sep("Fragen-Statistik")

    stats = await conn.fetchrow("""
        SELECT
            COUNT(*)                                     AS gesamt,
            SUM(CASE WHEN reviewed = FALSE THEN 1 END)  AS offen,
            SUM(CASE WHEN decision = 'behalten'  THEN 1 END) AS behalten,
            SUM(CASE WHEN decision = 'verwerfen' THEN 1 END) AS verworfen,
            SUM(CASE WHEN decision = 'unklar'    THEN 1 END) AS unklar
        FROM filtered_questions
    """)

    print(f"\n  Gefilterte Fragen gesamt:  {stats['gesamt']}")
    print(f"  Offen (nicht reviewed):    {stats['offen']}")
    print(f"  Entschieden – behalten:    {stats['behalten'] or 0}")
    print(f"  Entschieden – verworfen:   {stats['verworfen'] or 0}")
    print(f"  Entschieden – unklar:      {stats['unklar'] or 0}")

    # Typ-Verteilung
    typen = await conn.fetch("""
        SELECT question_type, COUNT(*) AS cnt
        FROM filtered_questions
        GROUP BY question_type ORDER BY cnt DESC
    """)
    print(f"\n  Fragen-Typen:")
    for t in typen:
        print(f"    {t['question_type']:<15} {t['cnt']}")

    # Pro Dokument
    per_doc = await conn.fetch("""
        SELECT doc_title, doc_class, COUNT(*) AS cnt,
               SUM(CASE WHEN reviewed=FALSE THEN 1 END) AS offen
        FROM filtered_questions
        GROUP BY doc_title, doc_class
        ORDER BY cnt DESC
    """)
    print(f"\n  Pro Dokument:")
    for r in per_doc:
        print(f"    {r['doc_title'][:35]:<35} "
              f"[{r['doc_class']}]  {r['cnt']} Fragen  "
              f"({r['offen']} offen)")

    # ── Fragen anzeigen ───────────────────────────────────────────────────
    where_parts = []
    params = []

    if not args.all:
        where_parts.append("reviewed = FALSE")
    if args.doc:
        params.append(f"%{args.doc}%")
        where_parts.append(f"doc_title ILIKE ${len(params)}")

    where = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    rows = await conn.fetch(f"""
        SELECT id, doc_title, doc_class, question_type,
               question_text, context_before, context_after,
               section_path, chunk_position,
               reviewed, decision, reviewer_note, created_at
        FROM filtered_questions
        {where}
        ORDER BY created_at DESC, chunk_position
        LIMIT 100
    """, *params)

    if not rows:
        print(f"\n  Keine {'offenen ' if not args.all else ''}Fragen gefunden.")
    else:
        label = "Alle Fragen" if args.all else "Offene Fragen (noch nicht reviewed)"
        sep(label)

        TYPE_ICON = {
            "direkt":      "❓",
            "rhetorisch":  "💭",
            "liste":       "📋",
            "unbekannt":   "❔",
        }
        DECISION_ICON = {
            "behalten":  "✅",
            "verwerfen": "❌",
            "unklar":    "⚠️",
            None:        "⏳",
        }

        for r in rows:
            icon     = TYPE_ICON.get(r["question_type"], "❔")
            dec_icon = DECISION_ICON.get(r["decision"])
            id_short = str(r["id"])[:8]

            print(f"\n  {icon} [{id_short}]  {dec_icon}  "
                  f"{r['doc_title'][:30]}  "
                  f"[Klasse {r['doc_class']}]  "
                  f"Typ: {r['question_type']}")
            print(f"     Frage:    {r['question_text'][:100]}")
            if r["context_before"]:
                print(f"     Vorher:   ...{r['context_before'][-60:]}")
            if r["context_after"]:
                print(f"     Nachher:  {r['context_after'][:60]}...")
            if r["section_path"]:
                print(f"     Abschnitt: {r['section_path']}")
            if r["reviewer_note"]:
                print(f"     Notiz:    {r['reviewer_note']}")

        print(f"\n  {len(rows)} Fragen angezeigt.")

        if not args.all and stats["offen"] and stats["offen"] > 0:
            print(f"\n  Tipp – Entscheidung setzen:")
            print(f"  python question_report.py --decide <ID> behalten")
            print(f"  python question_report.py --decide <ID> verwerfen --note 'FAQ-Text'")

    # ── CSV-Export ────────────────────────────────────────────────────────
    if args.export:
        all_rows = await conn.fetch("""
            SELECT id, doc_title, doc_class, question_type,
                   question_text, context_before, context_after,
                   section_path, chunk_position,
                   reviewed, decision, reviewer_note, created_at
            FROM filtered_questions
            ORDER BY doc_title, chunk_position
        """)

        out_path = args.out or f"questions_{datetime.now().strftime('%m%d%H%M%S')}.csv"
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, delimiter=";", fieldnames=[
                "id", "dokument", "klasse", "typ", "frage",
                "kontext_vorher", "kontext_nachher", "abschnitt",
                "position", "reviewed", "entscheidung", "notiz", "datum",
            ])
            writer.writeheader()
            for r in all_rows:
                writer.writerow({
                    "id":              str(r["id"]),
                    "dokument":        r["doc_title"],
                    "klasse":          r["doc_class"],
                    "typ":             r["question_type"],
                    "frage":           r["question_text"],
                    "kontext_vorher":  r["context_before"] or "",
                    "kontext_nachher": r["context_after"]  or "",
                    "abschnitt":       r["section_path"]   or "",
                    "position":        r["chunk_position"],
                    "reviewed":        r["reviewed"],
                    "entscheidung":    r["decision"]        or "",
                    "notiz":           r["reviewer_note"]   or "",
                    "datum":           r["created_at"].strftime("%d.%m.%Y %H:%M")
                                       if r["created_at"] else "",
                })
        print(f"\n  ✅ {len(all_rows)} Fragen exportiert → {out_path}")

    await conn.close()
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
