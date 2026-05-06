#!/usr/bin/env python3
# services/ingest/test_abbrev.py
#
# Testet den AbbrevNormalizer mit typischen Rechtstext-Beispielen.
# Prüft: Auflösung, Positions-Mapping, Traceability, Schutz von Normreferenzen.
#
# Aufruf: python test_abbrev.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.abbrev_normalizer import AbbrevNormalizer


def sep(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    normalizer = AbbrevNormalizer()

    # ── Test 1: Einfache Auflösung ─────────────────────────────────────────
    sep("Test 1: Einfache Auflösung")
    text1 = "Gemäß NHundG muss jeder Hundehalter die Sachkunde besitzen."
    result1 = normalizer.normalize(text1)

    print(f"\n  Original:  {result1.original_text}")
    print(f"  Aufgelöst: {result1.resolved_text}")
    print(f"  Ersetzungen: {len(result1.replacements)}")
    for r in result1.replacements:
        print(f"    '{r.abbrev}' → '{r.resolved}'  "
              f"orig[{r.orig_start}:{r.orig_end}] "
              f"res[{r.res_start}:{r.res_end}]")

    # ── Test 2: Normreferenzen bleiben unverändert ─────────────────────────
    sep("Test 2: Normreferenzen bleiben unverändert")
    text2 = "Nach § 3 Abs. 1 NHundG muss der Hundehalter gemäß VwVfG vorgehen."
    result2 = normalizer.normalize(text2)

    print(f"\n  Original:  {result2.original_text}")
    print(f"  Aufgelöst: {result2.resolved_text}")
    print(f"\n  Prüfung: '§ 3 Abs. 1' unverändert?", end=" ")
    if "§ 3 Abs. 1" in result2.resolved_text:
        print("✅ JA")
    else:
        print("❌ NEIN – Normreferenz wurde verändert!")
    print(f"  Prüfung: 'NHundG' aufgelöst?", end=" ")
    if "Niedersächsisches Gesetz" in result2.resolved_text:
        print("✅ JA")
    else:
        print("❌ NEIN")
    print(f"  Prüfung: 'VwVfG' aufgelöst?", end=" ")
    if "Verwaltungsverfahrensgesetz" in result2.resolved_text:
        print("✅ JA")
    else:
        print("❌ NEIN")

    # ── Test 3: Traceability – Position zurückrechnen ──────────────────────
    sep("Test 3: Traceability – Originalposition bestimmen")
    text3 = "Das BVerwG hat entschieden dass gemäß NHundG die Pflicht gilt."
    result3 = normalizer.normalize(text3)

    print(f"\n  Original:  {result3.original_text}")
    print(f"  Aufgelöst: {result3.resolved_text}")

    # BVerwG liegt im aufgelösten Text an einer anderen Position
    aufgelöst_lower = result3.resolved_text.lower()
    bverwg_pos = aufgelöst_lower.find("bundesverwaltungsgericht")
    if bverwg_pos >= 0:
        orig_pos = result3.original_position(bverwg_pos)
        print(f"\n  'Bundesverwaltungsgericht' in aufgelöstem Text: pos {bverwg_pos}")
        print(f"  → Originalposition: {orig_pos}")
        print(f"  → Originaltext dort: "
              f"'{result3.original_text[orig_pos:orig_pos+6]}'")

    # ── Test 4: Kontext-abhängige Auflösung (AG) ───────────────────────────
    sep("Test 4: Kontext-abhängige Auflösung")

    text4a = "Das AG München hat mit Urteil vom 1.1.2024 entschieden."
    text4b = "Die Muster AG hat den Antrag gestellt."

    result4a = normalizer.normalize(text4a)
    result4b = normalizer.normalize(text4b)

    print(f"\n  Mit Urteil-Kontext:")
    print(f"    Original:  {text4a}")
    print(f"    Aufgelöst: {result4a.resolved_text}")
    print(f"    AG aufgelöst? {'✅ JA' if 'Amtsgericht' in result4a.resolved_text else '❌ NEIN'}")

    print(f"\n  Ohne Urteil-Kontext (Firma):")
    print(f"    Original:  {text4b}")
    print(f"    Aufgelöst: {result4b.resolved_text}")
    print(f"    AG NICHT aufgelöst? {'✅ KORREKT' if 'Amtsgericht' not in result4b.resolved_text else '⚠️  FÄLSCHLICH aufgelöst'}")

    # ── Test 5: abbrev_map JSON-Ausgabe ────────────────────────────────────
    sep("Test 5: abbrev_map für JSONB-Speicherung")
    import json
    print(f"\n  abbrev_map (Test 2):")
    print(json.dumps(result2.abbrev_map, ensure_ascii=False, indent=4))

    # ── Test 6: get_original_snippet ──────────────────────────────────────
    sep("Test 6: Original-Textausschnitt abrufen")
    text6 = "Die Fachbehörde prüft gemäß NHundG die Sachkunde des Hundehalters."
    result6 = normalizer.normalize(text6)

    print(f"\n  Original:  {result6.original_text}")
    print(f"  Aufgelöst: {result6.resolved_text}")

    # Finde "Niedersächsisches Gesetz" im aufgelösten Text
    start = result6.resolved_text.find("Niedersächsisches Gesetz")
    if start >= 0:
        end   = start + len("Niedersächsisches Gesetz über das Halten von Hunden")
        snip  = result6.get_original_snippet(start, end)
        print(f"\n  Aufgelöster Bereich [{start}:{end}]: "
              f"'Niedersächsisches Gesetz...'")
        print(f"  → Original-Snippet: '{snip}'")

    print(f"\n{'='*65}")
    print("  Test abgeschlossen")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
