# M3 – Query-Konzept: Schritt 1.1 Query-Analyse & -Transformation

> Technisches Konzept für die Query-Analyse und -Transformation
> im Hybrid-Retrieval-System des Meta-Normen-Registers (MNR).
>
> Stand: Mai 2026 | Meilenstein M3 – Schritt 1
>
> Dieses Dokument beschreibt Aufbau, Datenstrukturen und Implementierungsstrategie
> des `QueryTransformer` – der ersten Stufe der RAG-Pipeline.

---

## Inhaltsverzeichnis

1. [Kontext und Einordnung](#1-kontext-und-einordnung)
2. [Gesamtübersicht Schritt 1](#2-gesamtübersicht-schritt-1)
3. [Aufgabe 1 – Query-Typ klassifizieren](#3-aufgabe-1--query-typ-klassifizieren)
4. [Aufgabe 2 – Norm-Referenz extrahieren](#4-aufgabe-2--norm-referenz-extrahieren)
5. [Aufgabe 3 – Query-Transformation](#5-aufgabe-3--query-transformation)
6. [Aufgabe 4 – Metadaten-Filter ableiten](#6-aufgabe-4--metadaten-filter-ableiten)
7. [Output: QueryBundle](#7-output-querybundle)
8. [Prompt-Suite: neue Prompts](#8-prompt-suite-neue-prompts)
9. [Konfiguration: rag_config.yaml](#9-konfiguration-rag_configyaml)
10. [Dateien und Implementierungsreihenfolge](#10-dateien-und-implementierungsreihenfolge)
11. [Abhängigkeiten aus M2](#11-abhängigkeiten-aus-m2)
12. [Qualitätsmetriken](#12-qualitätsmetriken)

---

## 1. Kontext und Einordnung

### Einbettung in M3

M3 implementiert die RAG-Engine (Retrieval-Augmented Generation). Sie ist die
zentrale Komponente zwischen der NLP-Pipeline (M2) und dem
Informationsmodell-Generator (M4).

```
M2 – NLP                    M3 – RAG-Engine               M4 – Informationsmodell
─────────────────────────   ──────────────────────────   ──────────────────────────
SVO-Extraktion          →   1.1 Query-Transformation  →   IM-Generator
NER (zweistufig)        →   1.2 Klassen-Retriever     →   Human-in-the-Middle
LLM-Normtyp             →   1.3 Parent-Child          →   Traceability-Matrix
Knowledge Graph (Fuseki)→   1.4 Cross-Reference       →
im_signals (neu)        →   1.5 Re-Ranker             →
                            1.6 Context Assembly      →
```

### Besonderheit: IM-Signal-Awareness

Durch die in M2 eingeführten Strukturen – `im_signals`-Spalte in
`norm_chunks`, `agnostische_datenobjekte` und `ner_extensions` in
`docs.yaml` – ist der Query-Transformer erstmals in der Lage,
Anfragen bezüglich des Informationsmodells (Entitäten, Attribute,
Persistenz) gezielt zu routen.

---

## 2. Gesamtübersicht Schritt 1

```
Query (Rohtext)
  │
  ▼
1.1 Query-Analyse & -Transformation          ← dieses Dokument
    │
    ├─ Typ erkennen (NORM / ENTITY / IM / GENERAL)
    ├─ Norm-Referenz extrahieren  → Direkt-Lookup
    ├─ Transformationen: Direct / HyDE / Step-Back / Multi-Query
    └─ Metadaten-Filter ableiten
    │
    ▼  QueryBundle
1.2 Klassen-sensitiver Retriever
    Klasse A: Direkt-Lookup + Vektor-Cosine + PostgreSQL FTS → RRF
    Klasse B: Requirement-ID-Lookup + Vektor + FTS → RRF
    Klasse C: nur Vektor (Score-Threshold ≥ 0.72)
    │
    ▼
1.3 Parent-Child-Expansion
    A: §-Parent immer nachladen
    B: Kapitel-Parent selektiv + Tabellen-Chunks
    C: kein Parent-Fetch
    │
    ▼
1.4 Cross-Reference-Expansion (nur Klasse A)
    "§ X gilt entsprechend" → §-X-Inhalt rekursiv nachladen (max. Tiefe 2)
    │
    ▼
1.5 Re-Ranker
    Cross-Encoder (ms-marco-MiniLM) × Klassen-Gewichtung
    (A: ×1.0 | B: ×0.85 | C: ×0.65) + Duplikat-Filter (content_hash)
    │
    ▼
1.6 Context Assembly
    Präambeln + IM-Signal-Annotierungen (Entity/Attribut/Persistenz-Kandidaten)
```

---

## 3. Aufgabe 1 – Query-Typ klassifizieren

### Vier Query-Typen

Der Query-Typ bestimmt welche Retrieval-Strategie und welche Datenbankfilter
verwendet werden.

| Query-Typ | Erkennungsmerkmal | Primäre Retrieval-Strategie |
|---|---|---|
| `NORM` | §, Absatz, Satz, Modalverb (muss, darf, soll) | Vektor + FTS auf Klasse A |
| `ENTITY` | fragt nach Akteur, Rolle, Behörde, Zuständigkeit | NER-Label-Filter + Klasse A+B |
| `IM` | fragt nach Tabelle, Attribut, Datenbankfeld, Entität | `im_signals`-Filter zuerst |
| `GENERAL` | freie Frage, keine obigen Signale | alle Klassen, Vektor-only |

### Erkennungslogik (Kaskade)

```python
def classify_query_typ(query: str) -> str:
    q = query.lower()

    # NORM: gesetzliche Signale
    if re.search(r'\b§\s*\d+|abs\.|absatz|satz\s+\d+', q):
        return "NORM"
    if re.search(r'\bmuss\b|\bdarf\b|\bsoll\b|\bbedarf\b|\bist\s+zu\b', q):
        return "NORM"

    # IM: Informationsmodell-Signale
    if re.search(r'\bentit[aä]t\b|\btabelle\b|\battribut\b|\bspalte\b|'
                 r'\bdatenbank\b|\bfeld\b|\bpersistier', q):
        return "IM"

    # ENTITY: Akteur/Rolle-Signale
    if re.search(r'\bzust[äa]ndig\b|\bwer\s+ist\b|\bwelche\s+beh[öo]rde\b|'
                 r'\bwelche\s+rolle\b', q):
        return "ENTITY"

    return "GENERAL"
```

### Beispiele

```
"Was muss ein Hundehalter nachweisen?"
  → Signale: "muss", "nachweisen"
  → Typ: NORM
  → Strategie: Vektor + FTS auf Klasse A, source_type: gesetz

"Welche Entitäten hat das Hunderegister?"
  → Signale: "Entitäten", "Hunderegister"
  → Typ: IM
  → Strategie: im_signals-Filter, entity_kandidaten enthält "Hunderegister"

"Wer ist zuständig für den Wesenstest?"
  → Signale: "zuständig"
  → Typ: ENTITY
  → Strategie: NER-Label BEHÖRDE + norm_type COMPETENCE

"Erkläre den Sachkundenachweis."
  → keine obigen Signale
  → Typ: GENERAL
  → Strategie: alle Klassen, reine Vektorsuche
```

---

## 4. Aufgabe 2 – Norm-Referenz extrahieren

### Motivation

Wenn eine kanonische Norm-Referenz im Query erkannt wird, ist ein
direkter Datenbankzugriff auf `norm_chunks.norm_reference` möglich.
Der Vektorvergleich wird übersprungen – schneller und präziser.

### Regex-Pattern

```python
# Vollständige Norm-Referenz: "§ 3 Abs. 1 Satz 2 NHundG"
NORM_REF_PATTERN = re.compile(
    r'§\s*(?P<para>\d+[a-z]?)'           # § 3 oder § 3a
    r'(?:\s+Abs\.\s*(?P<abs>\d+))?'       # Abs. 1  (optional)
    r'(?:\s+Satz\s*(?P<satz>\d+))?'       # Satz 2  (optional)
    r'(?:\s+Nr\.\s*(?P<nr>\d+))?'         # Nr. 1   (optional)
    r'(?:\s+(?P<gesetz>[A-Z][a-zA-ZÄÖÜäöü]+G\b))?'  # NHundG  (optional)
)
```

### Beispiele

```
"Was steht in § 3 Abs. 1 NHundG?"
  → norm_reference = "§ 3 Abs. 1 NHundG"
  → Direkt-Lookup: SELECT * FROM norm_chunks
                   WHERE norm_reference LIKE '§ 3 Abs. 1%'
                   AND metadata->>'gesetz' = 'NHundG'
  → kein Embedding-Vergleich

"§ 9 Satz 1"
  → norm_reference = "§ 9 Satz 1"
  → Direkt-Lookup mit Fuzzy-Fallback auf Vektor wenn nicht gefunden

"Darf ein Hundehalter einen gefährlichen Hund halten?"
  → keine Norm-Referenz erkannt
  → weiter zu Aufgabe 3 (Transformation)
```

### Fallback-Strategie

```
Direkt-Lookup erfolgreich  → Ergebnis direkt in QueryBundle
Direkt-Lookup leer         → Vektor-Suche mit norm_reference als Query-Text
Norm-Referenz nicht erkannt → normale Transformation (Aufgabe 3)
```

---

## 5. Aufgabe 3 – Query-Transformation

Vier Strategien werden je nach Query-Typ selektiv aktiviert und
parallel ausgeführt. Die Ergebnisse fließen als gewichtete
Vektoren in das QueryBundle.

### Strategie-Auswahl nach Query-Typ

| Query-Typ | Direct | HyDE | Step-Back | Multi-Query |
|---|---|---|---|---|
| NORM | ✅ | ✅ | ✅ | – |
| ENTITY | ✅ | – | – | ✅ |
| IM | ✅ | ✅ | – | – |
| GENERAL | ✅ | – | – | ✅ |

---

### 3a – Direct (immer aktiv)

Der Original-Query wird unverändert eingebettet.

```
Query:  "Was muss ein Hundehalter nachweisen?"
Vektor: embed("Was muss ein Hundehalter nachweisen?")
Gewicht: 1.0
```

---

### 3b – HyDE (Hypothetical Document Embedding)

**Problem:** Ein kurzer Query-Text erzeugt einen Vektor, der stilistisch weit
von Normtexten entfernt ist. Ein hypothetischer Normtext trifft echte Chunks
deutlich besser.

**Lösung:** Das LLM generiert einen plausiblen deutschen Normtext als
hypothetische Antwort. Dessen Embedding-Vektor wird als Suchvektor verwendet.

```
Query:    "Was muss ein Hundehalter nachweisen?"

Prompt (ps_hyde):
  System: "Du bist ein Experte für deutsches Verwaltungsrecht.
           Schreibe einen kurzen deutschen Normtext (1-3 Sätze)
           der die folgende Frage beantwortet.
           Verwende typische Gesetzessprache."
  User:   "Frage: {{query}}"

LLM-Antwort:
  "Wer einen Hund hält, hat der zuständigen Gemeinde auf Verlangen
   die erforderliche Sachkunde durch Vorlage eines Nachweises
   nach § 3 Abs. 1 NHundG zu belegen."

Vektor: embed(LLM-Antwort)
Gewicht: 0.8
```

**Wann deaktivieren:** Bei kurzen, eindeutigen Norm-Referenz-Queries
(`§ 3 Abs. 1 NHundG`) ist HyDE nicht sinnvoll – der Direkt-Lookup
ist präziser.

---

### 3c – Step-Back

**Problem:** Sehr spezifische Queries (einzelner § mit Absatz/Satz) finden
möglicherweise den exakten Chunk, aber keinen übergeordneten Kontext.

**Lösung:** Das LLM abstrahiert die Query auf die übergeordnete
Normenebene. Der Step-Back-Vektor findet thematisch verwandte
Paragraphen im gleichen Regelungskontext.

```
Query:     "§ 9 Satz 1 NHundG Frist Erlaubnisantrag"

Prompt (ps_step_back):
  System: "Du bist ein Experte für deutsches Verwaltungsrecht.
           Formuliere eine allgemeinere Frage, die den übergeordneten
           Regelungskontext der folgenden spezifischen Anfrage erfasst."
  User:   "Spezifische Anfrage: {{query}}"

LLM-Antwort:
  "Welche Fristen und Pflichten gelten für Halter gefährlicher Hunde
   nach dem NHundG?"

Vektor: embed(LLM-Antwort)
Gewicht: 0.6
```

**Sinnvoll bei:**
- Einzelnen §-Abschnitt-Anfragen
- Anfragen die Querverweise enthalten ("gilt entsprechend")

---

### 3d – Multi-Query

**Problem:** Vage Queries decken nur einen Aspekt des relevanten Normbereichs ab.

**Lösung:** Das LLM generiert n Varianten des Original-Queries.
Jede Variante fokussiert auf einen anderen Aspekt.

```
Query:     "Hund gefährlich Pflichten"

Prompt (ps_multi_query):
  System: "Du bist ein Experte für deutsches Verwaltungsrecht.
           Formuliere {{count}} unterschiedliche Suchanfragen,
           die verschiedene Aspekte der folgenden Frage abdecken.
           Gib ausschließlich ein JSON-Array zurück: [\"...\", \"...\"]"
  User:   "Frage: {{query}}"

LLM-Antwort:
  [
    "Pflichten des Halters eines gefährlichen Hundes NHundG",
    "Erlaubnispflicht Haltung gefährlicher Hund Fachbehörde",
    "Feststellung Gefährlichkeit Rechtsfolgen Wesenstest"
  ]

Vektoren: [embed(V1), embed(V2), embed(V3)]
Gewichte: [0.7, 0.7, 0.7]
```

**Deduplizierung:** RRF (Reciprocal Rank Fusion) im Retriever sorgt
dafür, dass ein Chunk der durch mehrere Varianten gefunden wird,
höher gerankt wird.

---

## 6. Aufgabe 4 – Metadaten-Filter ableiten

Basierend auf Query-Typ, erkannten Entitäten und Norm-Referenz werden
PostgreSQL-Filter für den Retriever abgeleitet.

### Filter je Query-Typ

```python
def build_metadata_filter(
    query_typ:      str,
    norm_reference: str | None,
    im_filter:      bool,
) -> dict:

    if query_typ == "NORM":
        return {
            "doc_class":   ["A"],
            "source_type": ["gesetz", "verordnung", "standard"],
            "valid_to":    None,              # noch gültig
        }

    elif query_typ == "IM":
        return {
            "im_signals_exists": True,        # nur Chunks mit IM-Signalen
            "doc_class":         ["A", "B"],
        }

    elif query_typ == "ENTITY":
        return {
            "doc_class":   ["A", "B"],
            "norm_type":   ["COMPETENCE", "DEF", "MUST"],
        }

    else:  # GENERAL
        return {}     # kein Filter – alle Klassen durchsuchen
```

### IM-Signal-Filter (neu durch M2-Änderungen)

Wenn der Query-Typ `IM` erkannt wird, werden zusätzlich die
`im_signals` aus `norm_chunks` ausgewertet:

```python
# PostgreSQL-Abfrage mit im_signals-Filter
"""
SELECT nc.*, nc.im_signals
FROM norm_chunks nc
WHERE nc.im_signals IS NOT NULL
  AND (
    nc.im_signals->'entity_kandidaten' @> '["Hundehalter"]'
    OR
    nc.im_signals->'attribut_kandidaten' @> '["Sachkunde"]'
  )
ORDER BY ... -- Vektor-Ähnlichkeit
"""
```

### Zeitliche Gültigkeit

Chunks mit `valid_to IS NOT NULL AND valid_to < CURRENT_DATE` werden
automatisch aus dem Retrieval ausgeschlossen – außer die Anfrage
bezieht sich explizit auf historische Regelungen.

---

## 7. Output: QueryBundle

Das QueryBundle ist die standardisierte Ausgabe des QueryTransformers
und die Eingabe für den Retriever (Schritt 1.2).

### Datenstruktur

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class QueryVector:
    text:       str        # einzubettender Text
    strategie:  str        # direct | hyde | step_back | multi_query
    gewicht:    float      # 0.0 – 1.0

@dataclass
class QueryBundle:
    original_query:   str
    query_typ:        str                   # NORM | ENTITY | IM | GENERAL
    norm_reference:   Optional[str]         # "§ 3 Abs. 1 NHundG" oder None
    vektoren:         list[QueryVector]     # ein oder mehrere Such-Vektoren
    metadata_filter:  dict                  # pgvector WHERE-Klauseln
    im_filter:        bool                  # True = im_signals bevorzugen
    direktlookup:     bool                  # True = Vektorsuche überspringen
```

### Beispiele

**Beispiel 1 – NORM mit Direkt-Lookup:**

```python
QueryBundle(
    original_query  = "Was steht in § 3 Abs. 1 NHundG?",
    query_typ       = "NORM",
    norm_reference  = "§ 3 Abs. 1 NHundG",
    vektoren        = [
        QueryVector("§ 3 Abs. 1 NHundG", "direct", 1.0),
    ],
    metadata_filter = {"doc_class": ["A"]},
    im_filter       = False,
    direktlookup    = True,    # Vektorsuche überspringen
)
```

**Beispiel 2 – NORM mit HyDE + Step-Back:**

```python
QueryBundle(
    original_query  = "Was muss ein Hundehalter nachweisen?",
    query_typ       = "NORM",
    norm_reference  = None,
    vektoren        = [
        QueryVector(
            "Was muss ein Hundehalter nachweisen?",
            "direct", 1.0),
        QueryVector(
            "Wer einen Hund haelt, hat der Gemeinde die Sachkunde nachzuweisen.",
            "hyde", 0.8),
        QueryVector(
            "Welche Nachweispflichten bestehen fuer Hundehalter?",
            "step_back", 0.6),
    ],
    metadata_filter = {
        "doc_class":   ["A"],
        "source_type": ["gesetz", "verordnung"],
    },
    im_filter       = False,
    direktlookup    = False,
)
```

**Beispiel 3 – IM-Query:**

```python
QueryBundle(
    original_query  = "Welche Entitaeten hat das Hunderegister?",
    query_typ       = "IM",
    norm_reference  = None,
    vektoren        = [
        QueryVector("Welche Entitaeten hat das Hunderegister?", "direct", 1.0),
        QueryVector(
            "Datenbankschema Hunderegister Entitaeten Attribute",
            "hyde", 0.8),
    ],
    metadata_filter = {"im_signals_exists": True},
    im_filter       = True,
    direktlookup    = False,
)
```

---

## 8. Prompt-Suite: neue Prompts

Drei neue Prompt-Paare werden in der Prompt-Suite angelegt.
Alle nutzen den bestehenden LLM-Gateway mit dem aktiven Provider.

### ps_hyde/system.txt

```
Du bist ein Experte fuer deutsches Verwaltungsrecht und Rechtssprache.

Deine Aufgabe: Schreibe einen kurzen deutschen Normtext (1-3 Saetze),
der die gestellte Frage als Antwort enthaelt.

Regeln:
- Verwende typische deutsche Gesetzessprache
- Nutze Modalverben: muss, hat zu, ist zu, darf, kann, bedarf
- Nenne konkrete Akteure (Hundehalterin oder Hundehalter, Fachbehoerde, ...)
- Nenne konkrete Objekte (Sachkunde, Erlaubnis, Transponder, ...)
- Keine Erlaeuterungen – nur den Normtext
- Kein JSON, keine Markdown-Formatierung
```

### ps_hyde/user.txt

```
Schreibe einen hypothetischen Normtext als Antwort auf diese Frage:

{{query}}
```

### ps_step_back/system.txt

```
Du bist ein Experte fuer deutsches Verwaltungsrecht.

Deine Aufgabe: Formuliere eine allgemeinere Suchanfrage, die den
uebergeordneten Regelungskontext der folgenden spezifischen Anfrage erfasst.

Regeln:
- Abstrahiere auf die Normenebene (z.B. von einem einzelnen § auf die gesamte Regelung)
- Nenne das Rechtsgebiet oder den Regelungsbereich
- Keine Erlaeuterungen – nur die neue Suchanfrage als einfachen Text
```

### ps_step_back/user.txt

```
Spezifische Anfrage: {{query}}

Allgemeinere Suchanfrage:
```

### ps_multi_query/system.txt

```
Du bist ein Experte fuer deutsches Verwaltungsrecht.

Deine Aufgabe: Formuliere {{count}} unterschiedliche Suchanfragen,
die verschiedene Aspekte der gestellten Frage abdecken.

Regeln:
- Jede Variante deckt einen anderen Aspekt ab (Pflichten, Berechtigungen, Zustaendigkeit, ...)
- Verwende juristische Fachbegriffe
- Gib ausschliesslich ein JSON-Array zurueck: ["...", "...", "..."]
- Kein Markdown, keine Erklaerungen
```

### ps_multi_query/user.txt

```
Frage: {{query}}
```

---

## 9. Konfiguration: rag_config.yaml

```yaml
# services/ingest/rag_config.yaml
#
# Konfiguration der M3 RAG-Engine.
# Alle Parameter koennen ohne Neustart geaendert werden.

# ── 1.1 Query-Transformation ──────────────────────────────────────────────────
query_transformation:

  # HyDE – Hypothetical Document Embedding
  hyde:
    enabled:      true
    gewicht:      0.8         # Gewicht des HyDE-Vektors
    prompt_key:   "ps_hyde"
    # Deaktivieren bei Direktlookup-Queries (norm_reference erkannt)
    skip_if_direktlookup: true

  # Step-Back – Query abstrahieren
  step_back:
    enabled:      true
    gewicht:      0.6
    prompt_key:   "ps_step_back"
    # Nur bei NORM-Queries mit Norm-Referenz
    nur_bei_typen: ["NORM"]

  # Multi-Query – Varianten generieren
  multi_query:
    enabled:      false       # teurer – per Bedarf aktivieren
    count:        3           # Anzahl Varianten
    gewicht:      0.7
    prompt_key:   "ps_multi_query"
    nur_bei_typen: ["ENTITY", "GENERAL"]

# ── 1.2 Retrieval ─────────────────────────────────────────────────────────────
retrieval:
  top_k_pro_klasse: 6         # Chunks pro Klasse vor Re-Ranking
  top_k_final:      8         # Chunks im finalen Kontext

  score_thresholds:
    class_a: 0.60
    class_b: 0.60
    class_c: 0.72             # erhoeht wegen niedrigem confidence_weight

  fts_enabled:      true      # PostgreSQL Volltextsuche aktiv
  rrf_k:            60        # Reciprocal Rank Fusion Parameter

# ── 1.5 Re-Ranker ─────────────────────────────────────────────────────────────
reranker:
  modell:           "cross-encoder/ms-marco-MiniLM-L-12-v2"
  top_n:            16        # Kandidaten vor Re-Ranking
  klassen_gewichte:
    A: 1.00
    B: 0.85
    C: 0.65

# ── 1.6 Context Assembly ──────────────────────────────────────────────────────
context:
  max_tokens:       4096
  im_signals:       true      # IM-Annotierungen in Kontext einbauen
  klassen_label:    true      # [RECHTSNORM] / [FACHKONZEPT] Praeambeln
  traceability:     true      # Normreferenz in jeder Praeambel
```

---

## 10. Dateien und Implementierungsreihenfolge

### Neue Dateien in M3

```
services/ingest/
  rag_config.yaml

  prompt_suite/
    ps_hyde/
      system.txt
      user.txt
    ps_step_back/
      system.txt
      user.txt
    ps_multi_query/
      system.txt
      user.txt

  app/services/rag/
    __init__.py
    query_transformer.py    # Schritt 1.1 – dieses Konzept
    retriever.py            # Schritt 1.2 + 1.3 + 1.4
    reranker.py             # Schritt 1.5
    context_assembler.py    # Schritt 1.6 (mit IM-Signalen)
    rag_pipeline.py         # Orchestrierung aller Stufen

  app/api/routes/
    rag_router.py           # POST /api/v1/rag/query
```

### Implementierungsreihenfolge

```
Phase A – Fundament:
  1. rag_config.yaml          Konfiguration
  2. rag_router.py            FastAPI-Gerüst (gibt vorerst Mock zurück)
  3. prompt_suite/ps_hyde     Prompts anlegen und testen

Phase B – Kern 1.1:
  4. query_transformer.py
     a) QueryBundle Datenstruktur
     b) classify_query_typ()
     c) extract_norm_reference()
     d) direct() – immer aktiv
     e) hyde()   – LLM-Aufruf via Gateway
     f) step_back() – LLM-Aufruf via Gateway
     g) multi_query() – LLM-Aufruf via Gateway
     h) build_metadata_filter()

Phase C – Retrieval:
  5. retriever.py             Schritt 1.2 + 1.3 + 1.4
  6. reranker.py              Schritt 1.5

Phase D – Ausgabe:
  7. context_assembler.py     Schritt 1.6 mit IM-Signalen
  8. rag_pipeline.py          alles zusammenfügen
```

---

## 11. Abhängigkeiten aus M2

Der Query-Transformer nutzt folgende M2-Ergebnisse direkt:

| M2-Komponente | Nutzung in 1.1 |
|---|---|
| `norm_chunks.norm_reference` | Direkt-Lookup (Aufgabe 2) |
| `norm_chunks.im_signals` | IM-Filter (Aufgabe 4) |
| `svo_extractions.norm_type` | COMPETENCE/MUST/DEF-Filter |
| `ner_entities.label` | BEHÖRDE-Filter bei ENTITY-Queries |
| `nlp_config.yaml → agnostische_datenobjekte` | Query-Typ IM erkennen |
| `docs.yaml → ner_extensions` | Register-spezifische Entity-Erkennung |
| LLM-Gateway | HyDE / Step-Back / Multi-Query |
| Prompt-Suite | ps_hyde, ps_step_back, ps_multi_query |

---

## 12. Qualitätsmetriken

Für M3 werden folgende RAGAS-Metriken gemessen:

| Metrik | Beschreibung | Zielwert |
|---|---|---|
| Retrieval Recall@K | Werden relevante Chunks gefunden? | > 0.90 |
| Context Precision | Anteil relevanter Chunks im Kontext | > 0.80 |
| Answer Faithfulness | Ist die Antwort durch Quellen belegt? | > 0.95 |
| Traceability Coverage | % der Aussagen mit Normreferenz | 100% |

### Messung der Query-Transformer-Qualität

Für 1.1 spezifisch:

```
Metric 1 – Direktlookup-Precision:
  Von allen Queries mit erkannter Norm-Referenz:
  Wie viele Direkt-Lookups fanden den richtigen Chunk?
  Ziel: > 95%

Metric 2 – HyDE-Recall-Verbesserung:
  Retrieval Recall@K mit HyDE vs. ohne HyDE
  Ziel: > +10% Verbesserung

Metric 3 – Query-Typ-Accuracy:
  Anteil korrekt klassifizierter Query-Typen (manuell annotiert)
  Ziel: > 90%
```

Evaluierungsframework: RAGAS (Open Source, Python) + manuelle
Annotation durch Juristen/Fachexperten.

---

*Dieses Konzept beschreibt die technische Zielarchitektur für M3 Schritt 1.1.
Es richtet sich an Entwicklungsteams im Bereich NLP, RAG und
Public-Sector-Digitalisierung.*
