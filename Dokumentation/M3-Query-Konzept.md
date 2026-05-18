# M3 – Query-Konzept: RAG-Pipeline vollständig implementiert

> Technisches Konzept und Implementierungsstand der RAG-Pipeline
> des Meta-Normen-Registers (MNR).
>
> Stand: Mai 2026 | Meilenstein M3 – abgeschlossen
>
> Alle Schritte 1.1–1.6 sowie Schritt 2 (LLM-Antwort-Generierung)
> sind vollständig implementiert und getestet.

---

## Inhaltsverzeichnis

1. [Kontext und Einordnung](#1-kontext-und-einordnung)
2. [Gesamtübersicht M3](#2-gesamtübersicht-m3)
3. [Schritt 1.1 – Query-Transformation](#3-schritt-11--query-transformation)
4. [Schritt 1.2–1.4 – Hybrid-Retriever](#4-schritt-12-14--hybrid-retriever)
5. [Schritt 1.5 – Re-Ranker](#5-schritt-15--re-ranker)
6. [Schritt 1.6 – Context Assembly](#6-schritt-16--context-assembly)
7. [Schritt 2 – LLM-Antwort-Generierung](#7-schritt-2--llm-antwort-generierung)
8. [RAG-Pipeline (Orchestrierung)](#8-rag-pipeline-orchestrierung)
9. [Konfiguration: rag_config.yaml](#9-konfiguration-rag_configyaml)
10. [Prompt-Suite: neue Prompts](#10-prompt-suite-neue-prompts)
11. [LLM-Gateway: json_mode Kompatibilität](#11-llm-gateway-json_mode-kompatibilität)
12. [API-Endpunkte](#12-api-endpunkte)
13. [Qualitätsmetriken (Ziele)](#13-qualitätsmetriken-ziele)
14. [Abhängigkeiten aus M2](#14-abhängigkeiten-aus-m2)

---

## 1. Kontext und Einordnung

M3 implementiert die RAG-Engine – die zentrale Komponente zwischen der
NLP-Pipeline (M2) und dem Informationsmodell-Generator (M4).

```
M2 – NLP                    M3 – RAG-Engine              M4 – Informationsmodell
────────────────────────    ─────────────────────────    ──────────────────────────
SVO-Extraktion          →   1.1 Query-Transformation →   IM-Generator
NER (zweistufig)        →   1.2 Klassen-Retriever    →   Human-in-the-Middle
LLM-Normtyp             →   1.3 Parent-Child         →   Traceability-Matrix
im_signals              →   1.4 Cross-Reference      →
Knowledge Graph         →   1.5 Re-Ranker            →
                            1.6 Context Assembly     →
                            2.0 LLM-Antwort          →
```

---

## 2. Gesamtübersicht M3

### Implementierte Dateien

```
services/ingest/
  rag_config.yaml                        Zentrale Konfiguration M3

  app/services/rag/
    query_transformer.py   ✅            Schritt 1.1
    retriever.py           ✅            Schritte 1.2 + 1.3 + 1.4
    reranker.py            ✅            Schritt 1.5
    context_assembler.py   ✅            Schritt 1.6
    rag_pipeline.py        ✅            Orchestrierung + Schritt 2

  app/api/routes/
    rag_router.py          ✅            FastAPI Endpunkte

  prompt_suite/
    ps_hyde/               ✅            HyDE-Prompts
    ps_step_back/          ✅            Step-Back-Prompts
    ps_multi_query/        ✅            Multi-Query-Prompts
    ps_antwort/            ✅            Antwort-Generierung
```

### Pipeline-Ablauf

```
Query (Rohtext)
  │
  ▼  1.1 QueryTransformer
     Typ: NORM | ENTITY | IM | GENERAL
     Norm-Referenz: § 3 Abs. 1 NHundG → Direktlookup
     Vektoren: Direct + HyDE + Step-Back (parallel)
  │
  ▼  1.2–1.4 HybridRetriever
     Klasse A: Direktlookup + Vektor + FTS → RRF
     Klasse B: Req-ID-Lookup + Vektor + FTS → RRF
     Klasse C: nur Vektor (Threshold ≥ 0.72)
     + Parent-Child-Expansion
     + Cross-Reference-Expansion (max. Tiefe 2)
  │
  ▼  1.5 ReRanker
     Duplikat-Entfernung (content_hash)
     Cross-Encoder (ms-marco-MiniLM-L-12-v2)
     Klassen-Gewichtung: A×1.0 | B×0.85 | C×0.65
     Direktlookup-Bonus: ×2.0
  │
  ▼  1.6 ContextAssembler
     Präambeln: [RECHTSNORM | § 3 NHundG | TATBESTAND]
     IM-Signale: [IM-ENTITY-KANDIDAT: Hundehalter → Tabelle]
     Token-Budget: 60% von max_tokens
  │
  ▼  2.0 LLM-Antwort
     System-Prompt: ps_antwort/system.txt
     JSON-Block-Extraktion (Regex)
     Strukturiertes Ergebnis: antwort, normtyp, quellen, konfidenz
```

---

## 3. Schritt 1.1 – Query-Transformation

### Query-Typen

| Typ | Erkennung | Strategie |
|---|---|---|
| `NORM` | §, Modalverben (muss, darf, soll) | Vektor + FTS auf Klasse A |
| `IM` | Entität, Tabelle, Attribut, Datenbank | im_signals-Filter |
| `ENTITY` | zuständig, wer ist, welche Behörde | NER-Label BEHÖRDE + COMPETENCE |
| `GENERAL` | Fallback | alle Klassen, Vektor-only |

### Norm-Referenz-Extraktion

```python
# Regex: "§ 3 Abs. 1 NHundG" → Direktlookup
NORM_REF_RE = re.compile(
    r'§\s*(?P<para>\d+[a-z]?)'
    r'(?:\s+Abs\.\s*(?P<abs>\d+))?'
    r'(?:\s+Satz\s*(?P<satz>\d+))?'
    r'(?:\s+(?P<gesetz>[A-ZÄÖÜ][a-zA-ZÄÖÜäöü]{2,}G\b))?'
)
```

Bei Fund: Vektorsearch wird übersprungen – direkter DB-Lookup auf
`norm_chunks.norm_reference`.

### Transformationsstrategien

**HyDE** – Hypothetischer Normtext als Suchvektor:
```
Query: "Was muss ein Hundehalter nachweisen?"
LLM:   "Wer einen Hund hält, hat der zuständigen Gemeinde
        die Sachkunde nachzuweisen."
Vektor: embed(LLM-Antwort) → trifft echte Chunks besser
```

**Step-Back** – Abstraktion für §-spezifische Queries:
```
Query: "§ 9 Satz 1 NHundG Frist"
LLM:   "Welche Fristen gelten für Halter gefährlicher Hunde?"
```

**Multi-Query** – 3 Varianten für vage Queries (deaktiviert, per Bedarf):
```
Query: "Hund gefährlich Pflichten"
→ ["Pflichten des Halters eines gefährlichen Hundes",
   "Erlaubnispflicht gefährlicher Hund Fachbehörde",
   "Wesenstest Rechtsfolgen"]
```

### QueryBundle

```python
@dataclass
class QueryBundle:
    original_query:   str
    query_typ:        str        # NORM | ENTITY | IM | GENERAL
    norm_reference:   str|None   # "§ 3 Abs. 1 NHundG" oder None
    vektoren:         list[QueryVector]
    metadata_filter:  dict
    im_filter:        bool
    direktlookup:     bool
```

---

## 4. Schritt 1.2–1.4 – Hybrid-Retriever

### Klassenspezifische Retrieval-Modi

| Dimension | Klasse A | Klasse B | Klasse C |
|---|---|---|---|
| Direktlookup | norm_reference | requirement_id | – |
| Vektorsuche | ≥ 0.60 | ≥ 0.60 | ≥ 0.72 |
| FTS | ja | ja | ja |
| Parent-Fetch | immer | selektiv (>450 Token) | nein |
| Cross-Reference | ja (Tiefe 2) | nein | nein |

### RRF-Fusion

Reciprocal Rank Fusion kombiniert Vektor- und FTS-Ranking:

```
RRF-Score = Σ 1 / (k + rank_i)    (k = 60)

Chunk in Vektor-Rang 2 + FTS-Rang 5:
  = 1/(60+2) + 1/(60+5) = 0.0161 + 0.0154 = 0.0315
```

Chunks die in beiden Listen erscheinen werden bevorzugt.

### Parent-Child-Expansion

```
Klasse A:  Child-Chunk (Absatz) gefunden
           → §-Parent immer nachladen (Kontext-Anker)

Klasse B:  Unterkapitel-Chunk > 450 Token
           → Kapitel-Parent nachladen
           Tabellen-Chunk → beschreibenden Abschnitt nachladen

Klasse C:  kein Parent-Fetch
```

### Cross-Reference-Expansion (nur Klasse A)

```
§ 9 "gilt entsprechend § 5"
  → cross_references: [uuid-§5]
  → § 5 automatisch nachladen
  → Rekursiv: max. Tiefe 2
```

---

## 5. Schritt 1.5 – Re-Ranker

### Ablauf

```
1. Duplikat-Entfernung (chunk_id + content_hash)
2. Top-N vorauswählen (Standard: 16)
3. Cross-Encoder: ms-marco-MiniLM-L-12-v2
4. Klassen-Gewichtung multiplizieren
5. Direktlookup-Bonus anwenden
6. Sortieren → Top-K auswählen (Standard: 8)
```

### Scoring-Formel

```
Score_final = CrossEncoder(query, chunk) × klassen_gewicht × direktlookup_boost

Beispiel:
  § 3 Sachkunde, Cross-Encoder: 4.9
  Klasse A × 1.0 = 4.9
  Direktlookup × 2.0 = 9.8   ← steht immer oben
```

### Fallback

Wenn `sentence_transformers` nicht installiert: RRF-Score aus
Schritt 1.2 wird unverändert verwendet.

---

## 6. Schritt 1.6 – Context Assembly

### Präambel-Format

```
[RECHTSNORM | § 3 Abs. 1 NHundG | TATBESTAND]
[IM-ENTITY-KANDIDAT: Hundehalter → Tabelle]
[IM-ATTRIBUT-KANDIDAT: Sachkunde → Spalte]
<Chunk-Inhalt>
```

| Klasse | Präfix |
|---|---|
| A | `[RECHTSNORM | §-Referenz | chunk_type]` |
| B | `[FACHKONZEPT | heading_breadcrumb]` |
| C | `[AUSLEGUNG | Titel | Konfidenz: niedrig]` |

### IM-Signal-Annotierungen

Aus `norm_chunks.im_signals` (M2):

```
[IM-ENTITY-KANDIDAT: Hundehalter → Tabelle]
[IM-ENTITY-DEF: Hundehalter = kanonische Definition]
[IM-ATTRIBUT-KANDIDAT: Sachkunde → Spalte]
[IM-PERSISTENZ: Datenpersistenz erforderlich]
[IM-RELATION: Hundehalter → Hund]
```

### Token-Budget

```
max_tokens: 4096
Kontext-Budget: 4096 × 0.60 = 2458 Token
Verbleibend für System-Prompt + Query + Antwort: 1638 Token
```

---

## 7. Schritt 2 – LLM-Antwort-Generierung

### Ablauf

```
Kontext (aus 1.6) + Query
    │
    ▼
LLM (ps_antwort/system.txt + user.txt)
    │
    ▼
JSON-Extraktion aus Markdown-Block
    │
    ▼
RAGAntwort(antwort, normtyp, quellen, konfidenz, hinweis)
```

### Output-Schema

```json
{
  "antwort":            "Wer einen Hund hält, muss...",
  "normtyp":            "MUST",
  "quellen":            ["§ 3 Abs. 1 NHundG"],
  "konfidenz":          "hoch",
  "hinweis":            null,
  "nicht_beantwortbar": false
}
```

### JSON-Extraktion

```python
# Schritt 1: Markdown-Fence extrahieren
m = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)

# Schritt 2: JSON-Objekt suchen falls kein Fence
if not text.startswith('{'):
    obj_match = re.search(r'\{[\s\S]+\}', text)

# Schritt 3: List-Fallback
if isinstance(data, list):
    data = data[0] if data else {}
```

---

## 8. RAG-Pipeline (Orchestrierung)

```python
# Verwendung im rag_router:
pipeline = RAGPipeline(pool=pool, embedder=embedder)
result   = await pipeline.run(query=query, debug=True)

# RAGResult enthält:
result.chunks          # Re-Ranked Chunks
result.kontext         # Assemblierter Kontext
result.traceability    # Quellenverweise
result.llm_antwort     # Strukturierte Antwort
result.stats           # Timing je Schritt
result.debug           # Zwischenstände (wenn debug=True)
```

### Timing-Statistik (typisch)

| Schritt | Dauer |
|---|---|
| 1.1 Transform | 1.000–4.000 ms (LLM für HyDE) |
| 1.2–1.4 Retrieval | 50–200 ms |
| 1.5 Re-Ranking | 100–500 ms |
| 1.6 Assembly | < 10 ms |
| 2.0 Antwort | 1.000–5.000 ms (LLM) |

---

## 9. Konfiguration: rag_config.yaml

```yaml
query_transformation:
  hyde:         {enabled: true,  gewicht: 0.8, prompt_key: ps_hyde}
  step_back:    {enabled: true,  gewicht: 0.6, prompt_key: ps_step_back}
  multi_query:  {enabled: false, count: 3,     prompt_key: ps_multi_query}

retrieval:
  top_k_pro_klasse: 6
  top_k_final:      8
  score_thresholds: {class_a: 0.60, class_b: 0.60, class_c: 0.72}
  fts_enabled:      true
  rrf_k:            60

reranker:
  modell:      "cross-encoder/ms-marco-MiniLM-L-12-v2"
  top_n:       16
  klassen_gewichte: {A: 1.00, B: 0.85, C: 0.65}
  direktlookup_boost: 2.0

context:
  max_tokens:    4096
  im_signale:    true
  klassen_label: true
  traceability:  true

antwort:
  enabled:     true
  prompt_key:  "ps_antwort"
  max_tokens:  1024
```

---

## 10. Prompt-Suite: neue Prompts

| Key | Datei | Zweck |
|---|---|---|
| `ps_hyde` | `prompt_suite/ps_hyde/` | Hypothetischen Normtext generieren |
| `ps_step_back` | `prompt_suite/ps_step_back/` | Query abstrahieren |
| `ps_multi_query` | `prompt_suite/ps_multi_query/` | 3 Query-Varianten |
| `ps_antwort` | `prompt_suite/ps_antwort/` | Strukturierte Antwort |

**Wichtig:** Dateinamen müssen exakt `system.txt` und `user.txt` sein.
Nicht `ps_antwort_system.txt` o.ä.

---

## 11. LLM-Gateway: json_mode Kompatibilität

### Problem

`json_mode=True` setzt `responseMimeType: application/json` im
Gemini-Request. Beide Gemini-Provider ignorieren dabei den System-Prompt
und geben ein generisches JSON-Template zurück.

### Lösung

```yaml
# llm_gateway_config.yaml
providers:
  gemini:            json_mode_unterstuetzt: false
  gemini_flash_lite: json_mode_unterstuetzt: false
  openai:            json_mode_unterstuetzt: true
  anthropic:         json_mode_unterstuetzt: false
  ollama:            json_mode_unterstuetzt: false
```

Das Gateway deaktiviert `responseMimeType` automatisch wenn
`json_mode_unterstuetzt: false`. JSON wird per Regex aus dem
Freitext extrahiert.

---

## 12. API-Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| `GET` | `/api/v1/rag/status` | Pipeline-Status, Config geladen |
| `POST` | `/api/v1/rag/query` | RAG-Abfrage ausführen |
| `GET` | `/api/v1/rag/config` | rag_config.yaml als JSON |
| `POST` | `/api/v1/rag/reload` | Config neu laden |

### Request/Response

```bash
curl -X POST /api/v1/rag/query \
  -d '{"query": "Was muss ein Hundehalter nachweisen?",
       "debug": true}'

# Response:
{
  "query":            "Was muss ein Hundehalter nachweisen?",
  "query_typ":        "NORM",
  "chunks":           [...],
  "kontext":          "Folgende 8 Quellen...",
  "traceability":     [...],
  "direktlookup":     false,
  "antwort":          "Wer einen Hund hält, muss...",
  "antwort_normtyp":  "MUST",
  "antwort_quellen":  ["§ 3 Abs. 1 NHundG"],
  "antwort_konfidenz":"hoch",
  "antwort_hinweis":  null,
  "debug_info":       {...}
}
```

---

## 13. Qualitätsmetriken (Ziele)

| Metrik | Zielwert |
|---|---|
| Retrieval Recall@K | > 0.90 |
| Context Precision | > 0.80 |
| Answer Faithfulness | > 0.95 |
| Traceability Coverage | 100% |
| Halluzinationsrate | < 0.02 |

Evaluierungsframework: RAGAS (Open Source, Python) +
manuelle Annotation durch Juristen.

---

## 14. Abhängigkeiten aus M2

| M2-Komponente | Nutzung in M3 |
|---|---|
| `norm_chunks.norm_reference` | Direktlookup Schritt 1.2 |
| `norm_chunks.im_signals` | IM-Annotierungen Schritt 1.6 |
| `norm_chunks.embedding` | Vektorsuche Schritt 1.2 |
| `norm_chunks.cross_references` | Cross-Ref-Expansion 1.4 |
| `svo_extractions.norm_type` | COMPETENCE/MUST-Filter |
| `ner_entities.label` | BEHÖRDE-Filter ENTITY-Queries |
| LLM-Gateway | HyDE, Step-Back, Antwort-Generierung |
| Prompt-Suite | ps_hyde, ps_step_back, ps_antwort |

---

*Dieses Konzept beschreibt die vollständig implementierte M3 RAG-Pipeline.
M4 (Informationsmodell-Generierung) baut direkt auf dem assemblierten
Kontext mit IM-Signalen auf.*
