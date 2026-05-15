# M2 – NLP-Processing: Konzept und Implementierung

> Technisches Konzept der NLP-Verarbeitungspipeline
> des Meta-Normen-Registers (MNR).
>
> Stand: Mai 2026 | Meilenstein M2 – abgeschlossen
>
> Dieses Dokument beschreibt Aufbau, Datenstrukturen und
> Implementierungsentscheidungen der NLP-Pipeline von der
> spaCy-Analyse bis zum angereicherten Knowledge Graph.

---

## Inhaltsverzeichnis

1. [Kontext und Einordnung](#1-kontext-und-einordnung)
2. [Gesamtübersicht](#2-gesamtübersicht)
3. [SVO-Extraktion](#3-svo-extraktion)
4. [NER – Named Entity Recognition](#4-ner--named-entity-recognition)
5. [LLM-Normtyp-Klassifikation](#5-llm-normtyp-klassifikation)
6. [Knowledge Graph Export](#6-knowledge-graph-export)
7. [IM-Signal-Erzeugung](#7-im-signal-erzeugung)
8. [Fragen-Filter](#8-fragen-filter)
9. [Konfiguration](#9-konfiguration)
10. [Datenbank-Schema](#10-datenbank-schema)
11. [API-Endpunkte und Worker](#11-api-endpunkte-und-worker)
12. [Qualitätsmetriken M2](#12-qualitätsmetriken-m2)

---

## 1. Kontext und Einordnung

### Einbettung in die MNR-Architektur

M2 baut auf den gechunkten und eingebetteten Dokumenten aus M1 auf.
Die NLP-Ergebnisse (SVOs, NER-Entitäten, Normtypen) sind die
semantische Grundlage für M3 (RAG-Engine) und M4 (Informationsmodell).

```
M1 – Ingest            M2 – NLP-Processing            M3/M4 – RAG & IM
─────────────────────  ───────────────────────────    ──────────────────
norm_chunks         →  SVO-Extraktion              →  Knowledge Graph
(A/B/C Klassen)     →  NER (zweistufig)            →  Hybrid-Retrieval
Embeddings          →  LLM-Normtyp                 →  Informationsmodell
                    →  IM-Signale                  →  Artefakt-Generierung
                    →  Knowledge Graph Export
```

### Grundprinzip: Hybrid-Ansatz

M2 kombiniert **drei Klassifikationsebenen** zu minimaler UNKNOWN-Rate:

```
Ebene 1 – Regex-Patterns (schnell, deterministisch):
  MUST bei "muss", MAY bei "kann", DEADLINE bei "binnen" usw.

Ebene 2 – spaCy Dependency Parsing (strukturell):
  Subjekt/Objekt aus Satzstruktur extrahieren
  Passiv-Auflösung, Infinitiv-Cluster

Ebene 3 – LLM-Batch-Klassifikation (semantisch):
  20 UNKNOWN-SVOs = 1 API-Call
  Kontextverständnis für komplexe Satzstrukturen
```

---

## 2. Gesamtübersicht

```
norm_chunks (aus M1)
  │
  ▼
spaCy-Analyse
  Tokenisierung, POS-Tagging, Dependency Parsing
  Modell: de_core_news_lg
  │
  ├──────────────────────────────────────────────────────────────────
  │                          Zwei parallele Extraktion-Pfade
  │
  ▼                                                    ▼
3. SVO-Extraktion                              4. NER-Extraktion
   Subjekt + Prädikat + Objekt                    Zweischicht-Architektur
   Normtyp via Regex (Ebene 1+2)                  Stufe 1: Regelbasiert
   Passiv-Auflösung                               Stufe 2: Flair Legal NER
   Infinitiv-Cluster                              Schicht 1: Agnostisch
        │                                         Schicht 2: Register-spez.
        ▼
5. LLM-Normtyp (Ebene 3)
   Batch: 20 UNKNOWN-SVOs = 1 API-Call
   Provider: Gemini 2.5 Flash / 3.1 Flash-Lite / Ollama
        │
        ▼
svo_extractions + ner_entities
        │
        ▼
6. Knowledge Graph Export         7. IM-Signal-Erzeugung
   Apache Jena Fuseki                norm_chunks.im_signals
   SPARQL 1.1 / RDF                  entity_kandidaten
                                     attribut_kandidaten
                                     persistenz-Signale
```

---

## 3. SVO-Extraktion

### Definition

SVO (Subjekt-Verb-Objekt) zerlegt einen Normtext in drei semantische
Grundkomponenten:

**Subjekt** – Wer handelt? Akteur oder Rolle, die eine Handlung
ausführt oder einer Pflicht unterliegt.

**Prädikat (Verb)** – Was wird getan? Angereichert um die
**rechtliche Modalität** (Normtyp: MUST, MAY, MUST_NOT usw.).

**Objekt** – Worauf bezieht sich die Handlung? Betroffenes
Datenobjekt, Verwaltungsakt oder Entität.

```
Normtext: "Die Fachbehörde überwacht die Einhaltung der Vorschriften."

Subjekt:   Fachbehörde      [BEHÖRDE]
Prädikat:  überwacht        [COMPETENCE]
Objekt:    Einhaltung       [DATENOBJEKT]
```

### Extraktion in drei Schichten

```python
def extract(analysis: ChunkAnalysis) -> list[SVOTriple]:

    # Schicht 1: Stop-Wort-Filter
    # Artikel, Pronomen, bekannte Artefakte herausfiltern
    stop_subjects   = {"er", "sie", "es", "man", "der", "die", ...}
    stop_objects    = {"der", "die", "das", "erforderlich", ...}
    stop_predicates = {"§", "S.", "Nr.", "Hundegesetz", ...}

    # Schicht 2: Normtyp via Regex-Patterns
    norm_type = classify_normtype(sentence.text)
    # z.B.: "muss" → MUST, "kann" → MAY, "darf nicht" → MUST_NOT

    # Schicht 3: Stop-Prädikate (Substantive als Verben)
    if predicate.lower() in stop_predicates:
        continue    # verwerfen

    # Objekt-Suche: Root → Vollverb-Cluster → Passiv-Infinitiv
    ...
```

### Vollverb-Cluster-Suche (Objekt)

Häufigster Fehlerfall: Das Objekt hängt am Infinitiv, nicht am Hilfsverb:

```
"Der Hundehalter muss die Sachkunde nachweisen."

spaCy-Root: "muss" (AUX)
  │
  └─► Kinder suchen nach VERB:
        "nachweisen" (VERB)
          │
          └─► Objekt von "nachweisen":
                "Sachkunde" ✅
```

```python
# Zweistufige Objekt-Suche
if object_token is None and root.pos_ == "AUX":
    for child in root.children:
        if child.pos_ in ("VERB", "AUX") and child.dep_ in ("oc", "mo", ...):
            deeper = find_dep(child, ["oa", "obj", "pd", "og", "op", "da"])
            if deeper is not None:
                object_token = deeper
                break
```

### Passiv-Infinitiv-Behandlung

```
"Der Wesenstest ist von der Fachbehörde durchzuführen."

Erkennung: re.search(r'\bzu\w+en\b', predicate)
Logik:     Subjekt wird logisches Objekt (grammatisch: Subjekt des Passivs)
Ergebnis:  Objekt = Wesenstest, Subjekt = None (unbekannt aus Kontext)
```

### Normtypen

| Normtyp | Bedeutung | Typische Auslöser |
|---|---|---|
| `MUST` | Pflicht | muss, hat zu, ist zu, soll, bedarf, pflichtig, ist nachzuweisen |
| `MAY` | Erlaubnis / Befugnis | kann, darf, ist berechtigt, erhält auf Antrag |
| `MUST_NOT` | Verbot | darf nicht, ist untersagt, hat keine aufschiebende Wirkung |
| `DEF` | Definition | im Sinne, gilt als, versteht man |
| `EXCEPT` | Ausnahme | es sei denn, sofern nicht, bleibt, gilt nicht |
| `DEADLINE` | Frist | binnen, spätestens, unverzüglich |
| `COMPETENCE` | Zuständigkeit | zuständig ist, obliegt, überwacht, erfüllen |
| `SCOPE` | Geltungsbereich | gilt für, findet Anwendung, dient der, entspricht |
| `CHANGE` | Rechtsänderung | geändert, aufgehoben, ersetzt |
| `STATUS` | Zustandsbeschreibung | ist anerkannt, besitzt, stellt fest |

---

## 4. NER – Named Entity Recognition

### Definition

NER erkennt und klassifiziert **benannte Entitäten** in Texten –
Personen, Organisationen, Orte, Dokumente. Im MNR-Kontext:
Behörden, Rollen, Gesetze, Datenobjekte, Fristen.

```
"Die Fachbehörde erteilt dem Hundehalter die Erlaubnis nach § 8 NHundG."
      │                    │                  │              │
   BEHÖRDE              ROLLE            DATENOBJEKT      GESETZ
```

### Zweistufige Technologie

**Stufe 1 – Regelbasiert:**
Suffix-, Exact- und Regex-Matching. Schnell, deterministisch,
ideal für Verwaltungsbegriffe die im juristischen Trainingskorpus selten sind.

```yaml
# nlp_config.yaml → ner.patterns
BEHÖRDE:
  suffixes: ["behörde", "ministerium", "amt", "register", "stelle"]
  exact:    ["Gemeinde", "Fachbehörde", "Fachministerium"]

ROLLE:
  exact:    ["Person", "Antragsteller", "Antragstellerin"]
  suffixes: ["halter", "halterin", "führer", "führerin"]
```

**Stufe 2 – Flair Legal NER:**
Modell `flair/ner-german-legal`, trainiert auf 67.000 deutschen
Gerichtsentscheidungen. Erkennt 19 feingranulare juristische
Entitätsklassen (GS=Gesetz, ORG=Organisation, LD=Land usw.).

```
Flair GS  (Gesetz)      → MNR GESETZ
Flair ORG (Organisation)→ MNR BEHÖRDE
Flair LD  (Land)        → MNR ORT
Flair RR  (Richter)     → MNR ROLLE
```

### Zweischicht-Architektur (agnostisch + register-spezifisch)

Eingeführt zur Trennung von plattformweiten und fachspezifischen Begriffen:

```
Text-Token
    │
    ├─► Schicht 2 (Register-spezifisch, höchste Priorität):
    │      docs.yaml → ner_extensions
    │      "Hundehaltung" → DATENOBJEKT  (nur im NHundG)
    │      "Transponder"  → DATENOBJEKT  (nur im NHundG)
    │
    ├─► Schicht 1 (Agnostisch, plattformweit):
    │      nlp_config.yaml → agnostische_datenobjekte
    │      "Antrag"        → DATENOBJEKT  (in jedem Verwaltungsgesetz)
    │      "Erlaubnis"     → DATENOBJEKT
    │      "Nachweis"      → DATENOBJEKT
    │
    └─► Flair + Regelbasiert (Standard-Klassifikation)
```

```yaml
# nlp_config.yaml – Schicht 1
agnostische_datenobjekte:
  - "Antrag"
  - "Erlaubnis"
  - "Genehmigung"
  - "Nachweis"
  - "Bescheinigung"
  - "Register"
  - "Prüfung"
  ...  (30 plattformweite Verwaltungsbegriffe)

# docs.yaml – Schicht 2 (register-spezifisch)
Hundegesetz.pdf:
  ner_extensions:
    DATENOBJEKT: ["Hundehaltung", "Wesenstest", "Transponder", ...]
    ROLLE:       ["Hundehalter", "Hundeführer", "Tierhalter"]
```

### Kombinationsstrategie: Merge

```python
# merge-Strategie: Beste Quelle gewinnt
def merge(rule_entities, flair_entities) -> list[NEREntity]:
    # Bei Überschneidung: höhere Konfidenz gewinnt
    # Flair-Treffer ergänzen regelbasierte Treffer
    # Duplikat-Entfernung via content_hash
    ...
```

### Label-Korrekturen (globale Flair-Fehler)

```yaml
# nlp_config.yaml → ner.label_corrections
# Nur für strukturell IMMER falsche Klassifikationen.
# Register-spezifische Korrekturen → docs.yaml → ner_extensions

"Feststellung":    "SONSTIGE"   # fälschlich als GESETZ
"Antragsteller":   "ROLLE"      # fälschlich als SONSTIGE
"Antragstellerin": "ROLLE"
```

---

## 5. LLM-Normtyp-Klassifikation

### Motivation

spaCy-Regex-Patterns erreichen ca. 50% Normtyp-Abdeckung.
Komplexe Satzstrukturen (Schachtelsätze, Passiv, Negation) erfordern
semantisches Verständnis – das liefert ein LLM.

### Batch-Prompt-Strategie

20 UNKNOWN-SVOs werden in **einem einzigen API-Call** klassifiziert.
Das verhindert Rate-Limiting beim Free-Tier.

```
vorher (ohne Batching):     20 SVOs = 20 API-Calls  → 429-Fehler
nachher (mit Batching):     20 SVOs = 1 API-Call    → kein Rate-Limiting
```

### User-Prompt (ps_normtyp)

```
Klassifiziere die Normtypen der folgenden {{count}} deutschen Rechtstexte.

Gib ein JSON-Array zurück – ein Objekt je Eintrag, gleiche Reihenfolge.

SVO-001:
  Normreferenz: § 3 Abs. 1 NHundG
  Prädikat:     ist nachzuweisen
  Text:         "Sie ist der Gemeinde auf Verlangen nachzuweisen."

SVO-002:
  ...

Ausgabe-Schema:
[
  {
    "id": "SVO-001",
    "norm_type": "MUST|MAY|...|UNKNOWN",
    "konfidenz": "hoch|mittel|niedrig",
    "begruendung": "kurze Begründung (max. 10 Wörter)"
  }
]
```

### JSON-Repair bei abgeschnittenen Antworten

Bei großen Batches kann die LLM-Antwort das Token-Limit überschreiten:

```python
def repair_json_array(text: str) -> str:
    """
    Repariert abgeschnittene JSON-Arrays.
    Findet letztes vollständiges Objekt und schließt Array.
    """
    depth = 0
    last_close = -1
    # Klammer-Tiefe zählen, letztes vollständiges } finden
    for i, ch in enumerate(text):
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_close = i
    # Array bis letztem vollständigen Objekt + "]"
    return text[:last_close + 1].rstrip().rstrip(',') + "\n]"
```

### Konfiguration

```yaml
# nlp_config.yaml → llm_normtyp
llm_normtyp:
  enabled:          true
  batch_size:       20      # SVOs pro Prompt/API-Call
  min_confidence:   0.75    # Unter diesem Wert → UNKNOWN bleibt
  test_max_batches: 0       # 0 = unbegrenzt, 3 = max. 60 SVOs
  delay_sek:        1.0     # Pause zwischen Batch-Calls
  log_prompts:      true    # Logging aktivieren
```

### Logging

Jeder LLM-Aufruf wird in getrennten Dateien protokolliert:

```
logs/
  llm_request_20260515_153000.log   ← alle Prompts (System + User)
  llm_response_20260515_153000.log  ← alle Antworten + Fehler
```

Requests und Responses sind durch eine fortlaufende Nummer
(`#0001`, `#0002`, …) eindeutig einander zugeordnet.

---

## 6. Knowledge Graph Export

### Technologie

```
Speicher:    Apache Jena Fuseki (TDB2-Dataset "mnr")
Protokoll:   REST API + SPARQL 1.1
Format:      RDF/Turtle
Ontologie:   FOAF + Verwaltungsontologie (LIDO/XÖV)
```

### Export-Inhalt

```turtle
# SVO-Triple als RDF
<svo:Hundehalter_COMPETENCE_Einhaltung>
  a mnr:SVOTriple ;
  mnr:subject     mnr:Hundehalter ;
  mnr:predicate   "überwacht" ;
  mnr:object      mnr:Einhaltung ;
  mnr:normType    "COMPETENCE" ;
  mnr:confidence  0.893 ;
  mnr:source      "§ 14 NHundG" .

# NER-Entität
mnr:Fachbehörde
  a mnr:Behörde ;
  rdfs:label "Fachbehörde" ;
  mnr:confidence 0.823 ;
  mnr:source     "NHundG" .
```

### Export-Statistik (NHundG-Testlauf)

```
Gesamte Tripel:  4.022
  SVO-Tripel:    1.536
  NER-Tripel:    2.470
  Norm-Tripel:      16
```

### Einsatzgebiete des Knowledge Graph

| Einsatz | Beschreibung |
|---|---|
| Impact-Analyse | Welche Entitäten sind betroffen wenn § X geändert wird? |
| Register-Abhängigkeiten | Welche Register referenzieren dieselbe Norm? |
| Context-Retrieval | Graph-RAG: Kontextualisierung via SPARQL (M3) |

---

## 7. IM-Signal-Erzeugung

### Motivation

In Fachgesetzen stecken zwei Informationsebenen gleichzeitig:

```
§ 3 Abs. 1 NHundG:
"Wer einen Hund hält, muss die erforderliche Sachkunde besitzen."

Normebene:   MUST → Hundehalter → besitzen → Sachkunde
IM-Ebene:    ENTITY: Hundehalter  → Tabelle "hundehalter"
             ATTRIBUT: Sachkunde  → Spalte "sachkunde_nachweis"
             RELATION: Hundehalter HAS Sachkunde
```

### Drei Signal-Typen

**Entity-Kandidat:**
NER-Entität mit Typ ROLLE oder DATENOBJEKT + Normtyp MUST oder DEF

```json
"entity_kandidaten": ["Hundehalter", "Fachbehörde"]
```

**Attribut-Kandidat:**
Normtyp MUST + Prädikat aus `attribut_praedikate`-Liste
(ist anzugeben, enthält, umfasst, besteht aus)

```json
"attribut_kandidaten": ["Sachkunde", "Name", "Anschrift"]
```

**Persistenz-Signal:**
Normtyp MUST + Prädikat aus `persistenz_praedikate`-Liste
(speichern, führen, anlegen, registrieren, eintragen)

```json
"persistenz": true
```

### Konfiguration

```yaml
# nlp_config.yaml → ner

agnostische_datenobjekte:
  - "Antrag"
  - "Erlaubnis"
  - "Nachweis"
  - "Register"
  ...

persistenz_praedikate:
  - "speichern"
  - "führen"
  - "anlegen"
  - "registrieren"
  - "eintragen"
  - "erfassen"
  ...

attribut_praedikate:
  - "ist anzugeben"
  - "enthält"
  - "umfasst"
  - "besteht aus"
  - "hat"
  ...
```

### Speicherung

```sql
-- norm_chunks.im_signals (JSONB)
{
  "entity_kandidaten":   ["Hundehalter", "Person"],
  "attribut_kandidaten": ["Sachkunde", "Name"],
  "persistenz":          true,
  "entity_def":          "Hundehalter",
  "relation_kandidaten": [{"von": "Hundehalter", "zu": "Hund"}]
}
```

---

## 8. Fragen-Filter

### Problem

FAQ-Dokumente (Klasse B/C) enthalten Fragen die für den NLP-Worker
ungeeignet sind (keine normativen Aussagen, kein SVO möglich).

### Lösung

Fragen werden beim Ingest erkannt, aus den Chunks ausgeschlossen
und in `filtered_questions` gespeichert. Sie können nachträglich
reviewt und bei Bedarf eingeschlossen werden.

```yaml
question_filter:
  nlp:
    enabled: true
    apply_to_classes: [B, C]
    action: "skip"    # NLP überspringt Fragen-Chunks
```

### Statistik (NHundG-Testlauf)

```
Gefilterte Fragen gesamt: 92
  FAQ1 – Sachkundenachweis: 42 Fragen
  FAQ2 – Hunderegister:     50 Fragen
Entschieden (verworfen):    92 (alle – kein normativer Inhalt)
```

---

## 9. Konfiguration

### nlp_config.yaml – Zentrale NLP-Steuerung

```yaml
# Auszug der wichtigsten Sektionen

spacy:
  model:      "de_core_news_lg"
  batch_size: 32

svo:
  min_confidence:            0.5
  pronoun_confidence_penalty: 0.3
  subject_deps:  [sb, sp, nsubj]
  object_deps:   [oa, og, op, oc, pd, obj, obl, da, mo, nk, attr]
  stop_subjects:   [er, sie, es, man, der, ...]
  stop_objects:    [der, die, das, erforderlich, ...]
  stop_predicates: [§, S., Nr., Hundegesetz, ...]

normtypen:
  MUST:
    patterns: ['\bmuss\b', '\bist\s+zu\b', ...]
  MAY:
    patterns: ['\bkann\b', '\bdarf\b', ...]
  # ... alle 10 Normtypen

ner:
  flair_enabled:        true
  flair_model:          "flair/ner-german-legal"
  flair_min_confidence: 0.75
  combination_strategy: "merge"
  agnostische_datenobjekte: [...]
  persistenz_praedikate:    [...]
  attribut_praedikate:      [...]
  label_corrections:        {...}

llm_normtyp:
  enabled:          true
  batch_size:       20
  min_confidence:   0.75
  test_max_batches: 0
  delay_sek:        1.0
  log_prompts:      true

worker:
  process_classes:    [A, B, C]
  chunk_batch_size:   50
  overwrite_existing: true
```

---

## 10. Datenbank-Schema

### NLP-Tabellen

```sql
nlp_jobs (
  id UUID PRIMARY KEY,
  doc_id UUID REFERENCES norm_documents(id),
  status TEXT CHECK (status IN ('pending','running','done','error')),
  svo_count INT,
  ner_count INT,
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
)

svo_extractions (
  id UUID PRIMARY KEY,
  chunk_id UUID REFERENCES norm_chunks(id),
  doc_id UUID REFERENCES norm_documents(id),
  job_id UUID,
  subject TEXT,
  subject_type TEXT,
  predicate TEXT,
  predicate_lemma TEXT,
  object TEXT,
  object_type TEXT,
  context TEXT,
  norm_type TEXT CHECK (norm_type IN (
    'MUST','MAY','MUST_NOT','DEF','EXCEPT','DEADLINE',
    'COMPETENCE','SCOPE','CHANGE','STATUS','UNKNOWN'
  )),
  norm_type_confidence NUMERIC(4,3),
  confidence NUMERIC(4,3),
  sentence_text TEXT,
  created_at TIMESTAMPTZ
)

ner_entities (
  id UUID PRIMARY KEY,
  chunk_id UUID REFERENCES norm_chunks(id),
  doc_id UUID REFERENCES norm_documents(id),
  job_id UUID,
  text TEXT,
  label TEXT,    -- BEHÖRDE|ROLLE|GESETZ|DATENOBJEKT|FRIST|ORT|SONSTIGE
  start_char INT,
  end_char INT,
  confidence NUMERIC(4,3),
  source TEXT,   -- rule | flair
  created_at TIMESTAMPTZ
)
```

### DB-Migrationen M2

| Datei | Inhalt |
|---|---|
| `04_nlp.sql` | `nlp_jobs`, `svo_extractions`, `ner_entities` |
| `08_normtype_constraint.sql` | CHECK-Constraint für alle 11 Normtypen |
| `09_im_signals.sql` | `norm_chunks.im_signals JSONB` + GIN-Index |

---

## 11. API-Endpunkte und Worker

### NLP-Worker

```bash
# Alle Chunks verarbeiten (spaCy + LLM-Klassifikation):
python nlp_worker.py --run

# Ein Dokument:
python nlp_worker.py --run --doc-id <uuid>

# NLP-Reset und Neustart:
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  -c "TRUNCATE TABLE svo_extractions, ner_entities, nlp_jobs CASCADE;"
python nlp_worker.py --run
```

### API-Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| `POST` | `/api/v1/nlp/run` | NLP-Job starten |
| `GET` | `/api/v1/nlp/status` | Aktueller Job-Status |
| `GET` | `/api/v1/nlp/jobs` | Alle Jobs auflisten |
| `GET` | `/api/v1/reports/nlp_quality` | Qualitätsbericht (7 Abschnitte) |
| `GET` | `/api/v1/reports/nlp_monitor` | Live-Status |
| `GET` | `/api/v1/kg/stats` | Knowledge-Graph-Statistik |
| `POST` | `/api/v1/kg/export` | Export nach Fuseki |
| `POST` | `/api/v1/llm/complete` | LLM-Direktaufruf mit Prompt |
| `GET` | `/api/v1/llm/status` | Gateway-Status |

### Qualitätsbericht (7 Abschnitte)

```bash
python nlp_quality_report.py
```

```
1. Überblick          Chunks, SVOs, NER, letzter Lauf
2. SVO-Normtypen      Verteilung, UNKNOWN-Rate, Konfidenz
3. SVO-Vollständigkeit Mit Subjekt/Objekt, Pronomen-Rate
4. NER-Qualität       Flair/Regelbasiert-Verteilung, Labels
5. Abkürzungen        Auflösungsrate, häufigste Abkürzungen
6. NLP-Abdeckung      % Chunks mit SVO/NER
7. Handlungsempfehlungen Automatisch generiert mit Maßnahmen
```

---

## 12. Qualitätsmetriken M2

### Erreichte Ergebnisse (NHundG-Testlauf, 288 SVOs)

| Kennzahl | Ergebnis | Ziel | Status |
|---|---|---|---|
| UNKNOWN-Rate | 16.0% | < 20% | erreicht |
| Mit Subjekt | 65.3% | > 80% | offen |
| Mit Objekt | ~52% | > 50% | erreicht |
| Pronomen als Subjekt | 0.0% | < 15% | erreicht |
| Ø SVO-Konfidenz | 0.823 | > 0.65 | erreicht |
| Ø Normtyp-Konfidenz | 0.718 | > 0.75 | knapp |
| Chunks mit SVO | 91.5% | > 60% | erreicht |
| Chunks mit NER | 90.4% | > 70% | erreicht |
| Flair-Anteil | 21.5% | > 40% | offen |

### Normtyp-Verteilung (finale Ergebnisse)

| Normtyp | Anzahl | Anteil | Ø Konfidenz |
|---|---|---|---|
| MUST | 81 | 29.5% | 0.889 |
| MAY | 52 | 18.9% | 0.919 |
| UNKNOWN | 44 | 16.0% | 0.673 |
| COMPETENCE | 42 | 15.3% | 0.891 |
| STATUS | 14 | 5.1% | 0.869 |
| SCOPE | 14 | 5.1% | 0.930 |
| DEADLINE | 11 | 4.0% | 0.974 |
| EXCEPT | 6 | 2.2% | 0.985 |
| MUST_NOT | 5 | 1.8% | 0.928 |
| CHANGE | 4 | 1.5% | 0.580 |
| DEF | 2 | 0.7% | 0.788 |

### Hinweise zu offenen Zielwerten

**Subjekt-Rate (65.3% statt > 80%):**
Viele deutsche Normtexte sind passivisch formuliert
(„ist durchzuführen", „ist nachzuweisen") – kein explizites Subjekt.
Dies ist ein strukturelles Merkmal der Rechtssprache, kein Fehler.

**Flair-Anteil (21.5% statt > 40%):**
Das Flair-Modell wurde auf Gerichtsentscheidungen trainiert.
Verwaltungsspezifische Begriffe werden überwiegend regelbasiert erkannt.
Der niedrige Anteil ist für M3 ausreichend.

**UNKNOWN-Rate (16.0%):**
Die verbleibenden 44 UNKNOWN-SVOs sind überwiegend Kopula-Konstruktionen
ohne eindeutige Normaussage (ist, hat, sind) oder sehr kurze Fragmente.
Für M3 ist dieser Wert ausreichend.

### Entwicklung UNKNOWN-Rate über alle Iterationen

```
Start (nur spaCy Regex):     70.5%
+ Erweiterte Patterns:       53.1%   (-17.4%)
+ LLM Batch 1 (Test):        34.7%   (-18.4%)
+ LLM vollständig (2.5 Flash):26.7%  (- 8.0%)
+ Neue Patterns + 3.1 Lite:  16.0%   (-10.7%)
```

---

*Dieses Konzept beschreibt die Zielarchitektur und den Implementierungsstand
von M2 (NLP-Processing). Es richtet sich an Entwicklungsteams und
Facharchitekten im Bereich NLP, Public-Sector-Digitalisierung und
Knowledge-Graph-Engineering.*
