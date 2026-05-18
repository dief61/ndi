# M1 – Ingest-Pipeline: Konzept und Implementierung

> Technisches Konzept der Dokumenten-Ingest-Pipeline
> des Meta-Normen-Registers (MNR).
>
> Stand: Mai 2026 | Meilenstein M1 – abgeschlossen
>
> Dieses Dokument beschreibt Aufbau, Datenstrukturen und
> Implementierungsentscheidungen der Ingest-Pipeline vom
> Rohdokument bis zum eingebetteten, gespeicherten Chunk.

---

## Inhaltsverzeichnis

1. [Kontext und Einordnung](#1-kontext-und-einordnung)
2. [Gesamtübersicht](#2-gesamtübersicht)
3. [Dokumenten-Empfang und Parsing](#3-dokumenten-empfang-und-parsing)
4. [Dokumentklassen-Erkennung](#4-dokumentklassen-erkennung)
5. [Chunking-Strategien](#5-chunking-strategien)
6. [Abkürzungsauflösung und Synonym-Normalisierung](#6-abkürzungsauflösung-und-synonym-normalisierung)
7. [Embedding](#7-embedding)
8. [Speicherung](#8-speicherung)
9. [Konfiguration](#9-konfiguration)
10. [Datenbank-Schema](#10-datenbank-schema)
11. [API-Endpunkte](#11-api-endpunkte)
12. [Qualitätsmetriken M1](#12-qualitätsmetriken-m1)

---

## 1. Kontext und Einordnung

### Einbettung in die MNR-Architektur

M1 bildet die Datenbasis für alle nachfolgenden Meilensteine.
Ohne korrekt gechunkte, eingebettete und gespeicherte Dokumente
ist weder NLP (M2) noch RAG (M3) möglich.

```
Eingabe                    M1 – Ingest-Pipeline               Ausgabe
────────────────────────   ────────────────────────────────   ──────────────────────
PDF, DOCX, HTML, RTF   →   Parsing → Klassen-Erkennung    →   norm_documents
Gesetz, Verordnung     →   Abkürzungen → Chunking          →   norm_chunks (A/B/C)
Standard, FAQ          →   Embedding → Speicherung         →   Embeddings (pgvector)
                                                           →   MinIO (Rohdateien)
```

### Fachagnostik als Designprinzip

Die Pipeline verarbeitet **jedes** normative Dokument ohne
register-spezifische Anpassungen im Code. Fachliche Konfiguration
erfolgt ausschließlich über YAML-Dateien:

- `docs.yaml` – Dokument-Metadaten und Typ-Erkennung
- `chunker_config.yaml` – Token-Limits und Chunk-Typen
- `abbrev_dict.yaml` – Abkürzungen und Synonyme

---

## 2. Gesamtübersicht

```
Dokument (PDF/DOCX/HTML/RTF)
  │
  ▼
3. Parsing (Apache Tika + Docling)
   Text, Struktur, Tabellen, Metadaten extrahieren
   Textnormierung (Zeichensatz, Bindestriche, PUA-Zeichen)
  │
  ▼
4. Dokumentklassen-Erkennung (Chunking-Router)
   Kaskade: §-Marker → Heading-Hierarchie → Fallback
   Ergebnis: doc_class A | B | C
  │
  ▼
6. Abkürzungsauflösung & Synonym-Normalisierung
   abbrev_dict.yaml → normalisierter Text
   Originaltext wird in content_original gesichert
  │
  ▼
5. Chunking (klassen-spezifisch)
   A: §-hierarchisches Chunking (Parent → Child → Satz)
   B: Kapitel-basiertes Chunking (ToC-Struktur)
   C: Semantisches Sliding-Window
  │
  ▼
7. Embedding (sentence-transformers)
   deepset-mxbai-embed-de-large-v1
   Vektor-Dimensionen: 1024
  │
  ▼
8. Speicherung
   PostgreSQL (norm_documents, norm_chunks, Embeddings via pgvector)
   MinIO (Rohdateien, Artefakte)
```

---

## 3. Dokumenten-Empfang und Parsing

### Eingabeformate

```
Unterstützt: PDF, HTML, DOC, DOCX, RTF
Primär:      PDF (Gesetzestexte als PDF-Publikationen)
```

### Parsing-Stack

| Schritt | Aufgabe | Technologie |
|---|---|---|
| Upload + Validierung | Dateiformat, Größe, Virenscan-Hook | FastAPI + MinIO |
| Text-Extraktion | Multi-Format, Tabellen, Struktur | Apache Tika 2.x |
| Struktur-Erkennung | §-Paragraphen, Überschriften, Tabellen | Docling |
| Textnormierung | Zeichensatz, Bindestriche, Sonderzeichen | Custom Python |
| Metadaten | Normtyp, Datum, Jurisdiktion, Version | Regelbasiert + LLM |

### Textnormierung im Detail

Die Normierung entfernt Parsing-Artefakte die bei PDF-Extraktion
typisch auftreten:

```python
def clean_text(text: str) -> str:
    # Bulletpoint-Artefakte: ▪ • · → Leerzeichen
    text = re.sub(r'[▪•·]', ' ', text)

    # PUA-Zeichen (Private Use Area) entfernen
    # Häufig bei PDFs: U+EEA4, U+EEB1 etc.
    text = re.sub(r'[\uE000-\uF8FF]', '', text)

    # Typografische Anführungszeichen normalisieren
    text = text.replace('„', '"').replace('"', '"')
    text = text.replace('‚', "'").replace('\u2018', "'")

    # Mehrzeilige Paragraphen zusammenführen
    # (Zeilenumbrüche innerhalb eines §-Textes)
    text = re.sub(r'(?<=[^.\n])\n(?=[a-z])', ' ', text)

    # Mehrfache Leerzeichen
    text = re.sub(r'  +', ' ', text)

    return text.strip()
```

### Metadaten-Schema je Dokument

```json
{
  "doc_id":         "uuid",
  "source_type":    "gesetz|verordnung|standard|fachkonzept|leitfaden|lastenheft|auslegung",
  "title":          "NHundG – Niedersächsisches Hundegesetz",
  "jurisdiction":   "NDS",
  "valid_from":     "2020-01-01",
  "valid_to":       null,
  "version":        "2020",
  "language":       "de",
  "register_scope": ["hundehaltung"],
  "approved_by":    null,
  "ingest_ts":      "2026-05-15T10:00:00Z"
}
```

---

## 4. Dokumentklassen-Erkennung

### Die drei Dokumentklassen

| Klasse | Dokumenttyp | Strukturprinzip | Beispiele |
|---|---|---|---|
| **A** | Normative Rechtstexte | §-Paragraphen | Gesetze, Verordnungen |
| **B** | Strukturierte Fachdokumente | Kapitel/ToC | Standards, Fachkonzepte, Lastenhefte |
| **C** | Unstrukturierte Ergänzungen | Kein stabiles Gerüst | FAQs, Handreichungen, Auslegungshinweise |

### Erkennungskaskade (Chunking-Router)

```python
def detect_doc_class(text: str, structure: dict) -> str:
    """
    Kaskade: Erste zutreffende Bedingung gewinnt.
    """

    # Klasse A: §-Marker vorhanden
    paragraph_count = len(re.findall(r'^\s*§\s*\d+', text, re.MULTILINE))
    if paragraph_count >= 3:
        return "A"

    # Klasse B: Heading-Hierarchie erkennbar
    # H1/H2/H3 oder nummerierte Überschriften (1.1.2)
    heading_count = len(re.findall(
        r'^#{1,3}\s|^\d+\.\d+(\.\d+)?\s', text, re.MULTILINE))
    if heading_count >= 3:
        return "B"

    # Klasse C: Fallback
    return "C"
```

**Wichtig:** Die erkannte Klasse wird als `doc_class`-Feld
persistent im Chunk-Metadatum gespeichert und steuert alle
nachgelagerten Prozesse (Chunking, NLP-Retrieval, Context Assembly).

### Manuelle Überschreibung

In `docs.yaml` kann die Klasse pro Dokument erzwungen werden:

```yaml
Hundegesetz.pdf:
  source_type: gesetz
  force_class: A    # Automatik überschreiben
```

---

## 5. Chunking-Strategien

### Klasse A – §-Hierarchisches Chunking

**Strukturprinzip:** §-Paragraphen sind die primäre Sinneinheit.
Die Granularität folgt der Rechtshierarchie:
Paragraph → Absatz → Satz.

```
Parent-Chunk: §-Paragraph (max. 1.024 Token)
  ├─ Child-Chunk: Absatz 1 (max. 256 Token)
  │    chunk_type: tatbestand | rechtsfolge | definition | ausnahme | zustaendigkeit
  ├─ Child-Chunk: Absatz 2 (max. 256 Token)
  └─ Satz-Chunk: nur bei sehr langen Absätzen (max. 128 Token)

Legaldefinitionen: eigener Chunk unabhängig von Position
  chunk_type: definition

Verweise (§ X gilt entsprechend):
  chunk_type: verweis
  cross_references: [uuid-ref-zum-anderen-para]
```

**Token-Limits Klasse A:**

| Ebene | Max. Token | Overlap | Besonderheit |
|---|---|---|---|
| Parent (§-Paragraph) | 1.024 | keiner | Kontext-Anker |
| Child (Absatz) | 256 | keiner | chunk_type klassifiziert |
| Satz | 128 | keiner | Nur bei langen Absätzen |
| Legaldefinition | 256 | keiner | Immer eigener Chunk |
| Verweis | 128 | keiner | cross_references gesetzt |

### Klasse B – Kapitel-Basiertes Chunking

**Strukturprinzip:** Das Inhaltsverzeichnis (ToC) ist die
Strukturreferenz. Kapitel und Unterkapitel sind die primären
Sinneinheiten.

```
Parent-Chunk: Kapitel-Titel + Einleitung (max. 128 Token)
  ├─ Child-Chunk: Unterkapitel (max. 512 Token, 15% Overlap zum Vorgänger)
  │    heading_breadcrumb: "Datenanf. > Opt. Felder > Datum"
  ├─ Tabellen-Chunk: variabel, JSON-kodiert
  │    Rückreferenz auf beschreibenden Abschnitts-Chunk
  └─ Anforderungs-Mini-Chunk: max. 128 Token
       requirement_id: "A-001" (bei Lastenheften)
```

**Token-Limits Klasse B:**

| Ebene | Max. Token | Overlap | Besonderheit |
|---|---|---|---|
| Parent (Kapitel) | 128 | keiner | Titel + Einleitungssatz |
| Child (Unterkapitel) | 512 | 15% | heading_breadcrumb im Metadatum |
| Tabellen-Chunk | variabel | keiner | JSON-kodiert |
| Anforderungs-Chunk | 128 | keiner | requirement_id als Metadatum |

### Klasse C – Semantisches Sliding-Window

**Strukturprinzip:** Kein stabiles Strukturgerüst. Fließtext
wird anhand semantischer Ähnlichkeit segmentiert.

```
Segment-Grenze: Cosine-Similarity < 0.65
Token-Bereich:  min. 3 Sätze / max. 384 Token
Overlap:        20% Sliding-Window
```

**Qualitätskennzeichen:** Klasse-C-Chunks erhalten `confidence_weight: 0.65`
(niedrigste Priorität). Im RAG-Retrieval wird ein erhöhter Score-Threshold
(≥ 0.72 statt ≥ 0.60) angewendet, damit Klasse-C-Chunks nicht
präzisere Klasse-A-Chunks verdrängen.

### Gemeinsames Chunk-Metadatum

```json
{
  "chunk_id":           "uuid",
  "doc_id":             "uuid-ref",
  "doc_class":          "A|B|C",
  "source_type":        "gesetz|verordnung|...",
  "norm_reference":     "§ 3 Abs. 1 NHundG",
  "section_path":       "3.2.1",
  "heading_breadcrumb": "Datenanf. > Opt. Felder",
  "requirement_id":     "A-001",
  "chunk_type":         "tatbestand|rechtsfolge|definition|ausnahme|anforderung|tabelle|verweis",
  "hierarchy_level":    2,
  "parent_chunk_id":    "uuid-ref",
  "cross_references":   ["uuid-ref-2"],
  "overlap_with_prev":  0.15,
  "confidence_weight":  1.0,
  "token_count":        128,
  "version":            "2020",
  "valid_from":         "2020-01-01",
  "valid_to":           null,
  "im_signals":         null
}
```

---

## 6. Abkürzungsauflösung und Synonym-Normalisierung

Die Normalisierung erfolgt **vor** dem Chunking und Embedding.
Der Originaltext wird in `content_original` gesichert (Traceability).

### Abkürzungsauflösung

```yaml
# abbrev_dict.yaml – Beispiele
abbreviations:

  # Einfach: immer auflösen
  - abbrev:   "NHundG"
    resolved: "Niedersächsisches Gesetz über das Halten von Hunden"
    context:  []          # leer = immer auflösen
    label:    "GESETZ"

  # Kontextabhängig: nur in bestimmtem Kontext auflösen
  - abbrev:   "AG"
    resolved: "Amtsgericht"
    context:  ["Urteil", "Beschluss", "Az."]
    label:    "BEHÖRDE"
```

**Priorisierung:** Längere Abkürzungen werden zuerst geprüft.
Normreferenzen (`§ 3 Abs. 1`) werden **niemals** aufgelöst.

### Synonym-Normalisierung

Varianten werden auf eine Normalform abgebildet – verbessert die
Retrieval-Qualität, weil gleiche Konzepte ähnliche Embedding-Vektoren
bekommen:

```yaml
synonyms:
  - canonical: "Hundehalter"
    variants:
      - "Hundehalterin"
      - "Halterin oder Halter"
      - "Halterinnen und Halter"
    context:  ["Hund", "Tier"]
    label:    "ROLLE"

  - canonical: "Sachkundenachweis"
    variants:
      - "Sachkundeprüfung"
      - "erforderliche Sachkunde"
      - "Nachweis der Sachkunde"
    context:  []
    label:    "DATENOBJEKT"
```

### Qualitätsergebnis M1

```
Abkürzungs-Abdeckung: 100% (94/94 Chunks)
Aufgelöste Abkürzungen: 22 verschiedene
Häufigste Auflösung: Hundehalterin → Hundehalter (1.048 Ersetzungen)
```

### Fragen-Filter

FAQ-Dokumente (Klasse B/C) enthalten Fragen die für das NLP
ungeeignet sind. Der Fragen-Filter erkennt und sichert sie:

```yaml
question_filter:
  ingest:
    enabled: true
    apply_to_classes: [B, C]
    action: "exclude"     # aus Chunks ausschließen
  detection:
    min_length: 10
    max_length: 300
    require_question_mark: true
    question_words: [wer, was, wie, wo, wann, warum, welche, kann, darf, muss]
```

Gefilterte Fragen werden in `filtered_questions` gespeichert und
können nachträglich über den Config-Manager reviewt werden.

---

## 7. Embedding

### Modell

```
Aktives Modell:  mixedbread-ai/deepset-mxbai-embed-de-large-v1
Dimensionen:     1.024
Lizenz:          Apache 2.0
Optimiert für:   Deutsche Texte, semantische Suche (Retrieval)

Präfix Query:    "query: "      (bei Suchanfragen in M3)
Präfix Passage:  "passage: "   (bei Chunk-Texten beim Ingest)
```

**Warum dieser Präfix-Ansatz:** Das Modell ist für Asymmetric
Semantic Search trainiert – Query und Passage liegen in
unterschiedlichen Embedding-Räumen. Die Präfixe signalisieren
dem Modell welcher Modus gilt.

### Konfiguration

```yaml
# embedder_config.yaml
active_model: "deepset-mxbai"

models:
  deepset-mxbai:
    model_id:       "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    dimensions:     1024
    prefix_query:   "query: "
    prefix_passage: "passage: "

runtime:
  batch_size:     16       # CPU: 8-16, GPU: 32-64
  device:         "auto"   # auto | cpu | cuda
  cache_enabled:  true
  cache_max_size: 1000
```

### Embedding-Qualität

```
Embedding-Quote:   100% (94/94 Chunks)
Token min/max:     5 / 1.067
Token Durchschnitt: 150
Token Median:       119
Sehr kurze Chunks (<10 Zeichen): 0
Klasse A ohne norm_reference:    0
```

---

## 8. Speicherung

### PostgreSQL – Strukturierte Daten

```sql
-- Haupt-Schema (vereinfacht)
norm_documents (
  id UUID PRIMARY KEY,
  filename TEXT,
  source_type TEXT,
  title TEXT,
  doc_class CHAR(1),
  jurisdiction TEXT,
  valid_from DATE,
  valid_to DATE,
  metadata JSONB,
  created_at TIMESTAMPTZ
)

norm_chunks (
  id UUID PRIMARY KEY,
  doc_id UUID REFERENCES norm_documents(id),
  doc_class CHAR(1) CHECK (doc_class IN ('A','B','C')),
  norm_reference TEXT,
  cross_references UUID[],
  section_path TEXT,
  heading_breadcrumb TEXT,
  requirement_id TEXT,
  chunk_type TEXT,
  hierarchy_level INT,
  parent_id UUID REFERENCES norm_chunks(id),
  confidence_weight NUMERIC(3,2) DEFAULT 1.0,
  content TEXT NOT NULL,
  content_original TEXT,          -- vor Abkürzungsauflösung
  content_hash TEXT,              -- md5(content) für Duplikat-Filter
  abbrev_map JSONB,               -- welche Abkürzungen wurden aufgelöst
  im_signals JSONB,               -- IM-relevante Signale (M3/M4)
  embedding VECTOR(1024),         -- pgvector
  token_count INT,
  valid_from DATE,
  valid_to DATE,
  created_at TIMESTAMPTZ
)
```

### pgvector-Indizes

```sql
-- Vektorsuche (IVFFlat – guter Kompromiss Speed/Qualität)
CREATE INDEX ON norm_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Metadaten-Filter (häufig kombiniert mit Vektorsuche)
CREATE INDEX ON norm_chunks USING gin (metadata);
CREATE INDEX ON norm_chunks (doc_id, doc_class);
CREATE INDEX ON norm_chunks (norm_reference) WHERE norm_reference IS NOT NULL;
CREATE INDEX ON norm_chunks (requirement_id) WHERE requirement_id IS NOT NULL;

-- Cross-References
CREATE INDEX ON norm_chunks USING gin (cross_references)
  WHERE cross_references IS NOT NULL;

-- IM-Signale (M3)
CREATE INDEX ON norm_chunks USING gin (im_signals)
  WHERE im_signals IS NOT NULL;
```

### MinIO – Objektspeicher

```
Bucket: mnr-artefakte
  ├── raw/                    # Original-Rohdateien
  │   └── {doc_id}/document.pdf
  ├── parsed/                 # Extrahierter Text (Tika-Output)
  │   └── {doc_id}/text.txt
  └── artefakte/              # Generierte Artefakte (M5)
      └── {im_id}/schema.sql
```

---

## 9. Konfiguration

### docs.yaml – Dokument-Metadaten

```yaml
# Priorität für source_type:
# 1. CLI/API-Parameter
# 2. Eintrag in docs.yaml
# 3. Automatische Erkennung (DocTypeClassifier)
# 4. Fallback: "text"

Hundegesetz.pdf:
  source_type:  gesetz
  title:        NHundG – Niedersächsisches Hundegesetz
  jurisdiction: NDS
  version:      "2020"
  force_class:  A        # Klasse erzwingen (optional)
  ner_extensions:        # Register-spezifische NER (M2/M3)
    DATENOBJEKT:
      - "Hundehaltung"
      - "Wesenstest"
    ROLLE:
      - "Hundehalter"
```

### chunker_config.yaml – Token-Limits

```yaml
chunking:
  class_a:
    token_limit_parent: 1024
    token_limit_child:  256
    token_limit_satz:   128

  class_b:
    token_limit_parent: 128
    token_limit_child:  512
    overlap_ratio:      0.15

  class_c:
    token_min:        50
    token_max:        384
    overlap_ratio:    0.20
    score_threshold:  0.72    # Cosine-Grenze für Segmentierung
```

---

## 10. Datenbank-Schema

### Initialisierungs-Skripte

| Datei | Inhalt |
|---|---|
| `01_schema.sql` | `norm_documents`, `norm_chunks`, `information_models`, `im_review_log` |
| `02_jobs.sql` | `ingest_jobs` |
| `03_pakete.sql` | `ingest_pakete`, `ingest_paket_jobs` |
| `05_abbrev.sql` | Spalten `content_original`, `abbrev_map` |
| `06_questions.sql` | `filtered_questions` |
| `09_im_signals.sql` | Spalte `im_signals JSONB` + GIN-Index |

### Ingest-Job-Tracking

```sql
ingest_jobs (
  id UUID PRIMARY KEY,
  filename TEXT,
  status TEXT CHECK (status IN ('pending','running','done','error')),
  doc_class CHAR(1),
  chunk_count INT,
  error_message TEXT,
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
)
```

---

## 11. API-Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| `POST` | `/api/v1/ingest/document` | Dokument einspielen (Upload + async Pipeline) |
| `GET` | `/api/v1/ingest/jobs` | Alle Ingest-Jobs auflisten |
| `GET` | `/api/v1/ingest/jobs/{id}` | Status eines Jobs |
| `POST` | `/api/v1/ingest/paket` | Mehrere Dokumente als Paket |
| `GET` | `/api/v1/reports/ingest` | Ingest-Qualitätsbericht |

### Ingest-Aufruf

```bash
# Einzeldokument via CLI:
python ingest_cli.py --pdf ~/docs/Hundegesetz.pdf \
  --source-type gesetz --title "NHundG"

# Via API:
curl -X POST http://localhost:8000/api/v1/ingest/document \
  -F "file=@Hundegesetz.pdf" \
  -F "source_type=gesetz" \
  -F "title=NHundG"

# Status prüfen:
python ingest_cli.py --status <job_id>
```

---

## 12. Qualitätsmetriken M1

### Erreichte Ergebnisse (NHundG-Testlauf)

```
Dokumente:           4 (1× Klasse A, 3× Klasse B)
Chunks gesamt:       94
Mit Embedding:       94 / 94 (100%)
Embedding-Quote:     100%

Token min/max:       5 / 1.067
Token Durchschnitt:  150
Token Median:        119
Kurze Chunks (<10):  0
Klasse A ohne Normref: 0

Abkürzungs-Abdeckung: 100% (94/94 Chunks)
Aufgelöste Abkürzungen: 22 verschiedene
```

### Chunk-Typ-Verteilung

```
Klasse A:
  tatbestand    79 Chunks  ø 135 Token  Konfidenz: 1.00
  rechtsfolge    1 Chunk   ø  92 Token  Konfidenz: 1.00

Klasse B:
  tatbestand    11 Chunks  ø 275 Token  Konfidenz: 0.85
  anforderung    3 Chunks  ø 105 Token  Konfidenz: 0.85
```

### Hierarchie-Verteilung

```
Level 1 (Parent):  25 Chunks  (26.6%)  ø 238 Token
Level 2 (Child):   56 Chunks  (59.6%)  ø 120 Token
Level 3 (Satz):    13 Chunks  (13.8%)  ø 113 Token
```

### Zielwerte für produktiven Betrieb

| Kennzahl | Zielwert | Erreicht |
|---|---|---|
| Embedding-Quote | 100% | 100% ✅ |
| Kurze Chunks (<10 Zeichen) | 0 | 0 ✅ |
| Klasse A ohne norm_reference | 0 | 0 ✅ |
| Ingest-Fehlerrate | < 1% | 0% ✅ |
| Max. Ingest-Dauer je Dok. | < 300s | ~138s ✅ |

---

*Dieses Konzept beschreibt die Zielarchitektur und den Implementierungsstand
von M1 (Ingest-Pipeline). Es richtet sich an Entwicklungsteams und
Facharchitekten im Bereich Public-Sector-Digitalisierung.*
