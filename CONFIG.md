# MNR / NDI – Konfigurationsreferenz

> Vollständige Übersicht aller Konfigurationsdateien, Startbefehle und
> häufigen Anpassungen des NDI-Systems (Meta-Normen-Register).
>
> Stand: Mai 2026 | Meilenstein M1 + M2
>
> **Alle YAML-Dateien werden zur Laufzeit frisch gelesen –
> ein Neustart ist nach Konfigurationsänderungen nicht erforderlich.**
> Ausnahme: `embedder_config.yaml` erfordert Neustart des NLP-Workers.

---

## Inhaltsverzeichnis

1. [System starten](#1-system-starten)
2. [Infrastruktur](#2-infrastruktur)
3. [Datenbank-Schema](#3-datenbank-schema)
4. [YAML-Konfigurationen](#4-yaml-konfigurationen)
5. [Programme & Skripte](#5-programme--skripte)
6. [Häufige Anpassungen](#6-häufige-anpassungen)
7. [Config-Manager](#7-config-manager)

---

## 1. System starten

### Voraussetzungen

```bash
cd ~/reg-mo/ndi

# Docker-Stack starten (PostgreSQL, MinIO, Apache Tika):
docker compose up -d

# Status prüfen:
docker ps
# Erwartet: mnr-postgres, mnr-minio, mnr-tika alle "Up"
```

### FastAPI-Server starten

```bash
cd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate

# Entwicklung (mit Auto-Reload bei Dateiänderungen):
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Produktion (mehrere Worker, kein Reload):
uvicorn main:app --workers 4 --host 127.0.0.1 --port 8000
```

Nach dem Start erreichbar unter:

| URL | Inhalt |
|---|---|
| `http://localhost:8000` | API-Root |
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/api/v1/config/ui` | Config-Manager |

### NLP-Worker starten

```bash
cd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate

# Alle Chunks verarbeiten:
python nlp_worker.py --run

# Nur ein Dokument:
python nlp_worker.py --run --doc-id <uuid>

# Status und Statistik:
python nlp_worker.py --jobs
python nlp_worker.py --stats
```

### Port-Weiterleitung (WSL2 → Windows)

Nach WSL-Neustart muss die Port-Weiterleitung neu gesetzt werden:

```powershell
# PowerShell als Administrator:
powershell -ExecutionPolicy Bypass -File C:\scripts\wsl-portforward.ps1
```

---

## 2. Infrastruktur

| Datei | Pfad | Beschreibung |
|---|---|---|
| `.env` | `NDI/.env` | Passwörter, Ports, Bucket-Namen. **Nicht im Git.** |
| `.env.example` | `NDI/.env.example` | Vorlage für `.env` ohne Secrets. Im Git. |
| `docker-compose.yml` | `NDI/docker-compose.yml` | Stack: postgres, minio, tika. |

### Variablen in `.env`

```bash
# PostgreSQL
POSTGRES_USER=mnr
POSTGRES_PASSWORD=<passwort>
POSTGRES_DB=mnr_db
POSTGRES_PORT=5432

# MinIO (Objektspeicher für Rohdateien und Artefakte)
MINIO_ROOT_USER=mnr_admin
MINIO_ROOT_PASSWORD=<passwort>
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_DEFAULT_BUCKET=mnr-artefakte

# Apache Tika (Dokument-Parser)
TIKA_PORT=9998

# Projekt
PROJECT_NAME=ndi
```

### Docker-Container

| Container | Image | Port | Funktion |
|---|---|---|---|
| `mnr-postgres` | `pgvector/pgvector:pg16` | 5432 | Datenbank + Vektorspeicher |
| `mnr-minio` | `minio/minio:latest` | 9000 / 9001 | Objektspeicher für Rohdateien |
| `mnr-tika` | `apache/tika:latest-full` | 9998 | Dokument-Parsing (PDF, DOCX, …) |

---

## 3. Datenbank-Schema

SQL-Dateien unter `infra/postgres/`. Initialisierungsskripte werden beim
ersten Container-Start automatisch eingespielt.

### Initialisierungs-Skripte

| Datei | Pfad | Tabellen |
|---|---|---|
| `01_schema.sql` | `infra/postgres/init/` | `norm_documents`, `norm_chunks`, `information_models`, `im_review_log` |
| `02_jobs.sql` | `infra/postgres/init/` | `ingest_jobs` |
| `03_pakete.sql` | `infra/postgres/init/` | `ingest_pakete`, `ingest_paket_jobs` |
| `04_nlp.sql` | `infra/postgres/init/` | `nlp_jobs`, `svo_extractions`, `ner_entities` |
| `05_abbrev.sql` | `infra/postgres/init/` | Spalten `content_original`, `abbrev_map` in `norm_chunks` |
| `06_questions.sql` | `infra/postgres/init/` | `filtered_questions` |

```bash
# Einzelnes Skript nachträglich einspielen:
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  < infra/postgres/init/05_abbrev.sql
```

### Maintenance-Skripte

| Datei | Wann |
|---|---|
| `maintenance/drop_all_tables.sql` | Kompletter Reset inkl. Struktur. Nur Entwicklung. |
| `maintenance/truncate_all_tables.sql` | Inhalte löschen, Struktur bleibt. Test-Reset. |

```bash
# Test-Reset (Inhalte löschen):
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  < infra/postgres/maintenance/truncate_all_tables.sql
```

---

## 4. YAML-Konfigurationen

Alle YAML-Dateien liegen in `services/ingest/`.

### Übersicht

| Datei | Funktion | Neustart nötig |
|---|---|---|
| `embedder_config.yaml` | Embedding-Modell, Batch-Size, Device | NLP-Worker |
| `abbrev_dict.yaml` | Abkürzungen und Synonyme | Nein |
| `nlp_config.yaml` | spaCy, SVO, NER, Fragen-Filter, Worker | Nein |
| `chunker_config.yaml` | Token-Limits, Chunk-Typ-Patterns | Nein |
| `docs.yaml` | Dokument-Metadaten, Typ-Erkennung | Nein |

---

### 4.1 embedder_config.yaml

Steuert das Embedding-Modell. Modellwechsel erfordert Neustart des NLP-Workers
und Neu-Einspielen aller Dokumente (Embeddings sind modellabhängig).

```yaml
active_model: "deepset-mxbai"      # deepset-mxbai | multilingual-e5

models:
  deepset-mxbai:
    model_id:       "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    dimensions:     1024
    prefix_query:   "query: "       # Präfix für Suchanfragen im RAG
    prefix_passage: "passage: "     # Präfix für Chunk-Texte beim Ingest

  multilingual-e5:
    model_id:       "intfloat/multilingual-e5-large"
    dimensions:     1024
    prefix_query:   "query: "
    prefix_passage: "passage: "

runtime:
  batch_size:     16        # Chunks pro Batch (CPU: 8–16, GPU: 32–64)
  device:         "auto"    # auto | cpu | cuda
  cache_enabled:  true
  cache_max_size: 1000      # Max. gecachte Embeddings im Speicher
```

---

### 4.2 abbrev_dict.yaml

Wörterbuch für Abkürzungsauflösung und Synonym-Normalisierung.
Wird zwischen Parser und Chunker angewendet – **vor** dem Chunking und Embedding.
Der Originaltext wird in `norm_chunks.content_original` gespeichert (Traceability).

#### Abkürzungen

```yaml
abbreviations:

  # Einfache Abkürzung – immer auflösen
  - abbrev:   "NHundG"
    resolved: "Niedersächsisches Gesetz über das Halten von Hunden"
    context:  []           # leer = immer auflösen
    label:    "GESETZ"

  # Kontextabhängig (AG = Amtsgericht ODER Aktiengesellschaft)
  - abbrev:   "AG"
    resolved: "Amtsgericht"
    context:  ["Urteil", "Beschluss", "Az.", "Aktenzeichen"]
    label:    "BEHÖRDE"
```

| Feld | Pflicht | Beschreibung |
|---|---|---|
| `abbrev` | ✅ | Abkürzung exakt wie im Dokument |
| `resolved` | ✅ | Vollständige Auflösung |
| `context` | ❌ | Signalwörter (±100 Zeichen). Leer = immer auflösen |
| `label` | ❌ | `GESETZ`, `BEHÖRDE`, `ROLLE`, `ORT`, `SONSTIGE` |

> Längere Abkürzungen werden zuerst geprüft.
> Normreferenzen (`§ 3 Abs. 1`) werden **niemals** aufgelöst.

#### Synonyme

Varianten werden auf eine Normalform abgebildet – verbessert die
Retrieval-Qualität weil gleiche Konzepte ähnliche Embedding-Vektoren bekommen.

```yaml
synonyms:

  - canonical: "Hundehalter"
    variants:
      - "Hundehalterin"
      - "Halterin oder Halter"
      - "haltende Person"
    context:  ["Hund", "Tier"]   # Kontext verhindert Fehlzuordnungen
    label:    "ROLLE"

  - canonical: "Sachkundenachweis"
    variants:
      - "Nachweis der Sachkunde"
      - "Sachkundeprüfung"
      - "Befähigungsnachweis"
    context:  []
    label:    "DATENOBJEKT"
```

---

### 4.3 nlp_config.yaml

Zentrale NLP-Konfiguration. Wirkt beim nächsten `python nlp_worker.py --run`.

#### spaCy

```yaml
spacy:
  model: "de_core_news_lg"   # sm (schnell) | md | lg (beste Qualität)
  batch_size: 32
```

#### SVO-Extraktion

```yaml
svo:
  min_confidence: 0.5
  pronoun_confidence_penalty: 0.3   # Abzug bei Pronomen als Subjekt

  subject_deps: [sb, sp, nsubj]
  object_deps:  [oa, og, op, oc, pd, obj, obl]

  stop_subjects: [er, sie, es, man, der, die, das, ...]
  stop_objects:  [der, die, das, dem, den, ...]
  pronouns:      [er, sie, es, man, wer, ...]
```

#### Normtypen

Regex-Patterns klassifizieren Normtexte nach Regeltyp.
Patterns **müssen** in einfachen Anführungszeichen stehen.

```yaml
normtypen:
  MUST:
    patterns: ['\bmuss\b', '\bmüssen\b', '\bist\s+zu\b', '\bhat\s+zu\b']
  MAY:
    patterns: ['\bkann\b', '\bdarf\b', '\bist\s+berechtigt\b']
  MUST_NOT:
    patterns: ['\bdarf\s+nicht\b', '\bist\s+untersagt\b']
  DEF:
    patterns: ['\bim\s+Sinne\b', '\bgilt\s+als\b']
  EXCEPT:
    patterns: ['\bes\s+sei\s+denn\b', '\bausgenommen\b']
  DEADLINE:
    patterns: ['\bbinnen\b', '\bspätestens\b', '\bunverzüglich\b']
  COMPETENCE:
    patterns: ['\bzuständig\s+ist\b', '\bobliegt\b']
```

#### NER

```yaml
ner:
  rules_enabled:        true
  flair_enabled:        true
  flair_model:          "flair/ner-german-legal"
  flair_min_confidence: 0.85        # Höher = weniger False Positives
  combination_strategy: "merge"     # merge | flair_only | rules_only

  # Ansatz 1: Entitäten die niemals gespeichert werden
  blacklist: [Google, GmbH, AG]

  # Ansatz 2: Label-Korrekturen statt Verwerfen
  label_corrections:
    "Telekom": "SONSTIGE"

  # Ansatz 4: Kontext-Validierung
  context_validation:
    enabled: true
    context_window: 80
    BEHÖRDE:
      required_context: [behörde, amt, ministerium, gericht, zuständig]
      fallback_label: "SONSTIGE"
    GESETZ:
      required_context: ['§', gesetz, verordnung]
      fallback_label: "SONSTIGE"
```

#### Fragen-Filter

```yaml
question_filter:

  ingest:                      # Wirkt beim Chunking
    enabled: true
    apply_to_classes: [B, C]
    action: "exclude"          # exclude | include | include_as_type

  nlp:                         # Wirkt beim NLP-Worker
    enabled: true
    apply_to_classes: [B, C]
    action: "skip"             # skip | process

  detection:
    min_length: 10
    max_length: 300
    require_question_mark: true
    question_words: [wer, was, wie, wo, wann, warum, welche, kann, darf, muss]
```

| `ingest` | `nlp` | Effekt |
|---|---|---|
| `true` | `true` | Fragen weder in Chunks noch in SVO/NER |
| `true` | `false` | Fragen nicht in Chunks, NLP verarbeitet sie |
| `false` | `true` | Fragen in Chunks, NLP überspringt sie |
| `false` | `false` | Fragen vollständig normal verarbeitet |

#### Worker

```yaml
worker:
  process_classes:    [A, B, C]
  chunk_batch_size:   50
  overwrite_existing: true      # Bestehende NLP-Ergebnisse überschreiben
  min_token_count:    10        # Chunks kürzer als X überspringen
  show_progress:      true
```

---

### 4.4 chunker_config.yaml

Steuert Token-Limits, Overlap-Ratios und Chunk-Typ-Erkennung.

#### Token-Limits

```yaml
chunking:
  class_a:                       # Normative Rechtstexte (§-Struktur)
    token_limit_parent: 1024     # §-Paragraph (Kontext-Anker)
    token_limit_child:  256      # Absatz
    token_limit_satz:   128      # Satz (bei sehr langen Absätzen)

  class_b:                       # Strukturierte Fachdokumente
    token_limit_parent: 128      # Kapitel-Titel + Einleitung
    token_limit_child:  512      # Unterkapitel
    overlap_ratio:      0.15     # 15% Überlapp zwischen Chunks

  class_c:                       # Unstrukturierte Texte
    token_min:     50
    token_max:     384
    overlap_ratio: 0.20          # 20% Sliding-Window
    score_threshold: 0.72        # Cosine-Grenze für Segmentierung
```

#### Chunk-Typ-Erkennung (Klasse A)

Erste zutreffende Regel gewinnt. Fallback: `tatbestand`.

```yaml
chunk_typen:
  MUST_NOT:
    chunk_type: rechtsfolge
    patterns: ['\bdarf\s+nicht\b', '\bist\s+untersagt\b']

  COMPETENCE:
    chunk_type: zustaendigkeit
    patterns: ['\bzuständig\s+ist\b', '\bobliegt\b']

  DEF:
    chunk_type: definition
    patterns: ['\bim\s+Sinne\b', '\bgilt\s+als\b']

  MUST:
    chunk_type: tatbestand
    patterns: ['\bmuss\b', '\bist\s+zu\b']
```

Verfügbare Chunk-Typen: `tatbestand`, `rechtsfolge`, `definition`,
`ausnahme`, `verweis`, `zustaendigkeit`, `deadline`, `anforderung`,
`tabelle`, `text`

---

### 4.5 docs.yaml

Zwei Funktionen: Dokument-Metadaten (Priorität 2) und Konfiguration
der automatischen Typ-Erkennung (Priorität 3).

#### Priorität für `source_type`

```
1. CLI/API-Parameter  →  --source-type gesetz
2. Eintrag in docs.yaml
3. Automatische Erkennung (DocTypeClassifier)
   └─ kein Typ erkannt  →  "text"
```

#### Dokument-Metadaten

```yaml
Hundegesetz.pdf:
  source_type:  gesetz
  title:        NHundG – Niedersächsisches Hundegesetz
  jurisdiction: NDS
  version:      "2020"
  force_class:  A          # optional: A | B | C erzwingen

FAQ1-NHundG.pdf:
  source_type: auslegung
  title:       FAQ Sachkundenachweis
```

#### Automatische Typ-Erkennung

```yaml
doc_type_detection:
  enabled: true
  global_min_confidence: 0.70

  typen:
    gesetz:
      min_confidence: 0.90
      signale:
        titel_muster:    ['(?i)\bgesetz\b', '(?i)[A-Z][a-z]+G$']
        volltext_signale: ['BGBl.', 'GVBl.']
        struktur_signale:
          paragraph_min: 3

    auslegung:
      min_confidence: 0.80
      signale:
        titel_muster:    ['(?i)\bfaq\b', '(?i)häufig.*fragen']
        volltext_signale: ['Frage:', 'Antwort:']

    text:
      min_confidence: 0.0    # Fallback – immer zugewiesen
```

Unterstützte Typen: `gesetz`, `verordnung`, `standard`, `fachkonzept`,
`leitfaden`, `lastenheft`, `auslegung`, `text`

---

## 5. Programme & Skripte

Alle Skripte werden aus `services/ingest/` aufgerufen:

```bash
cd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate
```

### Ingest

```bash
# Einzelnes Dokument:
python ingest_cli.py --pdf ~/docs/Hundegesetz.pdf \
  --source-type gesetz --title "NHundG"

# Dokumentklasse erzwingen:
python ingest_cli.py --pdf ~/docs/FAQ.pdf \
  --source-type auslegung --class B

# Status:
python ingest_cli.py --jobs
python ingest_cli.py --status <job_id>
```

### NLP-Worker

```bash
python nlp_worker.py --run                       # alle Chunks
python nlp_worker.py --run --doc-id <uuid>       # ein Dokument
python nlp_worker.py --run --no-overwrite        # ohne Überschreiben
python nlp_worker.py --jobs                      # laufende Jobs
python nlp_worker.py --stats                     # Statistik
```

### Reports

```bash
python ingest_report.py                          # Ingest-Qualität
python ingest_report.py --doc "NHundG"
python ingest_report.py --export --out out.csv

python nlp_quality_report.py                     # NLP-Qualität (7 Abschnitte)
python nlp_quality_report.py --doc "NHundG"
python nlp_quality_report.py --export

python question_report.py                        # Gefilterte Fragen
python question_report.py --all                  # inkl. reviewte
python question_report.py --decide <id> verwerfen --note "FAQ-Frage"
python question_report.py --export

python nlp_monitor.py                            # NLP-Fortschritt live
```

### Vollständiger Testlauf

```bash
cd ~/reg-mo/ndi/Test
source ../services/ingest/.venv/bin/activate

python run_test.py                               # Reset → Ingest → NLP → Log
python run_test.py --skip-reset                 # nur Ingest + NLP
python run_test.py --docs-dir /anderer/pfad
python run_test.py --timeout 600

# Metadaten für Testdokumente: Test/docs/docs.yaml
# Logfiles:                    Test/MMTTHHMMSS.log
```

---

## 6. Häufige Anpassungen

### Embedding-Modell wechseln

```yaml
# embedder_config.yaml
active_model: "multilingual-e5"
```

Danach NLP-Worker neu starten und Dokumente neu einspielen.

### Neue Abkürzung

```yaml
# abbrev_dict.yaml → abbreviations
- abbrev: "MeldeG BW"
  resolved: "Meldegesetz Baden-Württemberg"
  context: []
  label: "GESETZ"
```

### Neues Synonym

```yaml
# abbrev_dict.yaml → synonyms
- canonical: "Meldebehörde"
  variants: ["zuständige Behörde", "Einwohnermeldeamt"]
  context: ["Meldung"]
  label: "BEHÖRDE"
```

### Normtyp-Pattern ergänzen

```yaml
# nlp_config.yaml → normtypen.MUST.patterns
- '\bist\s+anzuwenden\b'
- '\bfindet\s+Anwendung\b'
```

### NER False Positive ausschließen

```yaml
# nlp_config.yaml → ner.blacklist
- "NeuerFalsePositive"

# Oder Label korrigieren:
# nlp_config.yaml → ner.label_corrections
"Falsches Label": "SONSTIGE"
```

### Fragen-Filter deaktivieren

```yaml
# nlp_config.yaml
question_filter:
  ingest:
    enabled: false
  nlp:
    enabled: false
```

### Neuen Chunk-Typ-Pattern ergänzen

```yaml
# chunker_config.yaml → chunk_typen
MEIN_TYP:
  chunk_type: tatbestand
  patterns:
    - '\bmein\s+pattern\b'
```

### Dokumentklasse erzwingen

```yaml
# docs.yaml
MeinDokument.pdf:
  source_type: fachkonzept
  force_class: B
```

### Min-Konfidenz für Typ-Erkennung anpassen

```yaml
# docs.yaml → doc_type_detection.typen
gesetz:
  min_confidence: 0.85    # senken = mehr Treffer, weniger Präzision
```

---

## 7. Config-Manager

Browser-basierte Verwaltungsoberfläche für alle Konfigurationsdateien.

**Voraussetzung:** FastAPI läuft (→ Abschnitt 1)

```
http://localhost:8000/api/v1/config/ui
```

| Menüpunkt | Inhalt |
|---|---|
| 🏠 Home | Startseite mit Übersicht |
| 📋 Config-Übersicht | Diese Datei (CONFIG.md, nur Anzeige) |
| 📊 Report & Monitor | Fragen, Ingest, NLP-Qualität, NLP-Monitor |
| 🧠 NLP-Konfiguration | spaCy, SVO, Normtypen, NER, Fragen-Filter, Worker |
| 🔢 Embedding-Modell | Modell-Auswahl, Batch-Size, Device |
| 📖 Wörterbücher | Abkürzungen und Synonyme (Karten-Layout) |
| ✂️ Chunker-Konfiguration | Token-Limits und Chunk-Typ-Patterns |
| 🗂️ Dokument-Metadaten | Dokumenttypen, Titel, Typ-Erkennung |

Vor jedem Speichern wird automatisch ein Backup als `.yaml.bak` angelegt.
Der **↩ Backup**-Button stellt das letzte Backup wieder her.

---

*Alle YAML-Konfigurationen werden zur Laufzeit frisch gelesen.
Ein Neustart des Services ist nicht erforderlich.*
*Ausnahme: `embedder_config.yaml` → NLP-Worker neu starten.*
