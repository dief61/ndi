# MNR / NDI – Konfigurationsreferenz

> Vollständige Übersicht aller Konfigurationsdateien, Startbefehle und
> häufigen Anpassungen des NDI-Systems (Meta-Normen-Register).
>
> Stand: Mai 2026 | Meilenstein M1 + M2 (abgeschlossen)
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
5. [LLM-Gateway & Prompt-Suite](#5-llm-gateway--prompt-suite)
6. [Programme & Skripte](#6-programme--skripte)
7. [Häufige Anpassungen](#7-häufige-anpassungen)
8. [Config-Manager](#8-config-manager)

---

## 1. System starten

### Docker-Stack

```bash
cd ~/reg-mo/ndi
docker compose up -d

# Status prüfen:
docker ps
# Erwartet: mnr-postgres, mnr-minio, mnr-tika, mnr-fuseki alle "Up"
```

### FastAPI-Server

```bash
cd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate

# Entwicklung:
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Produktion:
uvicorn main:app --workers 4 --host 127.0.0.1 --port 8000
```

| URL | Inhalt |
|---|---|
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/api/v1/config/ui` | Config-Manager |

### NLP-Worker

```bash
python nlp_worker.py --run          # alle Chunks + LLM-Klassifikation
python nlp_worker.py --run --doc-id <uuid>
python nlp_worker.py --jobs
python nlp_worker.py --stats
```

### NLP neu ausführen (nach Konfigurationsänderungen)

```bash
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  -c "TRUNCATE TABLE svo_extractions, ner_entities, nlp_jobs CASCADE;"
python nlp_worker.py --run
python nlp_quality_report.py
```

### Knowledge-Graph exportieren

```bash
python kg_worker.py --export
python kg_worker.py --stats
```

### Port-Weiterleitung (WSL2 → Windows)

```powershell
powershell -ExecutionPolicy Bypass -File C:\scripts\wsl-portforward.ps1
```

---

## 2. Infrastruktur

| Datei | Pfad | Beschreibung |
|---|---|---|
| `.env` | `NDI/.env` | Passwörter, Ports, API-Keys. **Nicht im Git.** |
| `.env.example` | `NDI/.env.example` | Vorlage ohne Secrets. Im Git. |
| `docker-compose.yml` | `NDI/docker-compose.yml` | Stack: postgres, minio, tika, fuseki. |

### Variablen in `.env`

```bash
# PostgreSQL
POSTGRES_USER=mnr
POSTGRES_PASSWORD=<passwort>
POSTGRES_DB=mnr_db
POSTGRES_PORT=5432

# MinIO
MINIO_ROOT_USER=mnr_admin
MINIO_ROOT_PASSWORD=<passwort>
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_DEFAULT_BUCKET=mnr-artefakte

# Apache Tika
TIKA_PORT=9998

# Apache Jena Fuseki
FUSEKI_PORT=3030
FUSEKI_ADMIN_PASSWORD=<passwort>

# LLM-API-Keys (ohne Anführungszeichen!)
GEMINI_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-...        # optional
ANTHROPIC_API_KEY=sk-ant-... # optional

PROJECT_NAME=ndi
```

### Docker-Container

| Container | Image | Port | Funktion |
|---|---|---|---|
| `mnr-postgres` | `pgvector/pgvector:pg16` | 5432 | Datenbank + Vektorspeicher |
| `mnr-minio` | `minio/minio:latest` | 9000 / 9001 | Objektspeicher |
| `mnr-tika` | `apache/tika:latest-full` | 9998 | Dokument-Parsing |
| `mnr-fuseki` | `secoresearch/fuseki:latest` | 3030 | Knowledge Graph (RDF/SPARQL) |

---

## 3. Datenbank-Schema

### Initialisierungs-Skripte (`infra/postgres/init/`)

| Datei | Tabellen |
|---|---|
| `01_schema.sql` | `norm_documents`, `norm_chunks`, `information_models`, `im_review_log` |
| `02_jobs.sql` | `ingest_jobs` |
| `03_pakete.sql` | `ingest_pakete`, `ingest_paket_jobs` |
| `04_nlp.sql` | `nlp_jobs`, `svo_extractions`, `ner_entities` |
| `05_abbrev.sql` | Spalten `content_original`, `abbrev_map` in `norm_chunks` |
| `06_questions.sql` | `filtered_questions` |
| `07_knowledge_graph.sql` | `kg_sync_log`, View `kg_sync_stats` |
| `08_normtype_constraint.sql` | CHECK-Constraint für alle Normtypen |

```bash
# Einzelnes Skript nachträglich einspielen:
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  < infra/postgres/init/08_normtype_constraint.sql
```

### Erlaubte Normtypen

`MUST` · `MAY` · `MUST_NOT` · `DEF` · `EXCEPT` · `DEADLINE` ·
`COMPETENCE` · `SCOPE` · `CHANGE` · `STATUS` · `UNKNOWN`

### Maintenance

```bash
# NLP-Reset:
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  -c "TRUNCATE TABLE svo_extractions, ner_entities, nlp_jobs CASCADE;"

# Vollständiger Test-Reset:
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  < infra/postgres/maintenance/truncate_all_tables.sql
```

---

## 4. YAML-Konfigurationen

Alle YAML-Dateien liegen in `services/ingest/`.

| Datei | Funktion | Neustart nötig |
|---|---|---|
| `nlp_config.yaml` | spaCy, SVO, NER, Normtypen, LLM-Normtyp, Worker | Nein |
| `embedder_config.yaml` | Embedding-Modell, Batch-Size, Device | NLP-Worker |
| `abbrev_dict.yaml` | Abkürzungen und Synonyme | Nein |
| `chunker_config.yaml` | Token-Limits, Chunk-Typ-Patterns | Nein |
| `docs.yaml` | Dokument-Metadaten, Typ-Erkennung | Nein |
| `llm_gateway_config.yaml` | LLM-Provider, Modell, API-Keys, Retry | Nein |
| `prompt_suite_index.yaml` | Übersicht aller Prompts | Nein |

---

### 4.1 nlp_config.yaml

#### spaCy

```yaml
spacy:
  model: "de_core_news_lg"   # sm | md | lg
  batch_size: 32
```

#### SVO-Extraktion

```yaml
svo:
  min_confidence: 0.5
  pronoun_confidence_penalty: 0.3

  subject_deps: [sb, sp, nsubj]
  object_deps:  [oa, og, op, oc, pd, obj, obl, da, mo, nk, attr]

  stop_subjects:   [er, sie, es, man, der, die, das, ...]
  stop_objects:    [der, die, das, erforderlich, zulässig, ...]
  stop_predicates: [§, S., Nr., Hundegesetz, ...]
  pronouns:        [er, sie, es, man, wer, ...]
```

#### Normtypen

| Normtyp | Typische Auslöser |
|---|---|
| `MUST` | muss, hat zu, ist zu, soll, bedarf, pflichtig, ist nachzuweisen |
| `MAY` | kann, darf, ist berechtigt, erhält auf Antrag |
| `MUST_NOT` | darf nicht, ist untersagt, hat keine aufschiebende Wirkung |
| `DEF` | im Sinne, gilt als, versteht man |
| `EXCEPT` | es sei denn, sofern nicht, bleibt, gilt nicht |
| `DEADLINE` | binnen, spätestens, unverzüglich |
| `COMPETENCE` | zuständig ist, obliegt, überwacht, erfüllen |
| `SCOPE` | gilt für, findet Anwendung, dient der, entspricht |
| `CHANGE` | geändert, aufgehoben, ersetzt |
| `STATUS` | ist anerkannt, besitzt, stellt fest |

#### LLM-Normtyp-Klassifikation

Klassifiziert UNKNOWN-SVOs nach spaCy via LLM. 20 SVOs = 1 API-Call.

```yaml
llm_normtyp:
  enabled:          true
  batch_size:       20      # SVOs pro LLM-Prompt
  min_confidence:   0.75    # Unter diesem Wert → UNKNOWN bleibt
  test_max_batches: 0       # 0 = unbegrenzt. 3 = max. 60 SVOs
  delay_sek:        1.0     # Pause zwischen Batch-Calls
  log_prompts:      true
```

Logfiles: `services/ingest/logs/llm_request_*.log` und `llm_response_*.log`
(durch fortlaufende Nummer #0001 einander zugeordnet)

#### NER

```yaml
ner:
  flair_enabled:        true
  flair_model:          "flair/ner-german-legal"
  flair_min_confidence: 0.75
  combination_strategy: "merge"
  label_corrections:
    "Feststellung": "SONSTIGE"
  context_validation:
    enabled: true
    context_window: 80
```

#### Fragen-Filter

```yaml
question_filter:
  ingest:
    enabled: true
    apply_to_classes: [B, C]
    action: "exclude"
  nlp:
    enabled: true
    apply_to_classes: [B, C]
    action: "skip"
```

---

### 4.2 embedder_config.yaml

```yaml
active_model: "deepset-mxbai"

models:
  deepset-mxbai:
    model_id:       "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    dimensions:     1024
    prefix_query:   "query: "
    prefix_passage: "passage: "

runtime:
  batch_size:    16
  device:        "auto"
  cache_enabled: true
```

---

### 4.3 abbrev_dict.yaml

```yaml
abbreviations:
  - abbrev:   "NHundG"
    resolved: "Niedersächsisches Gesetz über das Halten von Hunden"
    context:  []
    label:    "GESETZ"

synonyms:
  - canonical: "Hundehalter"
    variants:  ["Hundehalterin", "Halterin oder Halter"]
    context:   ["Hund", "Tier"]
    label:     "ROLLE"
```

---

### 4.4 chunker_config.yaml

```yaml
chunking:
  class_a:
    token_limit_parent: 1024
    token_limit_child:  256
  class_b:
    token_limit_parent: 128
    token_limit_child:  512
    overlap_ratio:      0.15
  class_c:
    token_min:       50
    token_max:       384
    overlap_ratio:   0.20
    score_threshold: 0.72
```

---

### 4.5 docs.yaml

```yaml
Hundegesetz.pdf:
  source_type:  gesetz
  title:        NHundG – Niedersächsisches Hundegesetz
  jurisdiction: NDS
  force_class:  A

FAQ1-NHundG.pdf:
  source_type: auslegung
  title:       FAQ Sachkundenachweis
```

---

## 5. LLM-Gateway & Prompt-Suite

### 5.1 llm_gateway_config.yaml

```yaml
active_provider: gemini_flash_lite  # gemini | gemini_flash_lite | openai | anthropic | ollama

providers:
  gemini:
    model:       "gemini-2.5-flash"
    api_key_env: "GEMINI_API_KEY"
    beschreibung: "Google Gemini 2.5 Flash – Free Tier"

  gemini_flash_lite:
    model:       "gemini-3.1-flash-lite"
    api_key_env: "GEMINI_API_KEY"
    beschreibung: "Google Gemini 3.1 Flash-Lite – 2.5× schneller, günstiger"

  ollama:
    model:    "mistral:7b-instruct"
    base_url: "http://localhost:11434"
    beschreibung: "Ollama – On-Premise, kein Internet nötig"

retry:
  max_versuche:  3
  wartezeit_sek: 8.0
```

**Provider wechseln:**
```bash
# 1. active_provider ändern
# 2. Gateway neu laden:
curl -X POST http://localhost:8000/api/v1/llm/reload
curl http://localhost:8000/api/v1/llm/status
```

### 5.2 Prompt-Suite

Verzeichnis: `services/ingest/prompt_suite/{key}/system.txt` + `user.txt`
Index: `services/ingest/prompt_suite_index.yaml`

| Key | Beschreibung | Einsatz |
|---|---|---|
| `ps_normtyp` | Batch-Klassifikation Normtypen (20 SVOs = 1 Call) | M2 NLP-Worker |
| `ps_pipeline` | 7-Stufen-Pipeline (Dokumentstruktur → Triples) | M3 |
| `ps_svo_enrich` | SVO-Anreicherung (Bedingungen, Fristen) | M3 |
| `ps_norm_logic` | Wenn-Dann-Normlogik | M3 |

**Prompt testen:**
```bash
curl -X POST http://localhost:8000/api/v1/llm/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_key": "ps_normtyp",
    "variablen": {
      "norm_reference": "§ 3 NHundG",
      "predicate": "muss",
      "sentence_text": "Der Hundehalter muss die Sachkunde besitzen."
    },
    "json_mode": true
  }'
```

---

## 6. Programme & Skripte

```bash
cd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate
```

| Skript | Beschreibung |
|---|---|
| `ingest_cli.py` | Dokumente einspielen |
| `nlp_worker.py` | NLP + LLM-Klassifikation ausführen |
| `nlp_quality_report.py` | NLP-Qualitätsbericht (7 Abschnitte) |
| `ingest_report.py` | Ingest-Qualitätsbericht |
| `question_report.py` | Gefilterte Fragen verwalten |
| `nlp_monitor.py` | NLP-Fortschritt live |
| `kg_worker.py` | Knowledge Graph exportieren |

```bash
# Testlauf:
cd ~/reg-mo/ndi/Test
python run_test.py
```

---

## 7. Häufige Anpassungen

### LLM-Provider wechseln
```yaml
# llm_gateway_config.yaml
active_provider: ollama    # On-Premise ohne Internet
```
```bash
curl -X POST http://localhost:8000/api/v1/llm/reload
```

### Test-Budget begrenzen
```yaml
# nlp_config.yaml → llm_normtyp
test_max_batches: 3    # max. 3 × 20 = 60 SVOs an LLM
```

### Normtyp-Pattern ergänzen
```yaml
# nlp_config.yaml → normtypen.MUST.patterns
- '\bist\s+anzuwenden\b'
```

### Neue Abkürzung
```yaml
# abbrev_dict.yaml → abbreviations
- abbrev: "MeldeG BW"
  resolved: "Meldegesetz Baden-Württemberg"
  context: []
  label: "GESETZ"
```

### NER-Fehler korrigieren
```yaml
# nlp_config.yaml → ner.label_corrections
"Feststellung": "SONSTIGE"
```

### Flair-Sensitivität anpassen
```yaml
# nlp_config.yaml → ner
flair_min_confidence: 0.70    # mehr Treffer (mehr False Positives)
flair_min_confidence: 0.85    # weniger Treffer (weniger False Positives)
```

---

## 8. Config-Manager

```
http://localhost:8000/api/v1/config/ui
```

| Menüpunkt | Inhalt |
|---|---|
| 🏠 Home | Übersicht aller Konfigurations-Karten |
| 📋 Config-Übersicht | Diese Datei (nur Anzeige) |
| 📊 Report & Monitor | Fragen, Ingest, NLP-Qualität (7 Abschnitte), Monitor |
| 🧠 NLP-Konfiguration | spaCy, SVO, Normtypen, NER, Fragen-Filter, Worker, LLM-Normtyp |
| 🔢 Embedding-Modell | Modell-Auswahl, Batch-Size, Device |
| 📖 Wörterbücher | Abkürzungen und Synonyme |
| ✂️ Chunker-Konfiguration | Token-Limits und Chunk-Typ-Patterns |
| 🗂️ Dokument-Metadaten | Dokumenttypen, Titel, Typ-Erkennung |
| 🤖 LLM-Gateway | Provider, Modell, Retry |
| 📝 Prompt-Suite | System/User-Prompts editieren + Test-Tab |

> Vor jedem Speichern wird automatisch ein `.yaml.bak` Backup angelegt.

---

*Alle YAML-Konfigurationen werden zur Laufzeit frisch gelesen –
kein Neustart erforderlich. Ausnahme: `embedder_config.yaml` → NLP-Worker neu starten.*
