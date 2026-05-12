# NDI / MNR – Konfigurationsdateien

> Übersicht aller Konfigurationsdateien des NDI-Systems mit Pfad und Beschreibung.
> Stand: Mai 2026 | Meilenstein M1 + M2

---

## Infrastruktur

| Datei | Pfad | Beschreibung |
|---|---|---|
| `.env` | `NDI/.env` | Passwörter, Ports, Bucket-Namen. **Nicht im Git.** |
| `.env.example` | `NDI/.env.example` | Vorlage für `.env` ohne Secrets. Im Git. |
| `docker-compose.yml` | `NDI/docker-compose.yml` | Stack-Definition: postgres, minio, tika. Ports und Volumes. |

### Enthaltene Variablen (.env)

```
POSTGRES_USER         Datenbankbenutzer
POSTGRES_PASSWORD     Datenbankpasswort
POSTGRES_DB           Datenbankname
POSTGRES_PORT         Port PostgreSQL (Standard: 5432)
MINIO_ROOT_USER       MinIO Admin-Benutzer
MINIO_ROOT_PASSWORD   MinIO Admin-Passwort
MINIO_PORT            MinIO API-Port (Standard: 9000)
MINIO_CONSOLE_PORT    MinIO Console-Port (Standard: 9001)
MINIO_DEFAULT_BUCKET  Standard-Bucket (mnr-artefakte)
TIKA_PORT             Apache Tika Port (Standard: 9998)
PROJECT_NAME          Projektname (ndi)
```

---

## Datenbank-Schema

Alle SQL-Dateien liegen unter `infra/postgres/` und werden beim ersten
Container-Start automatisch eingespielt. Maintenance-Skripte müssen
manuell ausgeführt werden.

### Initialisierungs-Skripte

| Datei | Pfad | Tabellen / Inhalt |
|---|---|---|
| `01_schema.sql` | `infra/postgres/init/01_schema.sql` | `norm_documents`, `norm_chunks`, `information_models`, `im_review_log` |
| `02_jobs.sql` | `infra/postgres/init/02_jobs.sql` | `ingest_jobs` – Status-Tracking für Ingest-Jobs |
| `03_pakete.sql` | `infra/postgres/init/03_pakete.sql` | `ingest_pakete`, `ingest_paket_jobs` – Batch-Ingest |
| `04_nlp.sql` | `infra/postgres/init/04_nlp.sql` | `nlp_jobs`, `svo_extractions`, `ner_entities` |
| `05_abbrev.sql` | `infra/postgres/init/05_abbrev.sql` | Erweiterung `norm_chunks`: `content_original`, `abbrev_map` |
| `06_questions.sql` | `infra/postgres/init/06_questions.sql` | `filtered_questions` – Fragen-Log mit Review-Workflow |

#### Manuell einspielen (auf bestehender Instanz)

```bash
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  < infra/postgres/init/02_jobs.sql
```

### Maintenance-Skripte

| Datei | Pfad | Beschreibung |
|---|---|---|
| `drop_all_tables.sql` | `infra/postgres/maintenance/drop_all_tables.sql` | Alle Tabellen **und Struktur** löschen. Nur in Entwicklung. |
| `truncate_all_tables.sql` | `infra/postgres/maintenance/truncate_all_tables.sql` | Alle Inhalte löschen, Struktur bleibt erhalten. Für Test-Reset. |

```bash
# Test-Reset:
docker exec -i mnr-postgres psql -U mnr -d mnr_db \
  < infra/postgres/maintenance/truncate_all_tables.sql
```

---

## Ingest-Service – YAML-Konfigurationen

### embedder_config.yaml

**Pfad:** `services/ingest/embedder_config.yaml`

Steuert das Embedding-Modell. Modellwechsel erfordert **keinen Code-Neustart** –
nur `active_model` ändern und NLP-Worker neu starten.

```yaml
active_model: "deepset-mxbai"   # deepset-mxbai | multilingual-e5

models:
  deepset-mxbai:
    model_id:    "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    dimensions:  1024
    prefix_query:   "query: "
    prefix_passage: "passage: "

runtime:
  batch_size:    16
  device:        "auto"     # auto | cpu | cuda
  cache_enabled: true
```

---

### abbrev_dict.yaml

**Pfad:** `services/ingest/abbrev_dict.yaml`

Wörterbuch für Abkürzungsauflösung. Wird zwischen Parser und Chunker angewendet.
Änderungen wirken beim nächsten Ingest sofort.

```yaml
abbreviations:
  - abbrev:    "NHundG"
    resolved:  "Niedersächsisches Gesetz über das Halten von Hunden"
    context:   []       # leer = immer auflösen
    label:     "GESETZ"

  - abbrev:    "AG"
    resolved:  "Amtsgericht"
    context:   ["Urteil", "Beschluss", "Az."]   # nur mit Kontext auflösen
    label:     "BEHÖRDE"
```

**Felder:**

| Feld | Pflicht | Beschreibung |
|---|---|---|
| `abbrev` | ✅ | Abkürzung exakt wie im Dokument |
| `resolved` | ✅ | Vollständige Auflösung |
| `context` | ❌ | Signalwörter im Umfeld (±100 Zeichen). Leer = immer auflösen |
| `label` | ❌ | Entitätstyp: `GESETZ`, `BEHÖRDE`, `ROLLE`, `ORT`, `SONSTIGE` |

> **Hinweis:** Längere Abkürzungen werden zuerst geprüft.
> Normreferenzen (`§ X Abs. Y`) werden niemals aufgelöst.

---

### nlp_config.yaml

**Pfad:** `services/ingest/nlp_config.yaml`

Zentrale NLP-Konfiguration. Alle Parameter wirken ohne Neustart
beim nächsten `nlp_worker.py --run`.

#### spaCy

```yaml
spacy:
  model: "de_core_news_lg"    # sm | md | lg
  components: [tagger, parser, lemmatizer]
  batch_size: 32
```

#### SVO-Extraktion

```yaml
svo:
  min_confidence: 0.5
  subject_deps: [sb, sp, nsubj]
  object_deps:  [oa, og, op, oc, pd, obj, obl]

  stop_subjects: [er, sie, es, man, der, die, das, ...]
  stop_objects:  [der, die, das, dem, den, ...]
  pronouns:      [er, sie, es, man, ...]
  pronoun_confidence_penalty: 0.3
```

#### Normtypen

```yaml
normtypen:
  MUST:      { patterns: ['\bmuss\b', '\bmüssen\b', ...], confidence_boost: 0.0 }
  MAY:       { patterns: ['\bkann\b', '\bdarf\b', ...],   confidence_boost: 0.0 }
  MUST_NOT:  { patterns: ['\bdarf\s+nicht\b', ...],       confidence_boost: 0.05 }
  DEF:       { patterns: ['\bim\s+Sinne\b', ...],         confidence_boost: 0.1 }
  EXCEPT:    { patterns: ['\bes\s+sei\s+denn\b', ...],    confidence_boost: 0.0 }
  DEADLINE:  { patterns: ['\bbinnen\b', ...],             confidence_boost: 0.05 }
  COMPETENCE:{ patterns: ['\bzuständig\s+ist\b', ...],    confidence_boost: 0.0 }
```

> Regex-Patterns **müssen** in einfachen Anführungszeichen stehen (`'\b...\b'`).

#### NER

```yaml
ner:
  rules_enabled: true
  flair_enabled: true
  flair_model:   "flair/ner-german-legal"
  flair_min_confidence: 0.85
  combination_strategy: "merge"   # merge | flair_only | rules_only

  blacklist:          [Google, Apple, Microsoft, ...]
  label_corrections:  { "Telekom": "SONSTIGE", ... }

  context_validation:
    enabled: true
    BEHÖRDE:
      required_context: [behörde, amt, ministerium, ...]
      fallback_label: "SONSTIGE"
```

#### Fragen-Filter

```yaml
question_filter:

  ingest:                       # Schalter für Ingest-Pipeline
    enabled: true               # true | false
    apply_to_classes: [B, C]
    action: "exclude"           # exclude | include | include_as_type

  nlp:                          # Schalter für NLP-Worker
    enabled: true               # true | false
    apply_to_classes: [B, C]
    action: "skip"              # skip | process

  detection:                    # gemeinsame Erkennungsparameter
    min_length: 10
    max_length: 300
    require_question_mark: true
    question_words: [wer, was, wie, wo, wann, ...]
```

**Kombinationen:**

| `ingest.enabled` | `nlp.enabled` | Effekt |
|---|---|---|
| `true` | `true` | Fragen weder in Chunks noch in SVO/NER |
| `true` | `false` | Fragen nicht in Chunks, aber NLP verarbeitet sie |
| `false` | `true` | Fragen in Chunks, aber NLP überspringt sie |
| `false` | `false` | Fragen werden vollständig normal verarbeitet |

#### Worker

```yaml
worker:
  process_classes:    [A, B, C]
  chunk_batch_size:   50
  overwrite_existing: true    # true = NLP-Ergebnisse bei Neustart überschreiben
  min_token_count:    10
  show_progress:      true
```

---

### test_chunker_config.yaml

**Pfad:** `services/ingest/test_chunker_config.yaml`

Parameter für `test_chunker.py`. Wird automatisch gefunden wenn im
gleichen Verzeichnis wie das Test-Skript.

```yaml
output:
  limit: 0              # 0 = alle Chunks
  file: null            # null = nur Konsole
  text_preview_length: 200

document:
  force_class: null     # null | A | B | C

synthetic_tests:
  enabled: true

chunking:
  class_a: { token_limit_parent: 1024, token_limit_child: 256 }
  class_b: { token_limit_child: 512, overlap_ratio: 0.15 }
  class_c: { token_max: 384, overlap_ratio: 0.20 }
```

---

## Test-Skripte

| Datei | Pfad | Beschreibung |
|---|---|---|
| `run_test.py` | `Test/run_test.py` | Vollständiger Testlauf: Reset → Einspielen → Prüfen → Log. Liest Credentials aus `.env`. |

```bash
cd /home/mdi/reg-mo/ndi/Test
source ../services/ingest/.venv/bin/activate

python run_test.py                              # Vollständiger Testlauf
python run_test.py --skip-reset                # Nur einspielen + prüfen
python run_test.py --docs-dir /anderer/pfad    # Anderes Docs-Verzeichnis
python run_test.py --timeout 600               # Längere Wartezeit (Sek.)
```

---

## Häufige Anpassungen

### Neues Embedding-Modell aktivieren

```yaml
# embedder_config.yaml
active_model: "multilingual-e5"   # Wechsel auf Fallback-Modell
```

### Neue Abkürzung ergänzen

```yaml
# abbrev_dict.yaml
- abbrev: "MeldeG BW"
  resolved: "Meldegesetz Baden-Württemberg"
  context: []
  label: "GESETZ"
```

### Neuen Normtyp-Pattern ergänzen

```yaml
# nlp_config.yaml → normtypen.MUST.patterns
- '\bist\s+anzuwenden\b'
```

### False Positive in NER ausschließen

```yaml
# nlp_config.yaml → ner.blacklist
- "NeuerFalsePositive"
```

### Fragen-Filter für Tests deaktivieren

```yaml
# nlp_config.yaml
question_filter:
  ingest:
    enabled: false
  nlp:
    enabled: false
```

---

*Alle YAML-Konfigurationen werden zur Laufzeit frisch gelesen –
ein Neustart des Services ist nicht erforderlich.*
