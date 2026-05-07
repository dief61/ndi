# NDI / MNR

System-Startanleitung für **Windows 11 / WSL2 / Ubuntu 24.04**.

> Version 1.0 · Stand: April 2026

---

## Überblick

Diese Anleitung beschreibt Voraussetzungen, Start, Stop, Prüfungen und typische Fehler für das NDI/MNR-System.

## Inhalt

- [Voraussetzungen](#voraussetzungen)
- [Normaler Systemstart](#normaler-systemstart)
- [Dienste und Ports](#dienste-und-ports)
- [Schnellchecks](#schnellchecks)
- [System stoppen](#system-stoppen)
- [Zugriff von anderen Rechnern](#zugriff-von-anderen-rechnern)
- [Häufige Probleme](#häufige-probleme)
- [CLI-Schnellreferenz](#cli-schnellreferenz)

---

## Voraussetzungen

> ⚠ Diese Schritte müssen nur einmalig durchgeführt werden. Beim normalen Neustart kannst du direkt mit dem Systemstart beginnen.

| Komponente | Version | Prüfbefehl (in WSL) |
|---|---:|---|
| WSL 2 + Ubuntu 24.04 | 24.04 | `lsb_release -a` |
| Docker Desktop | 26+ | `docker --version` |
| Python | 3.12 | `python3 --version` |
| Git | 2.x | `git --version` |
| Ollama | 0.3+ | `ollama --version` |

---

## Normaler Systemstart

### 1. Docker Desktop starten

Docker Desktop unter Windows starten und warten, bis es vollständig hochgefahren ist. Das Taskleisten-Icon wird grün.

### 2. WSL-Terminal öffnen

Ein WSL-Terminal mit Ubuntu 24.04 öffnen. Alle weiteren Befehle laufen in WSL.

### 3. Docker-Stack starten

```bash
cd ~/reg-mo/ndi
docker compose up -d
```

### 4. Stack-Status prüfen

```bash
docker compose ps
```

Erwartet wird, dass alle Container `running` sind:

- `mnr-postgres`
- `mnr-minio`
- `mnr-tika`

### 5. FastAPI-Service starten

In einem neuen WSL-Terminal:

```bash
cd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Erwartete Ausgabe:

- `Uvicorn running on http://0.0.0.0:8000`

> ⚠ Uvicorn blockiert das Terminal. Für weitere Befehle ein zweites WSL-Terminal öffnen.

### 6. System prüfen

In einem zweiten WSL-Terminal:

```bash
curl http://localhost:8000/health/ready
```

Erwartete Antwort:

```json
{"status":"ready","checks":{"postgres":"ok","tika":"ok"}}
```

---

## Dienste und Ports

| Service | Port | Adresse |
|---|---:|---|
| FastAPI / Swagger UI | 8000 | http://localhost:8000/docs |
| PostgreSQL | 5432 | localhost:5432 (User: `mnr` / DB: `mnr_db`) |
| MinIO Console | 9001 | http://localhost:9001 (mnr_admin) |
| MinIO API | 9000 | http://localhost:9000 |
| Apache Tika | 9998 | http://localhost:9998/tika |

---

## Schnellchecks

| Prüfung | Befehl |
|---|---|
| Docker-Container laufen | `docker compose ps` |
| PostgreSQL erreichbar | `docker exec -it mnr-postgres psql -U mnr -d mnr_db -c '\dt'` |
| Tika erreichbar | `curl http://localhost:9998/tika` |
| FastAPI erreichbar | `curl http://localhost:8000/health/` |
| FastAPI Readiness | `curl http://localhost:8000/health/ready` |

---

## System stoppen

### FastAPI stoppen

Im Uvicorn-Terminal `CTRL+C` drücken.

### Docker-Stack stoppen

```bash
cd ~/reg-mo/ndi
docker compose down
```

> ⚠ `docker compose down -v` löscht alle Volumes und damit alle Daten in PostgreSQL und MinIO. Nur in Entwicklung verwenden.

---

## Zugriff von anderen Rechnern

Da der Service in WSL läuft, muss eine Port-Weiterleitung von Windows nach WSL eingerichtet werden. Diese muss nach jedem WSL-Neustart aktualisiert werden, da sich die WSL-IP ändert.

Nach jedem WSL-Neustart in PowerShell als Administrator:

```powershell
powershell -ExecutionPolicy Bypass -File C:\scripts\wsl-portforward.ps1
```

Danach ist der Service unter `http://<Windows-IP>:8000/docs` erreichbar.

---

## Häufige Probleme

| Problem | Lösung |
|---|---|
| Port 8000 antwortet nicht | Uvicorn läuft nicht → Schritt 5 wiederholen |
| `docker compose ps` zeigt `exited` | `docker compose up -d` nochmal ausführen |
| venv nicht aktiviert: `No module named...` | `source .venv/bin/activate` ausführen |
| WSL-IP hat sich geändert | `wsl-portforward.ps1` als Admin ausführen |
| Tika antwortet nicht | `docker compose restart tika` |
| PostgreSQL-Verbindungsfehler | `docker compose restart postgres` |
| Embedding dauert sehr lang | Normal auf CPU (~1s/Chunk), GPU deutlich schneller |

---

## CLI-Schnellreferenz

### Dokument per CLI einspeisen

```bash
cd ~/reg-mo/ndi/services/ingest
source .venv/bin/activate

# Einzeldokument:
python ingest_cli.py --pdf /pfad/zum/dokument.pdf \
  --source-type gesetz --title "Titel" --jurisdiction NDS

# Mit Klasse erzwingen:
python ingest_cli.py --pdf /pfad/dokument.pdf --class A

# Jobs anzeigen:
python ingest_cli.py --jobs

# Job-Status:
python ingest_cli.py --status <job_id>
```

### Dokument per API einspeisen

```bash
# Einzeldokument:
curl -X POST http://localhost:8000/api/v1/ingest/document \
  -F "file=@/pfad/dokument.pdf" \
  -F "source_type=gesetz" \
  -F "title=Titel"

# Paket (Batch):
curl -X POST http://localhost:8000/api/v1/ingest/paket \
  -H "Content-Type: application/json" \
  -d '{
    "paket_id": "b7c9a0e1-...",
    "version_id": "a3f1e2d4-...",
    "paket_name": "Mein Paket",
    "version": "1.0.0",
    "dokument_ids": ["<doc_id_1>", "<doc_id_2>"]
  }'
```

---

## Hinweise

- Projektpfad: `~/reg-mo/ndi`
- FastAPI-Doku: `/docs`
- Für Netzwerkzugriff von außen ist die Windows-zu-WSL-Portweiterleitung erforderlich


## Ingest Pipeline

 POST /api/v1/ingest/document
         │
         ▼

| ingest.py | (FastAPI-Route) |
|---|---|
| Datei empfangen | doc_id und job_id generieren |
| Background-Task starten | sofort HTTP 200 antworten |
|---|---|

       │  (Background-Task)
       ▼

## Hauptingest-Pipeline

┌─────────────────────────────────────────────────────────┐
│ ingest_service.py run_pipeline() │
├─────────────────────────────────────────────────────────│
│ 1. Job in ingest_jobs anlegen (queued) │
│ 2. Rohdatei → MinIO mnr-dokumente/{doc_id}/{filename} │
│ 3. Metadaten → PostgreSQL → norm_documents │
│ 4. parser.py TikaParser.parse() → Text + Struktur + doc_class_hint (A/B/C) │
│ 5. chunker.py ChunkingRouter.route_and_chunk() → Chunks mit Metadaten │
│ 6. embedder.py Embedder.embed_chunks() → 1024-dim Vektoren je Chunk │
│ 7. storage.py DocumentStorage.store_chunks() → norm_chunks + Embeddings in PostgreSQL │
│ 8. Job-Status → done │
└─────────────────────────────────────────────────────────┘
      │  (manuell gestartet, NACH Ingest)
      ▼


┌─────────────────────────────────────┐
│ 1. Erste wichtige Information │
│ 2. Zweite wichtige Information │
│ 3. Dritte wichtige Information │
└─────────────────────────────────────┘




|---|
|  nlp_worker.py  (Option A – Post-Ingest)                |
|                                                         |
|  Schritt 1:  Chunks aus norm_chunks laden               |
|      │                                                  |
|  Schritt 2:  nlp_processor.py                           |
|      │       NLPProcessor.analyze_batch()               |
|      │       → spaCy: POS, Dependency Parsing           |
|      │                                                  |
|  Schritt 3:  svo_extractor.py                           |
|      │       SVOExtractor.extract()                     |
|      │       → SVO-Tripel + Normtypen                   |
|      │       → Stoppwort-Filter                         |
|      │                                                  |
|  Schritt 4:  ner_extractor.py                           |
|      │       NERExtractor.extract()                     |
|      │       Stufe 1: Regelbasiert                      |
|      │       Stufe 2: Flair Legal NER                   |
|      │       → Blacklist / Label-Korrekturen            |
|      │       → Kontext-Validierung                      |
|      │                                                  |
|  Schritt 5:  Ergebnisse → PostgreSQL                    |
|              → svo_extractions                          |
|              → ner_entities                             |
|---|


## Unterstützte Formate

| Gruppe | Formate | Klasse |
|--------|---------|--------|
| **Textdokumente** | PDF, DOCX, DOC, RTF, TXT, HTML | A/B/C auto |
| **LibreOffice** | ODT, ODS, ODP | A/B auto, ODP→C |
| **Microsoft Office** | XLSX, XLS, PPTX, PPT | XLSX/XLS auto, PPTX/PPT→C |
| **Strukturiert/XÖV** | XML, JSON, CSV, TSV | A/B/C auto |
| **E-Books** | EPUB | A/B/C auto |
| **Mail** | EML, MSG | C (kein Strukturgerüst) |
