# NDI / MNR – Meta-Normen-Register

**Meilenstein M2 abgeschlossen** | Stand: Mai 2026

KI-Plattform zur automatisierten Digitalisierung von Normen und
Verwaltungsvorschriften. Basiert ausschließlich auf Open-Source-Technologien,
on-premise-fähig und BSI-konform.

---

## Was ist das MNR?

Das Meta-Normen-Register (MNR) liest normative Quelldokumente (Gesetze,
Verordnungen, Standards), extrahiert daraus ein verifizierbares
Informationsmodell und leitet technische Artefakte ab (DB-Schema, OpenAPI,
Validierungsregeln). Jede Aussage im System ist auf eine Rechtsquelle
zurückführbar.

---

## Architekturprinzipien

| Prinzip | Ausprägung |
|---|---|
| Open Source Only | Ausschließlich OSS-Komponenten |
| Traceability | Jede Aussage auf Rechtsquelle zurückführbar |
| Human-in-the-Middle | Fachaufsicht genehmigt jeden Phasenübergang |
| Datensouveränität | On-Premise, vollständig selbst betreibbar |
| Fachagnostik | Plattform ist registerunabhängig wiederverwendbar |

---

## Aktueller Stand – M2 abgeschlossen

### Was M1 geliefert hat

- Dokumenten-Ingest (PDF, DOCX, HTML) via Apache Tika + Docling
- Dreiklassiges Chunking (A: §-Paragraphen, B: Kapitel, C: Fließtext)
- Embedding und Vektorspeicherung (pgvector)
- Abkürzungsauflösung und Synonym-Normalisierung
- Config-Manager (Browser-UI für alle Konfigurationen)

### Was M2 geliefert hat

- **SVO-Extraktion** – Subjekt-Prädikat-Objekt aus Normtexten via spaCy
- **NER** – Named Entity Recognition (Behörden, Rollen, Gesetze, Fristen)
- **LLM-Normtyp-Klassifikation** – Batch-Prompt: 20 SVOs = 1 API-Call
- **LLM-Gateway** – austauschbarer Adapter (Gemini, GPT, Claude, Ollama)
- **Prompt-Suite** – verwaltbare System/User-Prompts im Config-Manager
- **Knowledge Graph** – RDF-Export nach Apache Jena Fuseki
- **Qualitätsbericht** – 7-Abschnitte NLP-Report mit Handlungsempfehlungen

### M2-Qualitätsergebnis (NHundG, 94 Chunks, 288 SVOs)

| Kennzahl | Ergebnis | Ziel |
|---|---|---|
| UNKNOWN-Rate | **16.0%** | < 20% ✅ |
| Mit Subjekt | 65.3% | > 80% ⚠️ |
| Mit Objekt | ~52% | > 50% ✅ |
| Ø SVO-Konfidenz | 0.823 | > 0.65 ✅ |
| NLP-Abdeckung | 91.5% | > 60% ✅ |
| Flair NER | 21.5% | > 40% ⚠️ |

Die verbleibende UNKNOWN-Rate (16%) und der niedrige Flair-Anteil (21.5%)
sind auf die Besonderheiten der deutschen Rechtssprache zurückzuführen
und für M3 (RAG-Engine) ausreichend.

---

## Technologie-Stack

### Infrastruktur

| Komponente | Technologie | Lizenz |
|---|---|---|
| Datenbank + Vektorspeicher | PostgreSQL 16 + pgvector | Apache 2.0 |
| Objektspeicher | MinIO | AGPL 3.0 |
| Dokument-Parsing | Apache Tika + Docling | Apache 2.0 |
| Knowledge Graph | Apache Jena Fuseki | Apache 2.0 |
| Containerisierung | Docker + Docker Compose | Apache 2.0 |

### KI / ML

| Komponente | Technologie |
|---|---|
| LLM (Cloud) | Google Gemini 2.5 Flash / 3.1 Flash-Lite |
| LLM (On-Premise) | Ollama + Mistral 7B / Llama 3.1 8B |
| Embedding | mixedbread-ai/deepset-mxbai-embed-de-large-v1 |
| NLP Pipeline | spaCy 3.x + de_core_news_lg |
| NER (juristisch) | Flair + flair/ner-german-legal |
| RAG Framework | LlamaIndex (ab M3) |

### Backend

| Komponente | Technologie |
|---|---|
| API-Framework | FastAPI |
| Authentifizierung | Keycloak OIDC (ab M4) |
| Orchestrierung | Apache Airflow (ab M3) |

---

## Verzeichnisstruktur

```
NDI/
├── .env                         # Passwörter + API-Keys (nicht im Git)
├── .env.example                 # Vorlage
├── docker-compose.yml           # Infrastruktur-Stack
│
├── infra/
│   └── postgres/
│       ├── init/                # DB-Migrations (01–08)
│       └── maintenance/         # Reset-Skripte
│
├── services/
│   └── ingest/                  # FastAPI-Service
│       ├── main.py
│       ├── requirements.txt
│       │
│       ├── nlp_config.yaml      # NLP + LLM-Normtyp
│       ├── embedder_config.yaml
│       ├── abbrev_dict.yaml
│       ├── chunker_config.yaml
│       ├── docs.yaml
│       ├── llm_gateway_config.yaml
│       ├── prompt_suite_index.yaml
│       │
│       ├── llm_gateway/
│       │   ├── gateway.py       # LLM-Adapter (Gemini/GPT/Claude/Ollama)
│       │   ├── prompt_suite.py
│       │   └── schema_registry.py
│       │
│       ├── prompt_suite/
│       │   ├── ps_normtyp/      # Batch-Normtyp-Klassifikation
│       │   ├── ps_pipeline/     # 7-Stufen-Pipeline (M3)
│       │   └── ps_norm_logic/   # Wenn-Dann-Normlogik (M3)
│       │
│       ├── schema_registry/     # JSON-Ergebnisstrukturen
│       │
│       ├── logs/                # LLM Request/Response Logs
│       │
│       └── app/
│           ├── api/routes/      # FastAPI-Endpoints
│           ├── core/config.py   # pydantic-settings
│           └── services/
│               ├── nlp/
│               │   ├── nlp_service.py
│               │   ├── svo_extractor.py
│               │   ├── ner_extractor.py
│               │   └── llm_normtyp.py   # LLM-Klassifikation
│               ├── parser.py
│               ├── chunker.py
│               ├── embedder.py
│               ├── storage.py
│               └── knowledge_graph.py
│
└── Test/
    ├── run_test.py              # Vollständiger Testlauf
    └── docs/                   # Testdokumente
```

---

## Schnellstart

```bash
# 1. Infrastruktur starten
cd ~/reg-mo/ndi
docker compose up -d

# 2. FastAPI starten
cd services/ingest
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Testlauf (Ingest + NLP)
cd ~/reg-mo/ndi/Test
python run_test.py

# 4. Qualitätsbericht
cd ~/reg-mo/ndi/services/ingest
python nlp_quality_report.py

# 5. Config-Manager öffnen
# http://localhost:8000/api/v1/config/ui
```

---

## LLM-Konfiguration

Der LLM-Gateway unterstützt mehrere Provider.
Wechsel ohne Code-Änderung über `llm_gateway_config.yaml`:

| Provider | Modell | Kosten | Offline |
|---|---|---|---|
| `gemini` | gemini-2.5-flash | Free Tier | Nein |
| `gemini_flash_lite` | gemini-3.1-flash-lite | Free Tier | Nein |
| `openai` | gpt-4o | Kostenpflichtig | Nein |
| `anthropic` | claude-opus-4-5 | Kostenpflichtig | Nein |
| `ollama` | mistral:7b-instruct | Kostenlos | Ja ✅ |

```bash
# Provider wechseln:
# 1. llm_gateway_config.yaml → active_provider ändern
curl -X POST http://localhost:8000/api/v1/llm/reload
curl http://localhost:8000/api/v1/llm/status
```

---

## Meilenstein-Roadmap

| Meilenstein | Status | Beschreibung |
|---|---|---|
| **M1 – Ingest & Vektordatenbank** | ✅ abgeschlossen | Chunking, Embedding, Storage |
| **M2 – NLP & Knowledge Graph** | ✅ abgeschlossen | SVO, NER, LLM-Klassifikation |
| M3 – RAG-Engine | 🔜 nächster Schritt | LlamaIndex, Hybrid-Retrieval, Prompt-Pipeline |
| M4 – Informationsmodell | ⏳ geplant | IM-Generator, Human-in-the-Middle-Workflow |
| M5 – Artefakt-Generierung | ⏳ geplant | DDL, OpenAPI, DMN, XÖV |
| M6 – Produktionshärtung | ⏳ geplant | K3s, Multi-Tenancy, BSI-Härtung |

---

## API-Endpunkte (Auswahl)

| Methode | Pfad | Beschreibung |
|---|---|---|
| `GET` | `/health` | Service-Status |
| `POST` | `/api/v1/ingest/document` | Dokument einspielen |
| `POST` | `/api/v1/nlp/run` | NLP-Job starten |
| `GET` | `/api/v1/nlp/status` | NLP-Job-Status |
| `GET` | `/api/v1/llm/status` | LLM-Gateway-Status |
| `POST` | `/api/v1/llm/complete` | LLM-Aufruf mit Prompt-Suite |
| `GET` | `/api/v1/llm/prompts` | Alle Prompts auflisten |
| `GET` | `/api/v1/kg/stats` | Knowledge-Graph-Statistik |
| `GET` | `/api/v1/config/ui` | Config-Manager (Browser) |
| `GET` | `/docs` | Swagger UI |

---

## Konfigurationsdokumentation

→ **[CONFIG.md](CONFIG.md)** – vollständige Referenz aller YAML-Dateien,
Startbefehle und häufigen Anpassungen.

---

*Lizenz: Ausschließlich Open-Source-Technologien | On-Premise-fähig | BSI-konform*
