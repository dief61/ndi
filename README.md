# NDI / MNR – Meta-Normen-Register

**Meilenstein M2 abgeschlossen · M3 in Entwicklung** | Stand: Mai 2026

KI-Plattform zur automatisierten Digitalisierung von Normen und
Verwaltungsvorschriften. Basiert ausschließlich auf Open-Source-Technologien,
on-premise-fähig und BSI-konform.

---

## Inhaltsverzeichnis

1. [Was ist das MNR?](#1-was-ist-das-mnr)
2. [Architekturprinzipien](#2-architekturprinzipien)
3. [Kernkonzepte: SVO und NER](#3-kernkonzepte-svo-und-ner)
4. [Technologie-Stack](#4-technologie-stack)
5. [Meilenstein-Roadmap](#5-meilenstein-roadmap)
6. [Verzeichnisstruktur](#6-verzeichnisstruktur)
7. [Schnellstart](#7-schnellstart)
8. [LLM-Konfiguration](#8-llm-konfiguration)
9. [API-Endpunkte](#9-api-endpunkte)
10. [Qualitätsergebnisse M2](#10-qualitätsergebnisse-m2)

---

## 1. Was ist das MNR?

Das Meta-Normen-Register (MNR) liest normative Quelldokumente – Gesetze,
Verordnungen, Standards, Fachkonzepte – und erzeugt daraus automatisch ein
verifizierbares Informationsmodell. Aus diesem Modell werden technische
Artefakte abgeleitet: Datenbankschemas, OpenAPI-Spezifikationen,
Validierungsregeln und Register-Blueprints.

Das System folgt dem Prinzip **"Law as a System Source"**: Jede Aussage im
Informationsmodell ist auf eine konkrete Rechtsquelle zurückführbar.

Typischer Anwendungsfall:

```
Eingabe:  §3 NHundG – "Wer einen Hund haelt, muss die Sachkunde besitzen."

Ausgabe:  Entitaet:  Hundehalter        -> Tabelle in der Datenbank
          Attribut:  sachkunde_nachweis -> Pflichtspalte (NOT NULL)
          Regel:     MUST               -> Validierungsregel
          Quelle:    §3 Abs. 1 NHundG   -> Traceability
```

---

## 2. Architekturprinzipien

| Prinzip | Ausprägung |
|---|---|
| Open Source Only | Ausschliesslich OSS-Komponenten, kein proprietäres Lock-in |
| Traceability | Jede Aussage im Modell ist auf eine Rechtsquelle zurückführbar |
| Human-in-the-Middle | Fachaufsicht genehmigt jeden Phasenübergang |
| Fachagnostik | Plattform ist register- und behördenunabhängig wiederverwendbar |
| Datensouveränität | On-Premise-fähig, vollständig selbst betreibbar |
| Interoperabilität | XÖV, OpenAPI, JSON-LD, RDF/OWL als Ausgabeformate |

---

## 3. Kernkonzepte: SVO und NER

Die NLP-Pipeline von MNR basiert auf zwei sich ergänzenden Verfahren.

### SVO – Subjekt-Verb-Objekt

SVO ist eine Methode der Informationsextraktion aus natürlichsprachlichen
Texten. Sie zerlegt einen Satz in drei semantische Grundkomponenten:

**Subjekt** – Wer handelt? Der Akteur oder die Rolle, die eine Handlung
ausführt oder einer Pflicht unterliegt.

**Verb (Prädikat)** – Was wird getan? Die Handlung, Pflicht oder
Berechtigung – im MNR angereichert um die rechtliche Modalität (Normtyp).

**Objekt** – Worauf bezieht sich die Handlung? Das betroffene Datenobjekt,
der Verwaltungsakt oder die Entität.

Beispiel aus dem NHundG:

```
Normtext:  "Die Fachbehoerde ueberwacht die Einhaltung der Vorschriften."

Subjekt:   Fachbehoerde     [BEHOERDE]
Praedikat: ueberwacht       [COMPETENCE]
Objekt:    Einhaltung       [DATENOBJEKT]
```

Im MNR sind SVOs die **Bausteine des Knowledge Graph**. Jedes Triple
repräsentiert eine normative Aussage mit Quellenreferenz und Normtyp.

Unterstützte Normtypen:

| Normtyp | Bedeutung | Typische Auslöser |
|---|---|---|
| `MUST` | Pflicht | muss, hat zu, ist zu, bedarf |
| `MAY` | Erlaubnis / Befugnis | kann, darf, ist berechtigt |
| `MUST_NOT` | Verbot | darf nicht, ist untersagt |
| `DEF` | Definition | im Sinne, gilt als |
| `EXCEPT` | Ausnahme | es sei denn, sofern nicht |
| `DEADLINE` | Frist | binnen, spätestens, unverzüglich |
| `COMPETENCE` | Zuständigkeit | zuständig ist, obliegt, überwacht |
| `SCOPE` | Geltungsbereich | gilt für, findet Anwendung |
| `CHANGE` | Rechtsänderung | geändert, aufgehoben, ersetzt |
| `STATUS` | Zustandsbeschreibung | ist anerkannt, besitzt, stellt fest |

### NER – Named Entity Recognition

NER ist ein Verfahren des Natural Language Processing, das **benannte
Entitäten** in Texten automatisch erkennt und klassifiziert. Eine Entität
ist ein Begriff, der eine reale Größe benennt – eine Person, eine
Organisation, ein Ort, ein Dokument.

Im MNR werden folgende Entitätstypen erkannt:

| Label | Beschreibung | Beispiele |
|---|---|---|
| `BEHOERDE` | Behörden und staatliche Stellen | Fachbehörde, Gemeinde, Fachministerium |
| `ROLLE` | Personen-Rollen im Normkontext | Hundehalter, Antragsteller, Person |
| `GESETZ` | Gesetze, Verordnungen, Normen | NHundG, TierSchG, BGB |
| `DATENOBJEKT` | Verwaltungsakte und Datenobjekte | Sachkundenachweis, Erlaubnis, Register |
| `FRIST` | Zeitliche Angaben | binnen zwei Wochen, unverzüglich |
| `ORT` | Geografische Angaben | Niedersachsen, Bayern |
| `SONSTIGE` | Sonstige relevante Entitäten | – |

Die NER im MNR arbeitet zweistufig mit einer agnostischen Basisschicht und
einer register-spezifischen Erweiterungsschicht:

```
Schicht 1 – Agnostisch (nlp_config.yaml):
  Plattformweite Verwaltungsbegriffe, die in jedem Register auftreten.
  Beispiele: Antrag, Erlaubnis, Nachweis, Bescheinigung, Register

Schicht 2 – Register-spezifisch (docs.yaml -> ner_extensions):
  Fachbegriffe, die nur im Kontext eines bestimmten Registers gelten.
  Beispiele NHundG: Hundehaltung, Wesenstest, Transponder
```

Die Klassifikation erfolgt über zwei technische Stufen:

**Stufe 1 – Regelbasiert:** Suffix-, Exact- und Regex-Matching auf Basis
von `nlp_config.yaml`. Schnell, deterministisch, domänenspezifisch.

**Stufe 2 – Flair NER:** KI-Modell `flair/ner-german-legal`, trainiert auf
67.000 deutschen Gerichtsentscheidungen. Erkennt 19 feingranulare
juristische Entitätsklassen.

### Zusammenspiel von SVO und NER

```
Normtext
    |
    |-> NER  -> Entitaeten mit Labels
    |          "Fachbehoerde" [BEHOERDE]
    |          "Sachkunde"    [DATENOBJEKT]
    |
    `-> SVO  -> Triples mit Normtyp
               Fachbehoerde -> COMPETENCE -> Einhaltung
                    |
                    v
              Knowledge Graph (Apache Jena Fuseki)
                    |
                    v
              Informationsmodell (M4)
              -> Entitaeten, Attribute, Regeln, Traceability
```

NER liefert die **Typisierung** der Entitäten, SVO liefert die
**Beziehungen** zwischen ihnen. Beide zusammen bilden die semantische
Grundlage für den Knowledge Graph und das Informationsmodell.

### LLM-Anreicherung

Seit M2 werden UNKNOWN-SVOs durch ein LLM nachklassifiziert.
20 SVOs werden in einem Prompt-Aufruf verarbeitet (Batch-Prompt):

```
spaCy extrahiert SVOs
    |
    |-> Normtyp erkannt (MUST/MAY/...) -> direkt speichern
    `-> UNKNOWN -> Batch an LLM (20 SVOs = 1 API-Call)
                       |
                       `-> Normtyp + Begruendung + Konfidenz
```

---

## 4. Technologie-Stack

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
| Konfiguration | pydantic-settings |
| Orchestrierung | Apache Airflow (ab M3) |
| LLM-Gateway | Eigener Adapter (Gemini / GPT / Claude / Ollama) |

---

## 5. Meilenstein-Roadmap

| Meilenstein | Status | Beschreibung |
|---|---|---|
| **M1 – Ingest & Vektordatenbank** | abgeschlossen | Chunking (A/B/C), Embedding, Storage |
| **M2 – NLP & Knowledge Graph** | abgeschlossen | SVO, NER, LLM-Klassifikation, KG |
| **M3 – RAG-Engine** | in Entwicklung | Hybrid-Retrieval, Context Assembly mit IM-Signalen |
| M4 – Informationsmodell | geplant | IM-Generator, Human-in-the-Middle-Workflow |
| M5 – Artefakt-Generierung | geplant | DDL, OpenAPI, DMN, XÖV |
| M6 – Produktionshärtung | geplant | K3s, Multi-Tenancy, BSI-Härtung |
| M7 – Systemerweiterungen | geplant | Automatische Register-Konfiguration via LLM |

---

## 6. Verzeichnisstruktur

```
NDI/
|-- .env                              # Passwörter + API-Keys (nicht im Git)
|-- .env.example                      # Vorlage ohne Secrets
|-- docker-compose.yml                # Infrastruktur-Stack
|
|-- infra/
|   `-- postgres/
|       |-- init/                     # DB-Migrationen (01-09)
|       `-- maintenance/              # Reset-Skripte
|
|-- services/
|   `-- ingest/                       # FastAPI-Service
|       |
|       |-- main.py
|       |-- requirements.txt
|       |
|       |-- nlp_config.yaml           # NLP, SVO, NER, LLM-Normtyp
|       |-- embedder_config.yaml      # Embedding-Modell
|       |-- abbrev_dict.yaml          # Abkürzungen + Synonyme
|       |-- chunker_config.yaml       # Token-Limits, Chunk-Typen
|       |-- docs.yaml                 # Dokument-Metadaten + ner_extensions
|       |-- llm_gateway_config.yaml   # LLM-Provider
|       |-- prompt_suite_index.yaml   # Prompt-Übersicht
|       |
|       |-- llm_gateway/
|       |   |-- gateway.py            # LLM-Adapter
|       |   |-- prompt_suite.py       # Prompt-Verwaltung
|       |   `-- schema_registry.py    # JSON-Ergebnisstrukturen
|       |
|       |-- prompt_suite/
|       |   |-- ps_normtyp/           # Batch-Normtyp-Klassifikation
|       |   |-- ps_pipeline/          # 7-Stufen-Pipeline (M3)
|       |   `-- ps_norm_logic/        # Wenn-Dann-Normlogik (M3)
|       |
|       |-- schema_registry/
|       |-- logs/                     # LLM Request/Response Logs
|       |
|       `-- app/
|           |-- api/routes/
|           |-- core/config.py
|           `-- services/
|               |-- nlp/
|               |   |-- nlp_service.py
|               |   |-- svo_extractor.py
|               |   |-- ner_extractor.py   # Zweischicht-NER
|               |   `-- llm_normtyp.py     # LLM-Batch-Klassifikation
|               |-- parser.py
|               |-- chunker.py
|               |-- embedder.py
|               |-- storage.py
|               `-- knowledge_graph.py
|
`-- Test/
    |-- run_test.py
    `-- docs/
```

---

## 7. Schnellstart

```bash
# 1. Infrastruktur starten
cd ~/reg-mo/ndi
docker compose up -d

# 2. FastAPI starten
cd services/ingest
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Testlauf
cd ~/reg-mo/ndi/Test
python run_test.py

# 4. Qualitaetsbericht
cd ~/reg-mo/ndi/services/ingest
python nlp_quality_report.py

# 5. Config-Manager
# http://localhost:8000/api/v1/config/ui
```

---

## 8. LLM-Konfiguration

| Provider | Modell | Kosten | Offline |
|---|---|---|---|
| `gemini` | gemini-2.5-flash | Free Tier | Nein |
| `gemini_flash_lite` | gemini-3.1-flash-lite | Free Tier, schneller | Nein |
| `openai` | gpt-4o | Kostenpflichtig | Nein |
| `anthropic` | claude-opus-4-5 | Kostenpflichtig | Nein |
| `ollama` | mistral:7b-instruct | Kostenlos | Ja |

```bash
# Provider wechseln:
# 1. llm_gateway_config.yaml -> active_provider aendern
curl -X POST http://localhost:8000/api/v1/llm/reload
curl http://localhost:8000/api/v1/llm/status
```

---

## 9. API-Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| `GET` | `/health` | Service-Status |
| `POST` | `/api/v1/ingest/document` | Dokument einspielen |
| `POST` | `/api/v1/nlp/run` | NLP-Job starten |
| `GET` | `/api/v1/nlp/status` | NLP-Job-Status |
| `GET` | `/api/v1/llm/status` | LLM-Gateway-Status |
| `POST` | `/api/v1/llm/complete` | LLM-Aufruf mit Prompt-Suite |
| `GET` | `/api/v1/llm/prompts` | Alle Prompts auflisten |
| `PUT` | `/api/v1/llm/prompts/{key}/system` | System-Prompt speichern |
| `PUT` | `/api/v1/llm/prompts/{key}/user` | User-Prompt speichern |
| `GET` | `/api/v1/kg/stats` | Knowledge-Graph-Statistik |
| `POST` | `/api/v1/kg/export` | KG nach Fuseki exportieren |
| `GET` | `/api/v1/config/ui` | Config-Manager (Browser) |
| `GET` | `/docs` | Swagger UI |

---

## 10. Qualitätsergebnisse M2

Gemessen auf dem NHundG (Niedersächsisches Hundegesetz):
94 Chunks, 288 SVOs, 503 NER-Entitäten.

### SVO-Qualität

| Kennzahl | Ergebnis | Ziel | Status |
|---|---|---|---|
| UNKNOWN-Rate | 16.0% | < 20% | erreicht |
| Mit Subjekt | 65.3% | > 80% | offen |
| Mit Objekt | ~52% | > 50% | erreicht |
| Ø SVO-Konfidenz | 0.823 | > 0.65 | erreicht |
| Ø Normtyp-Konfidenz | 0.718 | > 0.75 | knapp |

### Normtyp-Verteilung

| Normtyp | Anzahl | Anteil |
|---|---|---|
| MUST | 81 | 29.5% |
| MAY | 52 | 18.9% |
| UNKNOWN | 44 | 16.0% |
| COMPETENCE | 42 | 15.3% |
| STATUS | 14 | 5.1% |
| SCOPE | 14 | 5.1% |
| DEADLINE | 11 | 4.0% |
| EXCEPT | 6 | 2.2% |
| MUST_NOT | 5 | 1.8% |
| CHANGE | 4 | 1.5% |
| DEF | 2 | 0.7% |

### NLP-Abdeckung

| Kennzahl | Ergebnis | Ziel | Status |
|---|---|---|---|
| Chunks mit SVO | 91.5% | > 60% | erreicht |
| Chunks mit NER | 90.4% | > 70% | erreicht |
| Flair NER-Anteil | 21.5% | > 40% | offen |

Der niedrige Flair-Anteil und die Subjekt-Rate sind auf strukturelle
Eigenschaften der deutschen Rechtssprache zurückzuführen
(Passivkonstruktionen, Nominalstil, Querverweise) und für M3 ausreichend.

---

## Konfigurationsdokumentation

Siehe **CONFIG.md** in `services/ingest/` – vollständige Referenz aller
YAML-Dateien, Startbefehle und häufigen Anpassungen.

---

*Lizenz: Ausschliesslich Open-Source-Technologien | On-Premise-faehig | BSI-konform*
