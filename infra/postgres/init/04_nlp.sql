-- NDI – NLP-Schema (Meilenstein M2)
-- Datei: infra/postgres/init/04_nlp.sql
--
-- Aufruf auf bestehender Instanz:
--   docker exec -i mnr-postgres psql -U mnr -d mnr_db \
--     < infra/postgres/init/04_nlp.sql

-- ─────────────────────────────────────────────────────────────────────────────
-- NLP-Job-Tabelle
-- Jeder NLP-Lauf (auch Neustart) erzeugt einen neuen Job-Datensatz.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nlp_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          TEXT UNIQUE NOT NULL,
    doc_id          TEXT,                           -- NULL = alle Dokumente
    status          TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN (
                            'queued','running','done','error'
                        )),
    config_snapshot JSONB,                          -- verwendete Konfiguration
    -- Statistik
    chunks_total    INT DEFAULT 0,
    chunks_done     INT DEFAULT 0,
    svo_count       INT DEFAULT 0,
    ner_count       INT DEFAULT 0,
    error_message   TEXT,
    -- Zeitstempel
    started_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS nlp_jobs_job_id_idx ON nlp_jobs (job_id);
CREATE INDEX IF NOT EXISTS nlp_jobs_status_idx ON nlp_jobs (status);

-- ─────────────────────────────────────────────────────────────────────────────
-- SVO-Extraktionen
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS svo_extractions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID NOT NULL REFERENCES norm_chunks(id) ON DELETE CASCADE,
    doc_id          UUID NOT NULL REFERENCES norm_documents(id) ON DELETE CASCADE,
    nlp_job_id      TEXT REFERENCES nlp_jobs(job_id),

    -- SVO-Tripel
    subject         TEXT,
    subject_type    TEXT,           -- Akteur | Behörde | Person | Unbekannt
    predicate       TEXT,
    predicate_lemma TEXT,           -- Grundform des Verbs
    object          TEXT,
    object_type     TEXT,           -- Datenobjekt | Anforderung | Unbekannt
    context         TEXT,           -- Bezugsobjekt / Adverbiale

    -- Normtyp
    norm_type       TEXT
                        CHECK (norm_type IN (
                            'MUST','MAY','MUST_NOT','DEF',
                            'EXCEPT','DEADLINE','COMPETENCE','UNKNOWN'
                        )),
    norm_type_confidence NUMERIC(4,3) DEFAULT 0.0,

    -- Qualität
    confidence      NUMERIC(4,3) DEFAULT 0.0,
    sentence_text   TEXT,           -- Originalsatz

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS svo_chunk_id_idx ON svo_extractions (chunk_id);
CREATE INDEX IF NOT EXISTS svo_doc_id_idx   ON svo_extractions (doc_id);
CREATE INDEX IF NOT EXISTS svo_norm_type_idx ON svo_extractions (norm_type);

-- ─────────────────────────────────────────────────────────────────────────────
-- NER-Entitäten
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ner_entities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID NOT NULL REFERENCES norm_chunks(id) ON DELETE CASCADE,
    doc_id          UUID NOT NULL REFERENCES norm_documents(id) ON DELETE CASCADE,
    nlp_job_id      TEXT REFERENCES nlp_jobs(job_id),

    -- Entität
    text            TEXT NOT NULL,
    label           TEXT NOT NULL
                        CHECK (label IN (
                            'BEHÖRDE','ROLLE','DATENOBJEKT',
                            'FRIST','GESETZ','ORT','SONSTIGE'
                        )),
    start_char      INT,
    end_char        INT,
    confidence      NUMERIC(4,3) DEFAULT 0.0,
    source          TEXT DEFAULT 'rule',  -- rule | flair | spacy

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ner_chunk_id_idx ON ner_entities (chunk_id);
CREATE INDEX IF NOT EXISTS ner_doc_id_idx   ON ner_entities (doc_id);
CREATE INDEX IF NOT EXISTS ner_label_idx    ON ner_entities (label);
CREATE INDEX IF NOT EXISTS ner_text_idx     ON ner_entities USING gin (to_tsvector('german', text));

-- ─────────────────────────────────────────────────────────────────────────────
-- Bestätigung
-- ─────────────────────────────────────────────────────────────────────────────
SELECT tablename AS tabelle, 'angelegt' AS status
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('nlp_jobs','svo_extractions','ner_entities')
ORDER BY tablename;
