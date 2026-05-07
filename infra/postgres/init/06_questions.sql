-- NDI – Fragen-Log-Tabelle
-- Datei: infra/postgres/init/06_questions.sql
--
-- Speichert Fragen die beim Ingest herausgefiltert wurden.
-- Ermöglicht nachträgliche Kontrolle und Entscheidung.
--
-- Aufruf:
--   docker exec -i mnr-postgres psql -U mnr -d mnr_db \
--     < infra/postgres/init/06_questions.sql

CREATE TABLE IF NOT EXISTS filtered_questions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Herkunft
    doc_id          UUID NOT NULL REFERENCES norm_documents(id) ON DELETE CASCADE,
    doc_title       TEXT,
    doc_class       CHAR(1),                    -- B oder C
    ingest_job_id   TEXT,                       -- Referenz auf ingest_jobs

    -- Frage
    question_text   TEXT NOT NULL,              -- vollständiger Fragetext
    question_type   TEXT                        -- direkt | rhetorisch | liste
                        CHECK (question_type IN (
                            'direkt','rhetorisch','liste','unbekannt'
                        )),

    -- Kontext
    context_before  TEXT,                       -- Text vor der Frage
    context_after   TEXT,                       -- Text nach der Frage
    section_path    TEXT,                       -- Abschnitts-Referenz (Klasse B)
    chunk_position  INT,                        -- Position im Dokument

    -- Entscheidung (für manuelle Nachkontrolle)
    reviewed        BOOLEAN DEFAULT FALSE,
    decision        TEXT                        -- behalten | verwerfen | unklar
                        CHECK (decision IN ('behalten','verwerfen','unklar')),
    reviewer_note   TEXT,

    -- Zeitstempel
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS fq_doc_id_idx
    ON filtered_questions (doc_id);
CREATE INDEX IF NOT EXISTS fq_ingest_job_idx
    ON filtered_questions (ingest_job_id);
CREATE INDEX IF NOT EXISTS fq_reviewed_idx
    ON filtered_questions (reviewed)
    WHERE reviewed = FALSE;

-- Bestätigung
SELECT 'filtered_questions angelegt' AS status;
