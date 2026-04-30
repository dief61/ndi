-- NDI – Paket-Tabelle
-- Datei: infra/postgres/init/03_pakete.sql
--
-- Aufruf auf bestehender Instanz:
--   docker exec -i mnr-postgres psql -U mnr -d mnr_db \
--     < infra/postgres/init/03_pakete.sql

CREATE TABLE IF NOT EXISTS ingest_pakete (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paket_id        TEXT UNIQUE NOT NULL,
    version_id      TEXT NOT NULL,
    paket_name      TEXT NOT NULL,
    version         TEXT NOT NULL,
    manifest_hash   TEXT,
    dokument_ids    TEXT[] NOT NULL,          -- Liste der Dokument-IDs
    status          TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN (
                            'queued','processing','done','partial','error'
                        )),
    -- Statistik
    total_docs      INT NOT NULL DEFAULT 0,
    done_docs       INT NOT NULL DEFAULT 0,
    error_docs      INT NOT NULL DEFAULT 0,
    -- Zeitstempel
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

-- Verknüpfung Paket ↔ Job
CREATE TABLE IF NOT EXISTS ingest_paket_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paket_id    TEXT NOT NULL REFERENCES ingest_pakete(paket_id),
    job_id      TEXT NOT NULL REFERENCES ingest_jobs(job_id),
    doc_id      TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingest_pakete_paket_id_idx
    ON ingest_pakete (paket_id);
CREATE INDEX IF NOT EXISTS ingest_paket_jobs_paket_id_idx
    ON ingest_paket_jobs (paket_id);
CREATE INDEX IF NOT EXISTS ingest_paket_jobs_job_id_idx
    ON ingest_paket_jobs (job_id);
