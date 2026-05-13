-- NDI – Knowledge Graph Synchronisation
-- Datei: infra/postgres/init/07_knowledge_graph.sql
--
-- Tracking welche Chunks bereits in den Knowledge Graph exportiert wurden.
-- Ermöglicht inkrementellen Sync ohne vollständigen Re-Export.

CREATE TABLE IF NOT EXISTS kg_sync_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID NOT NULL REFERENCES norm_chunks(id) ON DELETE CASCADE,
    doc_id          UUID NOT NULL REFERENCES norm_documents(id) ON DELETE CASCADE,

    -- Status des Exports
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','exported','failed','outdated')),

    -- Anzahl exportierter Tripel
    triple_count    INT DEFAULT 0,

    -- Named Graph in Fuseki
    graph_uri       TEXT,

    -- Fehlerdetails
    error_message   TEXT,

    -- Zeitstempel
    exported_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kg_sync_status_idx
    ON kg_sync_log (status)
    WHERE status IN ('pending', 'failed');

-- UNIQUE Constraint für ON CONFLICT in _update_sync_log
ALTER TABLE kg_sync_log
    ADD CONSTRAINT IF NOT EXISTS kg_sync_chunk_unique UNIQUE (chunk_id);

CREATE INDEX IF NOT EXISTS kg_sync_chunk_idx
    ON kg_sync_log (chunk_id);

CREATE INDEX IF NOT EXISTS kg_sync_doc_idx
    ON kg_sync_log (doc_id);

-- View: Export-Statistik
CREATE OR REPLACE VIEW kg_sync_stats AS
SELECT
    nd.title,
    COUNT(ksl.id)                                               AS chunks_gesamt,
    SUM(CASE WHEN ksl.status = 'exported' THEN 1 ELSE 0 END)   AS exportiert,
    SUM(CASE WHEN ksl.status = 'pending'  THEN 1 ELSE 0 END)   AS ausstehend,
    SUM(CASE WHEN ksl.status = 'failed'   THEN 1 ELSE 0 END)   AS fehlgeschlagen,
    SUM(ksl.triple_count)                                       AS tripel_gesamt,
    MAX(ksl.exported_at)                                        AS letzter_export
FROM kg_sync_log ksl
JOIN norm_documents nd ON ksl.doc_id = nd.id
GROUP BY nd.title
ORDER BY nd.title;

SELECT 'kg_sync_log und kg_sync_stats angelegt' AS status;
