-- NDI – Abkürzungs-Erweiterung für norm_chunks
-- Datei: infra/postgres/init/05_abbrev.sql
--
-- Fügt zwei neue Spalten zu norm_chunks hinzu:
--   content_original  → Originaltext aus Tika (unverändert)
--   abbrev_map        → Positions-Mapping original ↔ aufgelöst
--
-- Aufruf auf bestehender Instanz:
--   docker exec -i mnr-postgres psql -U mnr -d mnr_db \
--     < infra/postgres/init/05_abbrev.sql

-- content_original: Originaltext vor Abkürzungsauflösung
ALTER TABLE norm_chunks
    ADD COLUMN IF NOT EXISTS content_original TEXT;

-- abbrev_map: JSON-Array mit Ersetzungen
-- Format:
-- [
--   {
--     "abbrev":    "NHundG",
--     "resolved":  "Niedersächsisches Hundegesetz",
--     "orig_start": 11,
--     "orig_end":   17,
--     "res_start":  11,
--     "res_end":    43
--   }, ...
-- ]
ALTER TABLE norm_chunks
    ADD COLUMN IF NOT EXISTS abbrev_map JSONB;

-- Index für Abfragen auf abbrev_map
CREATE INDEX IF NOT EXISTS norm_chunks_abbrev_map_idx
    ON norm_chunks USING gin (abbrev_map)
    WHERE abbrev_map IS NOT NULL;

-- Bestätigung
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'norm_chunks'
  AND column_name IN ('content_original', 'abbrev_map')
ORDER BY column_name;
