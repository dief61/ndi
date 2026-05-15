-- infra/postgres/init/09_im_signals.sql
--
-- Ergänzt norm_chunks um im_signals JSONB:
-- Speichert IM-relevante Signale je Chunk (Entity-Kandidaten,
-- Attribut-Kandidaten, Persistenz-Signale) für M3 Context Assembly.
--
-- Struktur im_signals:
-- {
--   "entity_kandidaten":  ["Hundehalter", "Person"],
--   "attribut_kandidaten":["Sachkunde", "Name"],
--   "persistenz":         true,
--   "entity_def":         "Hundehalter",   -- bei DEF-Chunks
--   "relation_kandidaten":[{"von":"Hundehalter","zu":"Hund"}]
-- }

ALTER TABLE norm_chunks
  ADD COLUMN IF NOT EXISTS im_signals JSONB;

CREATE INDEX IF NOT EXISTS idx_norm_chunks_im_signals
  ON norm_chunks USING gin(im_signals)
  WHERE im_signals IS NOT NULL;

SELECT 'im_signals Spalte angelegt' AS status;
