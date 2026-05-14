-- infra/postgres/init/08_normtype_constraint.sql
-- Erweitert den norm_type CHECK-Constraint um SCOPE, CHANGE, STATUS

-- Alten Constraint entfernen
ALTER TABLE svo_extractions
  DROP CONSTRAINT IF EXISTS svo_extractions_norm_type_check;

-- Neuen Constraint mit allen Normtypen anlegen
ALTER TABLE svo_extractions
  ADD CONSTRAINT svo_extractions_norm_type_check
  CHECK (norm_type IN (
    'MUST', 'MAY', 'MUST_NOT', 'DEF', 'EXCEPT',
    'DEADLINE', 'COMPETENCE', 'SCOPE', 'CHANGE',
    'STATUS', 'UNKNOWN'
  ));

SELECT 'norm_type Constraint aktualisiert' AS status;
