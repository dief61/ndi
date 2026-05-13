# Alle Prädikate die UNKNOWN sind
docker exec -i mnr-postgres psql -U mnr -d mnr_db << 'SQL'
-- Welche Prädikate sind noch UNKNOWN?
SELECT s.predicate_lemma, s.predicate, COUNT(*) AS anzahl
FROM svo_extractions s
WHERE s.norm_type = 'UNKNOWN'
GROUP BY s.predicate_lemma, s.predicate
ORDER BY anzahl DESC
LIMIT 20;
SQL



# Alle FAQ-Fragen als "verworfen" markieren (Batch):
docker exec -i mnr-postgres psql -U mnr -d mnr_db << 'SQL'
UPDATE filtered_questions
SET reviewed = TRUE,
    decision = 'verwerfen',
    reviewer_note = 'FAQ-Fragen – kein normativer Inhalt'
WHERE doc_class IN ('B', 'C')
  AND reviewed = FALSE;
SQL

# Bestätigen:
python question_report.py --all


#Beispiele für SVOs ohne Objekt
docker exec -i mnr-postgres psql -U mnr -d mnr_db << 'SQL'
-- Beispiele für SVOs ohne Objekt
SELECT
    s.subject,
    s.predicate,
    s.sentence_text
FROM svo_extractions s
WHERE s.object IS NULL
  AND s.subject IS NOT NULL
ORDER BY random()
LIMIT 15;
SQL


docker exec -i mnr-postgres psql -U mnr -d mnr_db << 'SQL'
-- Vorkommen vs. unique
SELECT
    e.label,
    COUNT(*)                    AS vorkommen,
    COUNT(DISTINCT e.text)      AS unique_entitaeten
FROM ner_entities e
WHERE e.confidence > 0.7
GROUP BY e.label
ORDER BY vorkommen DESC;
SQL


















