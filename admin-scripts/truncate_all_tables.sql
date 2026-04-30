-- NDI – Alle Tabelleninhalte löschen (Struktur bleibt erhalten)
-- Datei: infra/postgres/maintenance/truncate_all_tables.sql
--
-- WARNUNG: Löscht alle Daten unwiderruflich!
-- Tabellen und Indizes bleiben erhalten.
-- Nur in Entwicklung verwenden.
--
-- Aufruf:
--   docker exec -i mnr-postgres psql -U mnr -d mnr_db \
--     < infra/postgres/maintenance/truncate_all_tables.sql

-- TRUNCATE mit CASCADE leert alle Tabellen auf einmal
-- und berücksichtigt FK-Abhängigkeiten automatisch.
-- RESTART IDENTITY setzt alle Sequenzen zurück.
TRUNCATE TABLE
    im_review_log,
    information_models,
    ingest_jobs,
    norm_chunks,
    norm_documents,
	ingest_pakete,
	ingest_paket_jobs
RESTART IDENTITY CASCADE;

-- Bestätigung
SELECT
    relname AS tabelle,
    n_live_tup AS zeilen
FROM pg_stat_user_tables
ORDER BY relname;
