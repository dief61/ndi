#!/usr/bin/env python3
# /home/mdi/reg-mo/ndi/Test/run_test.py
#
# NDI Vollständiger Testlauf
#
# Ablauf:
#   1. PostgreSQL – alle Tabellen leeren
#   2. MinIO      – alle Buckets leeren
#   3. Verifizieren dass DB + MinIO leer sind
#   4. FastAPI Readiness prüfen
#   5. Alle PDFs aus Test/docs/ einspielen
#   6. Ergebnis prüfen (Chunks, Embeddings)
#   7. Alles in Log-Datei schreiben
#
# Aufruf:
#   python run_test.py
#   python run_test.py --docs-dir /anderer/pfad
#   python run_test.py --skip-reset   # Reset überspringen

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Pfade
# ─────────────────────────────────────────────────────────────────────────────
TEST_DIR    = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent
INGEST_DIR  = PROJECT_DIR / "services" / "ingest"
ENV_FILE    = PROJECT_DIR / ".env"

# Python-Path für Ingest-Service-Module
sys.path.insert(0, str(INGEST_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# .env laden
# ─────────────────────────────────────────────────────────────────────────────

def load_env(env_path: Path) -> dict:
    """Liest .env-Datei und gibt Key-Value-Dict zurück."""
    env = {}
    if not env_path.exists():
        raise FileNotFoundError(f".env nicht gefunden: {env_path}")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip()
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class TestLogger:
    """Schreibt gleichzeitig auf Konsole und in Log-Datei."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%m%d%H%M%S")
        log_path = log_dir / f"{ts}.log"
        self.file = open(log_path, "w", encoding="utf-8")
        self.log_path = log_path
        self._write(f"NDI Testlauf – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"Log-Datei: {log_path}")
        self._write("=" * 65)

    def _write(self, text: str = ""):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        print(line)
        self.file.write(line + "\n")
        self.file.flush()

    def info(self, text: str):
        self._write(f"  ✓  {text}")

    def error(self, text: str):
        self._write(f"  ✗  {text}")

    def warn(self, text: str):
        self._write(f"  ⚠  {text}")

    def section(self, title: str):
        self._write("")
        self._write(f"{'─'*65}")
        self._write(f"  {title}")
        self._write(f"{'─'*65}")

    def result(self, label: str, value, ok: bool = True):
        icon = "✓" if ok else "✗"
        self._write(f"  {icon}  {label:<35} {value}")

    def close(self, success: bool):
        self._write("")
        self._write("=" * 65)
        if success:
            self._write("  ✅  TESTLAUF ERFOLGREICH")
        else:
            self._write("  ❌  TESTLAUF FEHLGESCHLAGEN")
        self._write(f"  Log gespeichert: {self.log_path}")
        self._write("=" * 65)
        self.file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 1: PostgreSQL leeren
# ─────────────────────────────────────────────────────────────────────────────

async def reset_postgres(env: dict, log: TestLogger) -> bool:
    log.section("Schritt 1: PostgreSQL – alle Tabellen leeren")
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host=env.get("POSTGRES_HOST", "localhost"),
            port=int(env.get("POSTGRES_PORT", 5432)),
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            database=env["POSTGRES_DB"],
        )

        await conn.execute("""
            TRUNCATE TABLE
                svo_extractions,
                ner_entities,
                nlp_jobs,
                im_review_log,
                information_models,
                ingest_paket_jobs,
                ingest_pakete,
                ingest_jobs,
                norm_chunks,
                norm_documents
            RESTART IDENTITY CASCADE
        """)

        # Zeilenanzahl prüfen
        tables = [
            "norm_documents", "norm_chunks", "ingest_jobs",
            "ingest_pakete", "nlp_jobs", "svo_extractions", "ner_entities",
        ]
        all_empty = True
        for table in tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            ok = count == 0
            if not ok:
                all_empty = False
            log.result(table, f"{count} Zeilen", ok)

        await conn.close()

        if all_empty:
            log.info("PostgreSQL vollständig geleert")
        else:
            log.error("PostgreSQL: nicht alle Tabellen leer!")
        return all_empty

    except Exception as e:
        log.error(f"PostgreSQL-Fehler: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 2: MinIO leeren
# ─────────────────────────────────────────────────────────────────────────────

def reset_minio(env: dict, log: TestLogger) -> bool:
    log.section("Schritt 2: MinIO – alle Buckets leeren")
    try:
        from minio import Minio

        client = Minio(
            endpoint=f"localhost:{env.get('MINIO_PORT', '9000')}",
            access_key=env["MINIO_ROOT_USER"],
            secret_key=env["MINIO_ROOT_PASSWORD"],
            secure=False,
        )

        buckets = [
            env.get("MINIO_DEFAULT_BUCKET", "mnr-artefakte"),
            "mnr-dokumente",
            "mnr-informationsmodelle",
        ]

        total_deleted = 0
        for bucket in buckets:
            try:
                objects = list(client.list_objects(bucket, recursive=True))
                for obj in objects:
                    client.remove_object(bucket, obj.object_name)
                log.result(f"{bucket}", f"{len(objects)} Objekte gelöscht")
                total_deleted += len(objects)
            except Exception as e:
                log.warn(f"{bucket}: {e}")

        log.info(f"MinIO geleert – {total_deleted} Objekte gelöscht")
        return True

    except Exception as e:
        log.error(f"MinIO-Fehler: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 3: DB + MinIO Verifizierung
# ─────────────────────────────────────────────────────────────────────────────

async def verify_empty(env: dict, log: TestLogger) -> bool:
    log.section("Schritt 3: Verifizierung – DB + MinIO leer?")
    success = True

    try:
        import asyncpg
        conn = await asyncpg.connect(
            host=env.get("POSTGRES_HOST", "localhost"),
            port=int(env.get("POSTGRES_PORT", 5432)),
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            database=env["POSTGRES_DB"],
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM norm_documents"
        )
        await conn.close()
        ok = total == 0
        log.result("norm_documents leer", f"{total} Einträge", ok)
        if not ok:
            success = False
    except Exception as e:
        log.error(f"DB-Prüfung fehlgeschlagen: {e}")
        success = False

    try:
        from minio import Minio
        client = Minio(
            endpoint=f"localhost:{env.get('MINIO_PORT', '9000')}",
            access_key=env["MINIO_ROOT_USER"],
            secret_key=env["MINIO_ROOT_PASSWORD"],
            secure=False,
        )
        objects = list(client.list_objects("mnr-dokumente", recursive=True))
        ok = len(objects) == 0
        log.result("mnr-dokumente leer", f"{len(objects)} Objekte", ok)
        if not ok:
            success = False
    except Exception as e:
        log.error(f"MinIO-Prüfung fehlgeschlagen: {e}")
        success = False

    if success:
        log.info("Verifizierung erfolgreich – System ist leer")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 4: FastAPI Readiness
# ─────────────────────────────────────────────────────────────────────────────

async def check_fastapi(env: dict, log: TestLogger) -> bool:
    log.section("Schritt 4: FastAPI Readiness prüfen")
    try:
        import httpx
        port = env.get("FASTAPI_PORT", "8000")
        url  = f"http://localhost:{port}/health/ready"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)

        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "unbekannt")
            checks = data.get("checks", {})
            ok = status == "ready"
            log.result("FastAPI Status", status, ok)
            for service, s in checks.items():
                log.result(f"  └ {service}", s, s == "ok")
            if ok:
                log.info("FastAPI ist bereit")
            return ok
        else:
            log.error(f"FastAPI antwortet mit HTTP {resp.status_code}")
            return False

    except Exception as e:
        log.error(f"FastAPI nicht erreichbar: {e}")
        log.warn("Stelle sicher dass uvicorn läuft: uvicorn main:app --host 0.0.0.0 --port 8000")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 5: Dokumente einspielen
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_documents(
    docs_dir: Path,
    env: dict,
    log: TestLogger,
) -> list[dict]:
    log.section("Schritt 5: Dokumente einspielen")

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        log.error(f"Keine PDF-Dateien gefunden in: {docs_dir}")
        return []

    log.info(f"{len(pdf_files)} PDF-Dateien gefunden:")
    for f in pdf_files:
        log._write(f"       {f.name}")

    import httpx
    port    = env.get("FASTAPI_PORT", "8000")
    results = []

    for pdf in pdf_files:
        title = pdf.stem   # Dateiname ohne Suffix
        log._write(f"\n       Einspielen: {pdf.name} → Titel: '{title}'")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(pdf, "rb") as f:
                    resp = await client.post(
                        f"http://localhost:{port}/api/v1/ingest/document",
                        files={"file": (pdf.name, f, "application/pdf")},
                        data={
                            "source_type": "gesetz",
                            "title":       title,
                        },
                    )

            if resp.status_code == 200:
                data = resp.json()
                log.result(
                    f"{pdf.name}",
                    f"job_id={data['job_id'][:8]}... status={data['status']}"
                )
                results.append({
                    "filename": pdf.name,
                    "title":    title,
                    "job_id":   data["job_id"],
                    "doc_id":   data["doc_id"],
                    "status":   data["status"],
                })
            else:
                log.error(f"{pdf.name}: HTTP {resp.status_code} – {resp.text[:100]}")
                results.append({
                    "filename": pdf.name,
                    "title":    title,
                    "job_id":   None,
                    "doc_id":   None,
                    "status":   "error",
                })

        except Exception as e:
            log.error(f"{pdf.name}: {e}")
            results.append({
                "filename": pdf.name,
                "title":    title,
                "job_id":   None,
                "doc_id":   None,
                "status":   "error",
            })

    log.info(f"{len(results)} Dokumente eingereiht")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Auf Job-Abschluss warten
# ─────────────────────────────────────────────────────────────────────────────

async def wait_for_jobs(
    jobs: list[dict],
    env: dict,
    log: TestLogger,
    timeout_secs: int = 300,
) -> list[dict]:
    """Wartet bis alle Jobs abgeschlossen sind."""
    import httpx

    port     = env.get("FASTAPI_PORT", "8000")
    pending  = [j for j in jobs if j.get("job_id")]
    deadline = asyncio.get_event_loop().time() + timeout_secs

    log._write(f"\n       Warte auf Abschluss von {len(pending)} Jobs...")

    while pending and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(5)
        still_pending = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for job in pending:
                try:
                    resp = await client.get(
                        f"http://localhost:{port}/api/v1/ingest/status/{job['job_id']}"
                    )
                    data = resp.json()
                    status = data.get("status", "unknown")
                    job["status"]      = status
                    job["doc_class"]   = data.get("doc_class")
                    job["chunk_count"] = data.get("chunk_count", 0)

                    if status in ("done", "error"):
                        icon = "✓" if status == "done" else "✗"
                        log._write(
                            f"       {icon} {job['filename']:<35} "
                            f"status={status} "
                            f"klasse={job.get('doc_class','–')} "
                            f"chunks={job.get('chunk_count', 0)}"
                        )
                    else:
                        still_pending.append(job)

                except Exception:
                    still_pending.append(job)

        pending = still_pending
        if pending:
            log._write(f"       Noch ausstehend: {len(pending)} Jobs...")

    if pending:
        log.warn(f"Timeout – {len(pending)} Jobs noch nicht abgeschlossen")

    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 6: Ergebnis prüfen
# ─────────────────────────────────────────────────────────────────────────────

async def verify_results(
    jobs: list[dict],
    env: dict,
    log: TestLogger,
) -> bool:
    log.section("Schritt 6: Ergebnis prüfen")
    success = True

    try:
        import asyncpg
        conn = await asyncpg.connect(
            host=env.get("POSTGRES_HOST", "localhost"),
            port=int(env.get("POSTGRES_PORT", 5432)),
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            database=env["POSTGRES_DB"],
        )

        # Gesamt-Statistik
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM norm_documents")
        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM norm_chunks")
        emb_count = await conn.fetchval(
            "SELECT COUNT(*) FROM norm_chunks WHERE embedding IS NOT NULL"
        )

        log.result("Dokumente in DB", doc_count, doc_count == len(jobs))
        log.result("Chunks gesamt",   chunk_count, chunk_count > 0)
        log.result(
            "Chunks mit Embedding",
            f"{emb_count}/{chunk_count}",
            emb_count == chunk_count,
        )

        if doc_count != len(jobs):
            success = False
        if chunk_count == 0:
            success = False
        if emb_count != chunk_count:
            success = False

        # Pro Dokument
        log._write("\n       Pro Dokument:")
        rows = await conn.fetch("""
            SELECT nd.title, nd.source_type, nd.doc_id,
                   COUNT(nc.id) AS chunks,
                   SUM(CASE WHEN nc.embedding IS NOT NULL THEN 1 ELSE 0 END) AS mit_emb,
                   nc.doc_class
            FROM norm_documents nd
            LEFT JOIN norm_chunks nc ON nc.doc_id = nd.id
            GROUP BY nd.title, nd.source_type, nd.doc_id, nc.doc_class
            ORDER BY nd.title
        """)
        for r in rows:
            chunks   = r["chunks"]   or 0
            mit_emb  = r["mit_emb"]  or 0
            ok       = chunks > 0 and mit_emb == chunks
            log.result(
                f"  {r['title'][:30]}",
                f"Klasse={r['doc_class']} Chunks={chunks} Emb={mit_emb}/{chunks}",
                ok,
            )
            if not ok:
                success = False

        # MinIO prüfen
        from minio import Minio
        client = Minio(
            endpoint=f"localhost:{env.get('MINIO_PORT', '9000')}",
            access_key=env["MINIO_ROOT_USER"],
            secret_key=env["MINIO_ROOT_PASSWORD"],
            secure=False,
        )
        minio_count = len(list(client.list_objects(
            "mnr-dokumente", recursive=True
        )))
        ok = minio_count == len(jobs)
        log.result("Dateien in MinIO", f"{minio_count}", ok)
        if not ok:
            success = False

        # Job-Status Zusammenfassung
        done_count  = sum(1 for j in jobs if j.get("status") == "done")
        error_count = sum(1 for j in jobs if j.get("status") == "error")
        log._write("\n       Job-Zusammenfassung:")
        log.result("Jobs erfolgreich", done_count,  done_count  == len(jobs))
        log.result("Jobs fehlerhaft",  error_count, error_count == 0)

        if error_count > 0:
            success = False
            for j in jobs:
                if j.get("status") == "error":
                    log.error(f"  Fehler: {j['filename']}")

        await conn.close()

    except Exception as e:
        log.error(f"Ergebnisprüfung fehlgeschlagen: {e}")
        success = False

    return success


# ─────────────────────────────────────────────────────────────────────────────
# Hauptprogramm
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NDI Vollständiger Testlauf")
    parser.add_argument(
        "--docs-dir", type=str,
        default=str(TEST_DIR / "docs"),
        help="Verzeichnis mit Test-PDFs (Standard: Test/docs/)",
    )
    parser.add_argument(
        "--skip-reset", action="store_true",
        help="Reset von DB und MinIO überspringen",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Maximale Wartezeit für Jobs in Sekunden (Standard: 300)",
    )
    return parser.parse_args()


async def main():
    args     = parse_args()
    docs_dir = Path(args.docs_dir)
    log      = TestLogger(TEST_DIR)

    log._write(f"  Projekt:     {PROJECT_DIR}")
    log._write(f"  Docs-Dir:    {docs_dir}")
    log._write(f"  Skip-Reset:  {args.skip_reset}")

    # .env laden
    try:
        env = load_env(ENV_FILE)
        log.info(f".env geladen: {ENV_FILE}")
    except FileNotFoundError as e:
        log.error(str(e))
        log.close(False)
        return

    success = True

    # Schritt 1+2: Reset
    if not args.skip_reset:
        if not await reset_postgres(env, log):
            success = False
        if not reset_minio(env, log):
            success = False
    else:
        log.section("Reset übersprungen (--skip-reset)")

    # Schritt 3: Verifizierung
    if success and not args.skip_reset:
        if not await verify_empty(env, log):
            success = False

    # Schritt 4: FastAPI prüfen
    if success:
        if not await check_fastapi(env, log):
            log.close(False)
            return

    # Schritt 5: Dokumente einspielen
    if success:
        jobs = await ingest_documents(docs_dir, env, log)
        if not jobs:
            success = False
        else:
            # Auf Abschluss warten
            jobs = await wait_for_jobs(jobs, env, log, timeout_secs=args.timeout)

            # Schritt 6: Ergebnis prüfen
            if not await verify_results(jobs, env, log):
                success = False

    log.close(success)


if __name__ == "__main__":
    asyncio.run(main())
