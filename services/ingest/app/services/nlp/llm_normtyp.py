# services/ingest/app/services/nlp/llm_normtyp.py
#
# LLMNormtypClassifier: Klassifiziert UNKNOWN-SVOs via LLM-Gateway.
#
# Strategie: Batch-Prompt
#   batch_size SVOs werden in EINEN Prompt verpackt → 1 API-Call pro Batch.
#   Kein Rate-Limiting-Problem mehr (20 SVOs = 1 Call statt 20 Calls).
#
# Konfiguration (nlp_config.yaml → llm_normtyp):
#   enabled:         true
#   batch_size:      20     # SVOs pro Prompt/API-Call
#   min_confidence:  0.75
#   test_max_batches: 3     # Max. 3 Prompts → max. 60 LLM-klassifizierte SVOs
#   delay_sek:       1.0    # Pause zwischen Batch-Calls
#   log_prompts:     true   # Prompts + Antworten in Logfile schreiben

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger()

_NLP_CFG_PATH  = Path(__file__).parents[3] / "nlp_config.yaml"
_LOG_DIR       = Path(__file__).parents[3] / "logs"


# ─────────────────────────────────────────────────────────────────────────────
# Datenklasse
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormtypResult:
    norm_type:   str
    konfidenz:   float
    begruendung: str
    via_llm:     bool = True


# ─────────────────────────────────────────────────────────────────────────────
# LLM-Logger
# ─────────────────────────────────────────────────────────────────────────────

class LLMLogger:
    """
    Schreibt Requests und Responses in getrennte Logdateien.

    Dateien:
        logs/llm_request_DATUM_ZEIT.log   – alle Prompts (System + User)
        logs/llm_response_DATUM_ZEIT.log  – alle Antworten + Fehler

    Beide Dateien verwenden dieselbe fortlaufende Nummer (#001, #002, ...)
    damit Request und Response eindeutig zugeordnet werden können.
    """

    def __init__(self, log_dir: Path = None):
        self._dir = log_dir or _LOG_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._req_path  = self._dir / f"llm_request_{ts}.log"
        self._resp_path = self._dir / f"llm_response_{ts}.log"
        self._counter   = 0
        # Header in beide Dateien schreiben
        start_ts = datetime.now(timezone.utc).astimezone().strftime(
            "%d.%m.%Y %H:%M:%S")
        self._write(self._req_path,
            f"# LLM Request-Log  |  gestartet: {start_ts}\n"
            f"# Jede Anfrage ist mit #NNN nummeriert.\n"
            f"# Zugehörige Antwort: llm_response_{ts}.log #NNN\n")
        self._write(self._resp_path,
            f"# LLM Response-Log  |  gestartet: {start_ts}\n"
            f"# Jede Antwort ist mit #NNN nummeriert.\n"
            f"# Zugehöriger Request: llm_request_{ts}.log #NNN\n")

    # ── interne Hilfsmethoden ──────────────────────────────────────────────

    @staticmethod
    def _write(path: Path, text: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def _header(nr: int, ts: str, label: str, extra: str = "") -> str:
        sep = "=" * 72
        return (
            f"\n{sep}\n"
            f"  #{nr:04d}  |  {ts}  |  {label}"
            + (f"  |  {extra}" if extra else "")
            + f"\n{sep}\n"
        )

    # ── öffentliche API ────────────────────────────────────────────────────

    def log_request(self, system_prompt: str, user_prompt: str) -> int:
        """
        Schreibt einen Request ins Request-Log.
        Gibt die Nummer zurück – mit dieser Nummer log_response() aufrufen.
        """
        self._counter += 1
        nr  = self._counter
        ts  = datetime.now(timezone.utc).astimezone().strftime(
            "%d.%m.%Y %H:%M:%S")

        text = (
            self._header(nr, ts, "REQUEST")
            + "\n── SYSTEM ───────────────────────────────────────────────────────────\n"
            + system_prompt.strip()
            + "\n\n── USER ─────────────────────────────────────────────────────────────\n"
            + user_prompt.strip()
            + "\n"
        )
        self._write(self._req_path, text)
        return nr

    def log_response(
        self,
        nr:       int,
        response: str,
        parsed:   Optional[list],
        dauer_ms: int,
        erfolg:   bool,
        fehler:   Optional[str] = None,
    ) -> None:
        """
        Schreibt eine Response ins Response-Log.
        nr muss der Rückgabewert des zugehörigen log_request() sein.
        Fehler werden ebenfalls ins Response-Log geschrieben.
        """
        ts     = datetime.now(timezone.utc).astimezone().strftime(
            "%d.%m.%Y %H:%M:%S")
        status = "✅ OK" if erfolg else "❌ FEHLER"
        extra  = f"{dauer_ms}ms  |  {status}"

        text = self._header(nr, ts, "RESPONSE", extra)

        if not erfolg and fehler:
            text += (
                "── FEHLER ───────────────────────────────────────────────────────────\n"
                + fehler.strip()
                + "\n"
            )
        else:
            text += (
                "── ANTWORT (RAW) ────────────────────────────────────────────────────\n"
                + (response or "").strip()
                + "\n"
            )
            if parsed:
                text += (
                    "\n── GEPARST ──────────────────────────────────────────────────────────\n"
                    + json.dumps(parsed, ensure_ascii=False, indent=2)
                    + "\n"
                )

        self._write(self._resp_path, text)

    @property
    def request_path(self) -> Path:
        return self._req_path

    @property
    def response_path(self) -> Path:
        return self._resp_path


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class LLMNormtypClassifier:
    """
    Klassifiziert UNKNOWN-SVOs via LLM-Gateway.

    Batch-Strategie:
      - batch_size SVOs werden als nummerierte Liste in EINEN Prompt verpackt
      - 1 API-Call pro Batch
      - Das LLM gibt ein JSON-Array mit einem Ergebnis je SVO zurück
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._cfg_path = config_path or _NLP_CFG_PATH
        self._gateway  = None
        self._suite    = None
        self._llm_log: Optional[LLMLogger] = None

    def _load_config(self) -> dict:
        try:
            with open(self._cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("llm_normtyp", {})
        except Exception:
            return {}

    def _get_gateway(self):
        if self._gateway is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[3]))
            from llm_gateway.gateway import LLMGateway
            self._gateway = LLMGateway()
        return self._gateway

    def _get_suite(self):
        if self._suite is None:
            from llm_gateway.prompt_suite import PromptSuite
            self._suite = PromptSuite()
        return self._suite

    def _get_logger(self, enabled: bool) -> Optional[LLMLogger]:
        if not enabled:
            return None
        if self._llm_log is None:
            self._llm_log = LLMLogger()
            logger.info(
                "LLM-Logfiles",
                requests =str(self._llm_log.request_path),
                responses=str(self._llm_log.response_path),
            )
        return self._llm_log

    # ── Batch-Prompt aufbauen ─────────────────────────────────────────────────

    def _build_batch_prompt(self, svos: list[dict]) -> tuple[str, list[str]]:
        """
        Verpackt SVOs als nummerierte Liste in den User-Prompt.

        Returns:
            (user_prompt_text, id_liste)
        """
        lines = []
        ids   = []
        for i, svo in enumerate(svos):
            svo_id = f"SVO-{i+1:03d}"
            ids.append(svo_id)
            lines.append(
                f"{svo_id}:\n"
                f"  Normreferenz: {svo.get('norm_reference', '(unbekannt)')}\n"
                f"  Prädikat:     {svo.get('predicate', '')}\n"
                f"  Text:         {svo.get('sentence_text', '')}"
            )

        svo_list = "\n\n".join(lines)
        return svo_list, ids

    # ── Antwort parsen ────────────────────────────────────────────────────────

    @staticmethod
    def _repair_json_array(text: str) -> str:
        """
        Repariert ein abgeschnittenes JSON-Array.
        Findet das letzte vollständige Objekt und schließt das Array.
        """
        import re
        # Suche alle vollständig geschlossenen Objekte { ... }
        # Strategie: Zähle Klammern und finde letztes vollständiges Objekt
        depth      = 0
        last_close = -1
        in_string  = False
        escape     = False

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last_close = i

        if last_close < 0:
            return ""

        # Array bis zum letzten vollständigen Objekt + schließende Klammer
        repaired = text[:last_close + 1].rstrip().rstrip(',') + "\n]"
        # Sicherstellen dass es mit [ beginnt
        if not repaired.lstrip().startswith('['):
            repaired = '[' + repaired
        return repaired

    def _parse_batch_response(
        self,
        raw:       str,
        ids:       list[str],
        min_conf:  float,
    ) -> list[NormtypResult]:
        """
        Parst die JSON-Array-Antwort des LLM.
        Gibt für jede ID ein NormtypResult zurück.
        Fehlerfall: UNKNOWN.
        """
        VALID = {
            "MUST", "MAY", "MUST_NOT", "DEF", "EXCEPT",
            "DEADLINE", "COMPETENCE", "SCOPE", "CHANGE",
            "STATUS", "UNKNOWN",
        }
        KONF_MAP = {"hoch": 0.9, "mittel": 0.7, "niedrig": 0.5}

        # JSON aus Antwort extrahieren
        text = raw.strip()
        # Markdown-Fences entfernen
        if "```" in text:
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
            if m:
                text = m.group(1).strip()

        # ── JSON parsen mit Repair-Logik ─────────────────────────────────
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Versuch 1: Abgeschnittenes Array reparieren
            # Suche letztes vollständiges Objekt und schließe Array
            repaired = self._repair_json_array(text)
            if repaired:
                try:
                    data = json.loads(repaired)
                    logger.info(
                        "JSON-Array repariert",
                        original_len=len(text),
                        repaired_len=len(repaired),
                        objekte=len(data) if isinstance(data, list) else 0,
                    )
                except json.JSONDecodeError:
                    pass

            if data is None:
                logger.warning(
                    "Batch-JSON-Parse fehlgeschlagen",
                    raw_start=raw[:100],
                    error=str(e),
                )
                return [
                    NormtypResult("UNKNOWN", 0.0, "JSON-Parse-Fehler", True)
                    for _ in ids
                ]

        # Array normalisieren
        if isinstance(data, dict):
            # Manchmal gibt das LLM {results: [...]} zurück
            data = data.get("results", data.get("items", [data]))

        if not isinstance(data, list):
            return [
                NormtypResult("UNKNOWN", 0.0, "Keine Liste", True)
                for _ in ids
            ]

        results = []
        for i, expected_id in enumerate(ids):
            # Eintrag mit passender ID suchen (oder Position nutzen)
            entry = None
            if i < len(data):
                item = data[i]
                if isinstance(item, dict):
                    # ID-Match oder Positions-Fallback
                    if item.get("id") == expected_id or True:
                        entry = item

            if not entry:
                results.append(
                    NormtypResult("UNKNOWN", 0.0, "Kein Eintrag", True)
                )
                continue

            norm_type = str(entry.get("norm_type", "UNKNOWN")).upper().strip()
            konf_str  = str(entry.get("konfidenz", "niedrig")).lower()
            begr      = str(entry.get("begruendung", ""))[:100]
            konfidenz = KONF_MAP.get(konf_str, 0.5)

            if norm_type not in VALID:
                norm_type = "UNKNOWN"
                konfidenz = 0.0

            if konfidenz < min_conf:
                norm_type = "UNKNOWN"

            results.append(NormtypResult(norm_type, konfidenz, begr, True))

        return results

    # ── Öffentliche API ───────────────────────────────────────────────────────

    async def classify_batch(
        self,
        svos: list[dict],
    ) -> list[NormtypResult]:
        """
        Klassifiziert SVOs via Batch-Prompt.

        Ablauf:
          1. SVOs in Gruppen à batch_size aufteilen
          2. Je Gruppe: 1 Prompt → 1 API-Call → Array-Antwort parsen
          3. test_max_batches begrenzt die Anzahl der API-Calls
          4. Prompt + Antwort ins Logfile schreiben

        Args:
            svos: Liste von Dicts mit keys:
                  predicate, sentence_text, norm_reference

        Returns:
            Liste von NormtypResult – gleiche Reihenfolge wie Input.
        """
        cfg          = self._load_config()
        enabled      = cfg.get("enabled", True)
        batch_size   = cfg.get("batch_size", 20)
        min_conf     = cfg.get("min_confidence", 0.75)
        max_batches  = cfg.get("test_max_batches", 0)
        delay_sek    = cfg.get("delay_sek", 1.0)
        log_prompts  = cfg.get("log_prompts", True)

        if not enabled or not svos:
            return [
                NormtypResult("UNKNOWN", 0.0, "LLM deaktiviert", True)
                for _ in svos
            ]

        gw      = self._get_gateway()
        suite   = self._get_suite()
        llm_log = self._get_logger(log_prompts)

        if not suite.exists("ps_normtyp"):
            logger.warning("Prompt 'ps_normtyp' nicht gefunden")
            return [
                NormtypResult("UNKNOWN", 0.0, "Prompt fehlt", True)
                for _ in svos
            ]

        system  = suite.get_system("ps_normtyp")
        results = []
        batch_nr = 0

        for i in range(0, len(svos), batch_size):

            # Test-Budget prüfen
            if max_batches > 0 and batch_nr >= max_batches:
                remaining = len(svos) - i
                logger.info(
                    "Test-Budget erreicht",
                    max_batches=max_batches,
                    uebersprungen=remaining,
                )
                results.extend([
                    NormtypResult(
                        "UNKNOWN", 0.0,
                        f"Test-Budget ({max_batches} Batches) erreicht",
                        True,
                    )
                    for _ in svos[i:]
                ])
                break

            batch    = svos[i:i + batch_size]
            batch_nr += 1

            # Fix: Zu kurze oder Artefakt-SVOs vor LLM filtern
            # Diese bekommen direkt UNKNOWN – kein API-Call
            filtered_batch  = []
            skipped_indices = []   # Positionen die übersprungen werden
            skip_results    = {}   # index → NormtypResult

            for j, svo in enumerate(batch):
                text = (svo.get("sentence_text") or "").strip()
                pred = (svo.get("predicate") or "").strip()

                # Zu kurz
                if len(text) < 15:
                    skip_results[j] = NormtypResult(
                        "UNKNOWN", 0.0, "Text zu kurz", True)
                    skipped_indices.append(j)
                # Prädikat ist Großbuchstaben-Wort (Artefakt wie "Transponder",
                # "Hundegesetz", "Halten")
                elif pred and pred[0].isupper() and pred.isalpha() and len(pred) > 3:
                    skip_results[j] = NormtypResult(
                        "UNKNOWN", 0.0, "Prädikat ist Substantiv", True)
                    skipped_indices.append(j)
                else:
                    filtered_batch.append((j, svo))

            # Wenn alle übersprungen → direkt weiter
            if not filtered_batch:
                results.extend([
                    skip_results.get(j,
                        NormtypResult("UNKNOWN", 0.0, "", True))
                    for j in range(len(batch))
                ])
                batch_counter += 1
                continue

            # Nur die gefilterten SVOs an LLM schicken
            llm_svos = [svo for _, svo in filtered_batch]

            # Batch-Prompt aufbauen
            svo_list, ids = self._build_batch_prompt(llm_svos)
            user_prompt   = suite.render_user(
                "ps_normtyp",
                count=len(batch),
                svo_list=svo_list,
            )

            skipped = len(batch) - len(filtered_batch)
            logger.info(
                "LLM-Batch-Prompt",
                batch=batch_nr,
                max=max_batches or "∞",
                svos_gesamt=len(batch),
                svos_an_llm=len(filtered_batch),
                svos_gefiltert=skipped,
            )

            # ── Request loggen (vor dem API-Call) ─────────────────────
            req_nr = None
            if llm_log:
                req_nr = llm_log.log_request(
                    system_prompt=system,
                    user_prompt=user_prompt,
                )

            # ── API-Call ──────────────────────────────────────────────────
            result = await gw.complete(
                system_prompt=system,
                user_prompt=user_prompt,
                json_mode=True,
            )

            # ── Response loggen (nach dem API-Call) ───────────────────────
            if llm_log and req_nr is not None:
                llm_log.log_response(
                    nr=req_nr,
                    response=result.content,
                    parsed=result.parsed,
                    dauer_ms=result.dauer_ms,
                    erfolg=result.erfolg,
                    fehler=result.fehler,
                )

            if not result.erfolg:
                logger.warning("Batch-Aufruf fehlgeschlagen",
                               batch=batch_nr, fehler=result.fehler)
                # Fehlerfall: alle (auch nicht-gefilterte) auf UNKNOWN
                err_results = []
                for j in range(len(batch)):
                    if j in skip_results:
                        err_results.append(skip_results[j])
                    else:
                        err_results.append(NormtypResult(
                            "UNKNOWN", 0.0,
                            f"API-Fehler: {result.fehler}", True))
                results.extend(err_results)
            else:
                llm_results = self._parse_batch_response(
                    result.content, ids, min_conf
                )

                # LLM-Ergebnisse und Skipped-Ergebnisse in
                # ursprüngliche Reihenfolge zusammenführen
                batch_results = []
                llm_idx = 0
                for j in range(len(batch)):
                    if j in skip_results:
                        batch_results.append(skip_results[j])
                    else:
                        if llm_idx < len(llm_results):
                            batch_results.append(llm_results[llm_idx])
                            llm_idx += 1
                        else:
                            batch_results.append(
                                NormtypResult("UNKNOWN", 0.0, "", True))
                results.extend(batch_results)

                # Statistik loggen
                known = sum(1 for r in batch_results if r.norm_type != "UNKNOWN")
                logger.info(
                    "Batch-Ergebnis",
                    batch=batch_nr,
                    svos=len(batch),
                    klassifiziert=known,
                    unknown=len(batch)-known,
                    tokens_in=result.input_tokens,
                    tokens_out=result.output_tokens,
                    dauer_ms=result.dauer_ms,
                )

            # Pause zwischen Batches
            if i + batch_size < len(svos) and delay_sek > 0:
                await asyncio.sleep(delay_sek)

        if llm_log:
            logger.info(
                "LLM-Logs gespeichert",
                requests =str(llm_log.request_path),
                responses=str(llm_log.response_path),
                anzahl   =llm_log._counter,
            )

        return results
