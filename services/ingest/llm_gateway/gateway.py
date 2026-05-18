# services/ingest/llm_gateway/gateway.py
#
# LLM-Gateway – zentraler Einstiegspunkt für alle LLM-Aufrufe im MNR.
#
# Nutzung:
#   from llm_gateway.gateway import llm_gateway
#
#   result = await llm_gateway.complete(
#       system_prompt="...",
#       user_prompt="...",
#       schema=MyPydanticModel,   # optional – JSON-Validierung
#   )
#
# Provider wechseln: llm_gateway/config.yaml → active_provider ändern

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import structlog
import yaml
from pydantic import BaseModel

logger = structlog.get_logger()

_CONFIG_PATH = Path(__file__).parent.parent / "llm_gateway_config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenklasse
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    """Standardisiertes Ergebnis eines LLM-Aufrufs."""
    content:      str                  # Rohtext der Antwort
    parsed:       Optional[Any]        # Geparste Struktur (wenn schema angegeben)
    provider:     str                  # genutzter Provider
    model:        str                  # genutztes Modell
    input_tokens: int   = 0
    output_tokens: int  = 0
    dauer_ms:     int   = 0
    erfolg:       bool  = True
    fehler:       Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Adapter-Interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMAdapter(ABC):
    """Basis-Interface für alle LLM-Adapter."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        json_mode:     bool = False,
    ) -> LLMResult:
        """Führt einen LLM-Aufruf aus."""
        ...

    def _strip_json_fences(self, text: str) -> str:
        """Entfernt ```json ... ``` Markdown-Fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Erste und letzte Zeile (Fences) entfernen
            start = 1 if lines[0].startswith("```") else 0
            end   = -1 if lines[-1].strip() == "```" else len(lines)
            text  = "\n".join(lines[start:end]).strip()
        return text

    def _parse_json(self, text: str) -> Optional[Any]:
        """Parst JSON aus der LLM-Antwort."""
        try:
            return json.loads(self._strip_json_fences(text))
        except json.JSONDecodeError as e:
            logger.warning("JSON-Parse fehlgeschlagen", error=str(e),
                           text_start=text[:100])
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Gemini-Adapter
# ─────────────────────────────────────────────────────────────────────────────

class GeminiAdapter(LLMAdapter):
    """
    Adapter für Google Gemini – nutzt direkt die REST API
    (identisch mit dem funktionierenden curl-Aufruf).
    Endpoint: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # API-Key aus Umgebung oder direkt aus .env laden
        self._api_key = os.environ.get(cfg["api_key_env"])
        if not self._api_key:
            # Fallback: .env direkt lesen
            try:
                from pathlib import Path
                env_path = Path(__file__).parents[3] / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith(cfg["api_key_env"] + "="):
                            self._api_key = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass
        if not self._api_key:
            raise ValueError(
                f"API-Key nicht gefunden: {cfg['api_key_env']} in .env"
            )

    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        json_mode:     bool = False,
    ) -> LLMResult:
        import httpx
        t0      = time.monotonic()
        model   = self.cfg["model"]
        url     = f"{self._BASE_URL}/{model}:generateContent"

        # Payload – REST-Aufruf
        gen_config: dict = {
            "temperature":     self.cfg.get("temperature", 0.0),
            "maxOutputTokens": self.cfg.get("max_tokens", 8192),
        }
        # json_mode nur aktivieren wenn der Provider es zuverlässig unterstützt.
        # Bekanntes Problem: Gemini Flash-Lite ignoriert bei json_mode=True
        # den System-Prompt und liefert ein generisches Template.
        # Steuerung: llm_gateway_config.yaml → providers.X.json_mode_unterstuetzt
        json_mode_ok = self.cfg.get("json_mode_unterstuetzt", True)
        if json_mode and json_mode_ok:
            gen_config["responseMimeType"] = "application/json"
        # Hinweis: thinkingConfig wird in v1beta REST API nicht unterstützt.
        # Das Modell wählt Thinking-Tiefe automatisch basierend auf der Aufgabe.

        payload: dict = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"parts": [{"text": user_prompt}]}
            ],
            "generationConfig": gen_config,
        }

        headers = {
            "X-goog-api-key":  self._api_key,
            "Content-Type":    "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.cfg.get("timeout_sek", 60)
            ) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # Antwort extrahieren
            content = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            usage   = data.get("usageMetadata", {})
            dauer   = int((time.monotonic() - t0) * 1000)

            # parsed nur wenn json_mode UND Provider unterstützt es
            # Bei json_mode_ok=False: rohen Text zurückgeben, kein Parse
            do_parse = json_mode and json_mode_ok
            return LLMResult(
                content=content,
                parsed=self._parse_json(content) if do_parse else None,
                provider="gemini",
                model=model,
                input_tokens=usage.get("promptTokenCount",     0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                dauer_ms=dauer,
            )
        except Exception as e:
            return LLMResult(
                content="", parsed=None,
                provider="gemini", model=model,
                dauer_ms=int((time.monotonic()-t0)*1000),
                erfolg=False, fehler=str(e),
            )


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-Adapter (auch für Azure OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIAdapter(LLMAdapter):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from openai import AsyncOpenAI
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise ValueError(
                f"API-Key nicht gefunden: {cfg['api_key_env']} in .env"
            )
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        json_mode:     bool = False,
    ) -> LLMResult:
        t0 = time.monotonic()
        kwargs: dict = {
            "model":       self.cfg["model"],
            "max_tokens":  self.cfg.get("max_tokens", 8192),
            "temperature": self.cfg.get("temperature", 0.0),
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp    = await self._client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            dauer   = int((time.monotonic() - t0) * 1000)
            return LLMResult(
                content=content,
                parsed=self._parse_json(content) if json_mode else None,
                provider="openai",
                model=self.cfg["model"],
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                dauer_ms=dauer,
            )
        except Exception as e:
            return LLMResult(
                content="", parsed=None,
                provider="openai", model=self.cfg["model"],
                dauer_ms=int((time.monotonic()-t0)*1000),
                erfolg=False, fehler=str(e),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic-Adapter (Claude)
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicAdapter(LLMAdapter):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import anthropic
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise ValueError(
                f"API-Key nicht gefunden: {cfg['api_key_env']} in .env"
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._anthropic = anthropic

    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        json_mode:     bool = False,
    ) -> LLMResult:
        t0 = time.monotonic()
        try:
            msg = await self._client.messages.create(
                model=self.cfg["model"],
                max_tokens=self.cfg.get("max_tokens", 8192),
                temperature=self.cfg.get("temperature", 0.0),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = msg.content[0].text if msg.content else ""
            dauer   = int((time.monotonic() - t0) * 1000)
            return LLMResult(
                content=content,
                parsed=self._parse_json(content) if json_mode else None,
                provider="anthropic",
                model=self.cfg["model"],
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                dauer_ms=dauer,
            )
        except Exception as e:
            return LLMResult(
                content="", parsed=None,
                provider="anthropic", model=self.cfg["model"],
                dauer_ms=int((time.monotonic()-t0)*1000),
                erfolg=False, fehler=str(e),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Ollama-Adapter (On-Premise)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaAdapter(LLMAdapter):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import httpx
        self._base_url = cfg.get("base_url", "http://localhost:11434")
        self._timeout  = httpx.Timeout(cfg.get("timeout_sek", 120.0))

    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        json_mode:     bool = False,
    ) -> LLMResult:
        import httpx
        t0 = time.monotonic()
        payload = {
            "model":  self.cfg["model"],
            "stream": False,
            "options": {
                "temperature": self.cfg.get("temperature", 0.0),
                "num_predict": self.cfg.get("max_tokens", 4096),
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }
        if json_mode:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data    = resp.json()
                content = data.get("message", {}).get("content", "")
                dauer   = int((time.monotonic() - t0) * 1000)
                return LLMResult(
                    content=content,
                    parsed=self._parse_json(content) if json_mode else None,
                    provider="ollama",
                    model=self.cfg["model"],
                    input_tokens=data.get("prompt_eval_count", 0),
                    output_tokens=data.get("eval_count", 0),
                    dauer_ms=dauer,
                )
        except Exception as e:
            return LLMResult(
                content="", parsed=None,
                provider="ollama", model=self.cfg["model"],
                dauer_ms=int((time.monotonic()-t0)*1000),
                erfolg=False, fehler=str(e),
            )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-Gateway – Haupt-Klasse
# ─────────────────────────────────────────────────────────────────────────────

class LLMGateway:
    """
    Zentraler Gateway für alle LLM-Aufrufe im MNR.

    Provider wechseln: llm_gateway/config.yaml → active_provider

    Beispiel:
        result = await llm_gateway.complete(
            system_prompt=suite.get_system("ps_normtyp"),
            user_prompt=suite.render_user("ps_normtyp", chunk_text=text),
            json_mode=True,
        )
        if result.erfolg:
            data = result.parsed
    """

    _ADAPTER_MAP = {
        "gemini":            GeminiAdapter,   # Gemini 2.5 Flash
        "gemini_flash_lite": GeminiAdapter,   # Gemini 3.1 Flash-Lite (gleicher Adapter)
        "openai":            OpenAIAdapter,
        "anthropic":         AnthropicAdapter,
        "ollama":            OllamaAdapter,
    }

    def __init__(self, config_path: Path = None):
        self._config_path = config_path or _CONFIG_PATH
        self._adapter: Optional[LLMAdapter] = None
        self._cfg:     Optional[dict]       = None

    def _load(self) -> LLMAdapter:
        """Lädt Konfiguration und instanziiert Adapter (lazy)."""
        if self._adapter is not None:
            return self._adapter

        with open(self._config_path, encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)

        provider = full_cfg.get("active_provider", "ollama")
        prov_cfg = full_cfg.get("providers", {}).get(provider)
        if not prov_cfg:
            raise ValueError(f"Provider '{provider}' nicht in config.yaml")

        self._cfg     = full_cfg
        adapter_class = self._ADAPTER_MAP.get(provider)
        if not adapter_class:
            raise ValueError(f"Kein Adapter für Provider '{provider}'")

        self._adapter = adapter_class(prov_cfg)
        logger.info("LLM-Gateway initialisiert",
                    provider=provider,
                    model=prov_cfg.get("model"))
        return self._adapter

    def reload(self) -> None:
        """Adapter neu laden – nach config.yaml-Änderung aufrufen."""
        self._adapter = None
        self._cfg     = None

    @property
    def active_provider(self) -> str:
        """Gibt den aktiven Provider zurück."""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                return yaml.safe_load(f).get("active_provider", "unbekannt")
        except Exception:
            return "unbekannt"

    @property
    def active_model(self) -> str:
        """Gibt das aktive Modell zurück."""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            prov = cfg.get("active_provider", "")
            return cfg.get("providers", {}).get(prov, {}).get("model", "")
        except Exception:
            return "unbekannt"

    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        json_mode:     bool         = False,
        schema:        Type[BaseModel] = None,
    ) -> LLMResult:
        """
        Führt einen LLM-Aufruf aus.

        Args:
            system_prompt: System-Instruktion
            user_prompt:   Anfrage mit Daten
            json_mode:     True → JSON-Ausgabe erzwingen + parsen
            schema:        Pydantic-Modell für Validierung (optional)
        """
        adapter = self._load()
        cfg     = self._cfg or {}
        retry   = cfg.get("retry", {})
        max_v   = retry.get("max_versuche", 3)
        warte   = retry.get("wartezeit_sek", 2.0)

        letzter_fehler = None
        for versuch in range(1, max_v + 1):
            result = await adapter.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=json_mode,
            )

            if result.erfolg:
                # Pydantic-Validierung wenn Schema angegeben
                if schema and result.parsed:
                    try:
                        result.parsed = schema.model_validate(result.parsed)
                    except Exception as e:
                        logger.warning("Schema-Validierung fehlgeschlagen",
                                       error=str(e))
                logger.info(
                    "LLM-Aufruf erfolgreich",
                    provider=result.provider,
                    model=result.model,
                    tokens_in=result.input_tokens,
                    tokens_out=result.output_tokens,
                    dauer_ms=result.dauer_ms,
                )
                return result

            letzter_fehler = result.fehler
            logger.warning("LLM-Aufruf fehlgeschlagen",
                           versuch=versuch, fehler=letzter_fehler)
            if versuch < max_v:
                import asyncio
                await asyncio.sleep(warte)

        return LLMResult(
            content="", parsed=None,
            provider=self.active_provider,
            model=self.active_model,
            erfolg=False,
            fehler=f"Alle {max_v} Versuche fehlgeschlagen: {letzter_fehler}",
        )

    async def ping(self) -> bool:
        """Testet ob der aktive Provider erreichbar ist."""
        try:
            result = await self.complete(
                system_prompt="Antworte nur mit: OK",
                user_prompt="Ping",
            )
            return result.erfolg
        except Exception:
            return False


# Singleton – wird in der Anwendung importiert
llm_gateway = LLMGateway()
