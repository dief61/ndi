# services/ingest/llm_gateway/prompt_suite.py
#
# Prompt-Suite – lädt und rendert System- und User-Prompts.
#
# Verzeichnisstruktur:
#   prompt_suite/
#     index.yaml          – Übersicht aller Prompts
#     ps_normtyp/
#       system.txt        – System-Prompt für Normtyp-Klassifikation
#       user.txt          – User-Prompt Template mit {{variablen}}
#     ps_pipeline/
#       system.txt        – dein claud-System-Prompt.txt
#       user.txt          – dein Prompt-Pipeline.txt
#     ps_svo_enrich/
#       system.txt
#       user.txt
#
# Nutzung:
#   suite = PromptSuite()
#   system = suite.get_system("ps_normtyp")
#   user   = suite.render_user("ps_normtyp", chunk_text="...", norm_ref="§ 3")

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger()

# Prompt-Texte: services/ingest/prompt_suite/{key}/system.txt + user.txt
_SUITE_ROOT = Path(__file__).parent.parent / "prompt_suite"
# Index: services/ingest/prompt_suite_index.yaml (neben den anderen YAMLs)
_INDEX_PATH = Path(__file__).parent.parent / "prompt_suite_index.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptMeta:
    """Metadaten eines Prompt-Paares aus index.yaml."""
    key:         str
    name:        str
    beschreibung: str
    version:     str    = "1.0"
    provider:    str    = "alle"   # alle | gemini | openai | anthropic | ollama
    schema:      Optional[str] = None  # Referenz auf schema_registry
    tags:        list[str] = field(default_factory=list)


@dataclass
class PromptPair:
    """System- und User-Prompt eines Prompt-Paares."""
    meta:          PromptMeta
    system_prompt: str
    user_template: str

    def render(self, **kwargs) -> str:
        """
        Rendert den User-Prompt durch Ersetzen von {{variablen}}.

        Beispiel:
            pair.render(chunk_text="...", norm_ref="§ 3")
            → "Analysiere folgenden Text:\n...\nNormreferenz: § 3"
        """
        result = self.user_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value or ""))
        # Nicht ersetzte Variablen mit Leerstring füllen
        result = re.sub(r'\{\{[^}]+\}\}', '', result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-Suite
# ─────────────────────────────────────────────────────────────────────────────

class PromptSuite:
    """
    Verwaltet alle Prompt-Paare der MNR-Anwendung.

    Lädt Prompts lazy aus dem Dateisystem.
    Änderungen an .txt-Dateien wirken ohne Neustart.
    """

    def __init__(self, suite_root: Path = None):
        self._root  = suite_root or _SUITE_ROOT
        self._cache: dict[str, PromptPair] = {}

    # ── Laden ─────────────────────────────────────────────────────────────────

    def _load_index(self) -> list[PromptMeta]:
        """
        Lädt prompt_suite_index.yaml aus services/ingest/
        (neben nlp_config.yaml, docs.yaml etc.).
        Fallback: index.yaml im prompt_suite/-Verzeichnis.
        """
        candidates = [_INDEX_PATH, self._root / "index.yaml"]
        index_path = next((p for p in candidates if p.exists()), None)
        if not index_path:
            return []
        with open(index_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return [
            PromptMeta(
                key=entry.get("key", ""),
                name=entry.get("name", ""),
                beschreibung=entry.get("beschreibung", ""),
                version=entry.get("version", "1.0"),
                provider=entry.get("provider", "alle"),
                schema=entry.get("schema"),
                tags=entry.get("tags", []),
            )
            for entry in data.get("prompts", [])
        ]

    def _load_pair(self, key: str) -> PromptPair:
        """Lädt ein Prompt-Paar aus dem Verzeichnis."""
        prompt_dir = self._root / key
        if not prompt_dir.exists():
            raise FileNotFoundError(
                f"Prompt-Verzeichnis nicht gefunden: {prompt_dir}"
            )

        system_path = prompt_dir / "system.txt"
        user_path   = prompt_dir / "user.txt"

        system_prompt = (
            system_path.read_text(encoding="utf-8")
            if system_path.exists() else ""
        )
        user_template = (
            user_path.read_text(encoding="utf-8")
            if user_path.exists() else ""
        )

        # Meta aus index.yaml oder Fallback
        index   = self._load_index()
        meta    = next((m for m in index if m.key == key), None)
        if not meta:
            meta = PromptMeta(key=key, name=key, beschreibung="")

        return PromptPair(
            meta=meta,
            system_prompt=system_prompt,
            user_template=user_template,
        )

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def get(self, key: str, force_reload: bool = False) -> PromptPair:
        """Gibt ein Prompt-Paar zurück (gecacht)."""
        if force_reload or key not in self._cache:
            self._cache[key] = self._load_pair(key)
        return self._cache[key]

    def get_system(self, key: str) -> str:
        """Gibt den System-Prompt zurück."""
        return self.get(key).system_prompt

    def render_user(self, key: str, **kwargs) -> str:
        """Rendert den User-Prompt mit den angegebenen Variablen."""
        return self.get(key).render(**kwargs)

    def list_prompts(self) -> list[PromptMeta]:
        """Gibt alle bekannten Prompt-Metadaten zurück."""
        return self._load_index()

    def exists(self, key: str) -> bool:
        """Prüft ob ein Prompt-Paar existiert."""
        return (self._root / key).exists()

    def reload_all(self) -> None:
        """Leert den Cache – alle Prompts werden neu geladen."""
        self._cache.clear()

    def get_variables(self, key: str) -> list[str]:
        """Gibt alle Variablen-Namen im User-Template zurück."""
        template = self.get(key).user_template
        return re.findall(r'\{\{(\w+)\}\}', template)


# Singleton
prompt_suite = PromptSuite()
