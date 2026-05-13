# services/ingest/llm_gateway/schema_registry.py
#
# Schema-Registry – verwaltet JSON-Ergebnisstrukturen für LLM-Ausgaben.
#
# Verzeichnisstruktur:
#   schema_registry/
#     index.yaml
#     svo_triple.json       – einfache SVO-Struktur
#     norm_logic.json       – Wenn-Dann-Normlogik
#     full_pipeline.json    – vollständiges Pipeline-Schema
#     validation.json       – Validierungsstruktur
#
# Nutzung:
#   registry = SchemaRegistry()
#   schema   = registry.get("norm_logic")
#   example  = registry.get_example("svo_triple")

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml

logger = structlog.get_logger()

_REGISTRY_ROOT = Path(__file__).parent.parent / "schema_registry"


@dataclass
class SchemaMeta:
    key:         str
    name:        str
    beschreibung: str
    version:     str = "1.0"
    tags:        list[str] = field(default_factory=list)


@dataclass
class Schema:
    meta:       SchemaMeta
    definition: dict           # JSON-Schema-Definition
    example:    Optional[dict] # Beispiel-Ausgabe


class SchemaRegistry:
    """
    Verwaltet JSON-Schemas für LLM-Ausgaben.

    Schemas liegen als .json-Dateien im schema_registry/-Verzeichnis.
    Optionale .example.json-Dateien enthalten Beispiel-Ausgaben.
    """

    def __init__(self, registry_root: Path = None):
        self._root  = registry_root or _REGISTRY_ROOT
        self._cache: dict[str, Schema] = {}

    def _load_index(self) -> list[SchemaMeta]:
        index_path = self._root / "index.yaml"
        if not index_path.exists():
            return []
        with open(index_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return [
            SchemaMeta(
                key=e.get("key",""),
                name=e.get("name",""),
                beschreibung=e.get("beschreibung",""),
                version=e.get("version","1.0"),
                tags=e.get("tags",[]),
            )
            for e in data.get("schemas",[])
        ]

    def _load_schema(self, key: str) -> Schema:
        schema_path  = self._root / f"{key}.json"
        example_path = self._root / f"{key}.example.json"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema nicht gefunden: {schema_path}")

        with open(schema_path, encoding="utf-8") as f:
            definition = json.load(f)

        example = None
        if example_path.exists():
            with open(example_path, encoding="utf-8") as f:
                example = json.load(f)

        index = self._load_index()
        meta  = next((m for m in index if m.key == key), None)
        if not meta:
            meta = SchemaMeta(key=key, name=key, beschreibung="")

        return Schema(meta=meta, definition=definition, example=example)

    def get(self, key: str, force_reload: bool = False) -> Schema:
        """Gibt ein Schema zurück."""
        if force_reload or key not in self._cache:
            self._cache[key] = self._load_schema(key)
        return self._cache[key]

    def get_definition(self, key: str) -> dict:
        return self.get(key).definition

    def get_as_string(self, key: str) -> str:
        """Gibt das Schema als formatierten JSON-String zurück."""
        return json.dumps(self.get_definition(key), ensure_ascii=False, indent=2)

    def get_example(self, key: str) -> Optional[dict]:
        return self.get(key).example

    def list_schemas(self) -> list[SchemaMeta]:
        return self._load_index()

    def exists(self, key: str) -> bool:
        return (self._root / f"{key}.json").exists()

    def reload_all(self) -> None:
        self._cache.clear()


# Singleton
schema_registry = SchemaRegistry()
