# services/ingest/app/api/routes/config_router.py
#
# REST-Endpoints für die Konfigurationsverwaltung.
# Liest und schreibt YAML-Konfigurationsdateien direkt auf dem Server.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# Bekannte Konfigurationsdateien
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_ROOT = Path(__file__).parents[3]  # services/ingest/

CONFIG_FILES = {
    "nlp":     SERVICE_ROOT / "nlp_config.yaml",
    "embedder":SERVICE_ROOT / "embedder_config.yaml",
    "abbrev":  SERVICE_ROOT / "abbrev_dict.yaml",
    "chunker": SERVICE_ROOT / "chunker_config.yaml",
    "docs":    SERVICE_ROOT / "docs.yaml",
}

CONFIG_META = {
    "nlp": {
        "label":       "NLP-Konfiguration",
        "description": "spaCy, SVO, Normtypen, NER, Fragen-Filter, Worker",
        "icon":        "🧠",
    },
    "embedder": {
        "label":       "Embedding-Modell",
        "description": "Aktives Modell, Batch-Size, Device, Cache",
        "icon":        "🔢",
    },
    "abbrev": {
        "label":       "Wörterbücher",
        "description": "Abkürzungen und Synonyme",
        "icon":        "📖",
    },
    "chunker": {
        "label":       "Chunker-Konfiguration",
        "description": "Chunking-Parameter: Token-Limits, Klassen",
        "icon":        "✂️",
    },
    "docs": {
        "label":       "Dokument-Metadaten",
        "description": "Dokumenttypen, Titel, Jurisdiktionen, Typ-Erkennung",
        "icon":        "🗂️",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ConfigResponse(BaseModel):
    name:        str
    label:       str
    description: str
    icon:        str
    path:        str
    content:     dict
    raw_yaml:    str


class ConfigUpdateRequest(BaseModel):
    raw_yaml: Optional[str] = None
    content:  Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _read_config(name: str) -> ConfigResponse:
    if name not in CONFIG_FILES:
        raise HTTPException(404, f"Konfiguration '{name}' nicht gefunden.")

    path = CONFIG_FILES[name]
    if not path.exists():
        raise HTTPException(404, f"Datei nicht gefunden: {path}")

    raw = path.read_text(encoding="utf-8")
    try:
        content = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise HTTPException(422, f"YAML-Fehler: {e}")

    meta = CONFIG_META[name]
    return ConfigResponse(
        name=name,
        label=meta["label"],
        description=meta["description"],
        icon=meta["icon"],
        path=str(path),
        content=content,
        raw_yaml=raw,
    )


def _write_config(name: str, raw_yaml: str) -> dict:
    """Schreibt YAML nach Validierung zurück auf Disk."""
    if name not in CONFIG_FILES:
        raise HTTPException(404, f"Konfiguration '{name}' nicht gefunden.")

    # YAML validieren
    try:
        parsed = yaml.safe_load(raw_yaml)
        if parsed is None:
            raise HTTPException(422, "YAML ist leer.")
    except yaml.YAMLError as e:
        raise HTTPException(422, f"Ungültiges YAML: {e}")

    # Backup der alten Datei
    path = CONFIG_FILES[name]
    if path.exists():
        backup = path.with_suffix(".yaml.bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    # Neue Datei schreiben
    path.write_text(raw_yaml, encoding="utf-8")
    return {"status": "ok", "message": f"{name} gespeichert.", "path": str(path)}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", summary="Alle Konfigurationen auflisten")
async def list_configs():
    result = []
    for name, meta in CONFIG_META.items():
        path = CONFIG_FILES[name]
        result.append({
            "name":        name,
            "label":       meta["label"],
            "description": meta["description"],
            "icon":        meta["icon"],
            "exists":      path.exists(),
            "path":        str(path),
        })
    return result


@router.get("/config_md/raw", summary="CONFIG.md lesen (nur Anzeige)")
async def get_config_md():
    """Liest CONFIG.md aus dem Projektverzeichnis (schreibgeschützt)."""
    candidates = [
        SERVICE_ROOT.parents[1] / "CONFIG.md",   # NDI/CONFIG.md
        SERVICE_ROOT.parents[0] / "CONFIG.md",   # services/CONFIG.md
        SERVICE_ROOT / "CONFIG.md",               # services/ingest/CONFIG.md
    ]
    for path in candidates:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            return {
                "name":     "config_md",
                "label":    "Config-Übersicht",
                "path":     str(path),
                "raw_yaml": raw,   # Feldname raw_yaml für JS-Kompatibilität
                "readonly": True,
            }
    raise HTTPException(404, "CONFIG.md nicht gefunden.")


@router.get("/ui", response_class=HTMLResponse,
            summary="Config Manager UI",
            include_in_schema=False)
async def config_ui():
    """
    Liefert die Config Manager HTML-Oberfläche aus.
    Muss VOR /{name} registriert sein – sonst matcht FastAPI 'ui' als name.
    Aufruf: http://localhost:8000/api/v1/config/ui
    """
    candidates = [
        SERVICE_ROOT / "config_manager.html",
        SERVICE_ROOT / "tools" / "config_manager.html",
        Path(__file__).parent / "config_manager.html",
    ]
    for path in candidates:
        if path.exists():
            return HTMLResponse(content=path.read_text(encoding="utf-8"))
    return HTMLResponse(
        status_code=404,
        content=(
            "<h2>config_manager.html nicht gefunden</h2>"
            "<p>Datei ablegen unter:</p>"
            f"<code>{SERVICE_ROOT / 'config_manager.html'}</code>"
        ),
    )


@router.get("/{name}", response_model=ConfigResponse,
            summary="Konfiguration lesen")
async def get_config(name: str):
    return _read_config(name)


@router.put("/{name}", summary="Konfiguration speichern")
async def update_config(name: str, body: ConfigUpdateRequest):
    if body.raw_yaml:
        return _write_config(name, body.raw_yaml)
    elif body.content:
        raw = yaml.dump(
            body.content,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        return _write_config(name, raw)
    else:
        raise HTTPException(422, "raw_yaml oder content muss angegeben werden.")


@router.post("/{name}/restore", summary="Backup wiederherstellen")
async def restore_backup(name: str):
    if name not in CONFIG_FILES:
        raise HTTPException(404, f"Konfiguration '{name}' nicht gefunden.")
    path   = CONFIG_FILES[name]
    backup = path.with_suffix(".yaml.bak")
    if not backup.exists():
        raise HTTPException(404, "Kein Backup vorhanden.")
    path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    return {"status": "ok", "message": f"Backup von '{name}' wiederhergestellt."}


@router.get("/{name}/validate", summary="YAML-Syntax prüfen")
async def validate_config(name: str, raw: str):
    try:
        yaml.safe_load(raw)
        return {"valid": True}
    except yaml.YAMLError as e:
        return {"valid": False, "error": str(e)}

