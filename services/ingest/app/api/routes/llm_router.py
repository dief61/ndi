# services/ingest/app/api/routes/llm_router.py
#
# FastAPI-Endpoints für LLM-Gateway, Prompt-Suite und Schema-Registry.

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Lazy imports – vermeidet Startup-Fehler wenn LLM nicht konfiguriert
def _gateway():
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[3]))
    from llm_gateway.gateway import llm_gateway
    return llm_gateway

def _suite():
    from llm_gateway.prompt_suite import prompt_suite
    return prompt_suite

def _registry():
    from llm_gateway.schema_registry import schema_registry
    return schema_registry


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status", summary="LLM-Gateway Status")
async def llm_status():
    gw = _gateway()
    ok = await gw.ping()
    return {
        "provider":  gw.active_provider,
        "model":     gw.active_model,
        "erreichbar": ok,
    }


# ── Prompt-Suite ──────────────────────────────────────────────────────────────

@router.get("/prompts", summary="Alle Prompts auflisten")
async def list_prompts():
    suite = _suite()
    return {"prompts": [
        {
            "key":         m.key,
            "name":        m.name,
            "beschreibung":m.beschreibung,
            "version":     m.version,
            "provider":    m.provider,
            "schema":      m.schema,
            "tags":        m.tags,
        }
        for m in suite.list_prompts()
    ]}


@router.get("/prompts/{key}", summary="Prompt-Paar lesen")
async def get_prompt(key: str):
    suite = _suite()
    if not suite.exists(key):
        raise HTTPException(404, f"Prompt '{key}' nicht gefunden")
    pair = suite.get(key, force_reload=True)
    return {
        "key":           key,
        "system_prompt": pair.system_prompt,
        "user_template": pair.user_template,
        "variablen":     suite.get_variables(key),
        "meta":          {
            "name":        pair.meta.name,
            "beschreibung":pair.meta.beschreibung,
            "version":     pair.meta.version,
        },
    }


# ── Schema-Registry ───────────────────────────────────────────────────────────

@router.get("/schemas", summary="Alle Schemas auflisten")
async def list_schemas():
    reg = _registry()
    return {"schemas": [
        {
            "key":         m.key,
            "name":        m.name,
            "beschreibung":m.beschreibung,
            "version":     m.version,
            "tags":        m.tags,
        }
        for m in reg.list_schemas()
    ]}


@router.get("/schemas/{key}", summary="Schema lesen")
async def get_schema(key: str):
    reg = _registry()
    if not reg.exists(key):
        raise HTTPException(404, f"Schema '{key}' nicht gefunden")
    s = reg.get(key, force_reload=True)
    return {
        "key":        key,
        "definition": s.definition,
        "example":    s.example,
        "meta":       {
            "name":        s.meta.name,
            "beschreibung":s.meta.beschreibung,
        },
    }


# ── LLM-Aufruf ────────────────────────────────────────────────────────────────

class LLMRequest(BaseModel):
    prompt_key:    str
    variablen:     dict = {}
    json_mode:     bool = True
    schema_key:    Optional[str] = None


@router.post("/complete", summary="LLM-Aufruf mit Prompt-Suite")
async def llm_complete(req: LLMRequest):
    """
    Führt einen LLM-Aufruf mit einem Prompt aus der Suite aus.

    Beispiel:
    {
      "prompt_key": "ps_normtyp",
      "variablen": {
        "norm_reference": "§ 3 Abs. 1 NHundG",
        "predicate": "ist nachzuweisen",
        "sentence_text": "Die Sachkunde ist der Gemeinde nachzuweisen."
      },
      "json_mode": true
    }
    """
    suite    = _suite()
    gw       = _gateway()

    if not suite.exists(req.prompt_key):
        raise HTTPException(404, f"Prompt '{req.prompt_key}' nicht gefunden")

    # Schema in User-Prompt einbetten wenn angegeben
    variablen = dict(req.variablen)
    if req.schema_key:
        reg = _registry()
        if reg.exists(req.schema_key):
            variablen["schema"] = reg.get_as_string(req.schema_key)

    system = suite.get_system(req.prompt_key)
    user   = suite.render_user(req.prompt_key, **variablen)

    result = await gw.complete(
        system_prompt=system,
        user_prompt=user,
        json_mode=req.json_mode,
    )

    return {
        "erfolg":        result.erfolg,
        "provider":      result.provider,
        "model":         result.model,
        "content":       result.content,
        "parsed":        result.parsed,
        "input_tokens":  result.input_tokens,
        "output_tokens": result.output_tokens,
        "dauer_ms":      result.dauer_ms,
        "fehler":        result.fehler,
    }


@router.post("/reload", summary="Gateway und Prompts neu laden")
async def llm_reload():
    """Lädt Gateway, Prompts und Schemas neu – nach Konfigurationsänderungen."""
    _gateway().reload()
    _suite().reload_all()
    _registry().reload_all()
    return {"status": "ok", "message": "Gateway, Prompts und Schemas neu geladen"}
