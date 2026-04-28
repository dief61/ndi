# services/ingest/app/services/embedder.py
#
# Embedding-Pipeline: Berechnet Vektoren für Chunks aus dem Chunking-Router.
#
# Unterstützte Modelle (via embedder_config.yaml):
#   - mixedbread-ai/deepset-mxbai-embed-de-large-v1  (Standard, Deutsch/Englisch)
#   - intfloat/multilingual-e5-large                 (Fallback, mehrsprachig)
#
# Beide Modelle:
#   - 1024 Dimensionen
#   - Prefix "passage:" für Chunks (Ingest)
#   - Prefix "query:"   für Suchanfragen (RAG-Engine, M3)

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from functools import lru_cache

import structlog
import yaml
from sentence_transformers import SentenceTransformer

from app.services.chunker import Chunk

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration laden
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "embedder_config.yaml"


def load_embedder_config(config_path: Optional[Path] = None) -> dict:
    """
    Lädt embedder_config.yaml.
    Sucht zuerst am angegebenen Pfad, dann automatisch im Service-Root.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"embedder_config.yaml nicht gefunden: {path}\n"
            f"Bitte Datei anlegen (Vorlage im Repository unter services/ingest/)."
        )

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    active = cfg.get("active_model", "deepset-mxbai")
    models = cfg.get("models", {})

    if active not in models:
        raise ValueError(
            f"Modell '{active}' nicht in embedder_config.yaml definiert. "
            f"Verfügbare Profile: {list(models.keys())}"
        )

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

class Embedder:
    """
    Berechnet Embeddings für Chunks mit sentence-transformers.

    Das aktive Modell wird aus embedder_config.yaml gelesen.
    Ein Wechsel des Modells erfordert nur eine Änderung in der Config-Datei –
    kein Code-Änderung.

    Prefixes:
      - Chunks (Ingest):       "passage: {text}"
      - Suchanfragen (M3):     "query: {text}"
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.cfg         = load_embedder_config(config_path)
        self.active_name = self.cfg["active_model"]
        self.model_cfg   = self.cfg["models"][self.active_name]
        self.runtime_cfg = self.cfg.get("runtime", {})

        self.model_id       = self.model_cfg["model_id"]
        self.dimensions     = self.model_cfg.get("dimensions", 1024)
        self.prefix_query   = self.model_cfg.get("prefix_query",   "query: ")
        self.prefix_passage = self.model_cfg.get("prefix_passage",  "passage: ")
        self.normalize      = self.model_cfg.get("normalize", True)

        self.batch_size    = self.runtime_cfg.get("batch_size",    16)
        self.show_progress = self.runtime_cfg.get("show_progress", True)
        self.cache_enabled = self.runtime_cfg.get("cache_enabled", True)
        self.cache_max     = self.runtime_cfg.get("cache_max_size", 1000)

        # Gerät bestimmen
        self.device = self._resolve_device(self.runtime_cfg.get("device", "auto"))

        # Embedding-Cache (text_hash → embedding)
        self._cache: dict[str, list[float]] = {}

        # Modell wird beim ersten Aufruf geladen (lazy loading)
        self._model: Optional[SentenceTransformer] = None

        logger.info(
            "Embedder initialisiert",
            model=self.active_name,
            model_id=self.model_id,
            device=self.device,
            dimensions=self.dimensions,
            batch_size=self.batch_size,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Öffentliche API
    # ─────────────────────────────────────────────────────────────────────────

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Berechnet Embeddings für eine Liste von Chunks.
        Schreibt das Embedding direkt in chunk.embedding.

        Args:
            chunks: Liste von Chunk-Objekten (aus dem Chunking-Router)

        Returns:
            Dieselbe Liste, jeder Chunk hat nun chunk.embedding gesetzt.
        """
        if not chunks:
            return chunks

        model = self._get_model()
        log   = logger.bind(chunk_count=len(chunks), model=self.active_name)
        log.info("Embedding-Berechnung gestartet")

        # Cache-Trennung: welche Chunks brauchen neue Embeddings?
        to_embed:   list[tuple[int, str]] = []  # (index, text)
        from_cache: dict[int, list[float]] = {}

        for i, chunk in enumerate(chunks):
            text_hash = self._hash(chunk.text)
            if self.cache_enabled and text_hash in self._cache:
                from_cache[i] = self._cache[text_hash]
            else:
                to_embed.append((i, chunk.text))

        log.info(
            "Cache-Auswertung",
            cache_hits=len(from_cache),
            to_compute=len(to_embed),
        )

        # Neue Embeddings berechnen
        if to_embed:
            indices = [i for i, _ in to_embed]
            texts   = [self.prefix_passage + t for _, t in to_embed]

            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=self.show_progress and len(texts) > 10,
                convert_to_numpy=True,
            )

            for idx, (orig_idx, orig_text) in enumerate(to_embed):
                emb = embeddings[idx].tolist()
                chunks[orig_idx].embedding = emb

                # Cache befüllen
                if self.cache_enabled:
                    text_hash = self._hash(orig_text)
                    if len(self._cache) < self.cache_max:
                        self._cache[text_hash] = emb

        # Cache-Treffer eintragen
        for i, emb in from_cache.items():
            chunks[i].embedding = emb

        log.info(
            "Embedding-Berechnung abgeschlossen",
            total=len(chunks),
            computed=len(to_embed),
            from_cache=len(from_cache),
        )

        return chunks

    def embed_query(self, query_text: str) -> list[float]:
        """
        Berechnet das Embedding für eine Suchanfrage (RAG-Engine, M3).
        Verwendet "query:" Prefix statt "passage:".

        Args:
            query_text: Die Suchanfrage als Text

        Returns:
            1024-dimensionaler Embedding-Vektor
        """
        model = self._get_model()
        prefixed = self.prefix_query + query_text

        embedding = model.encode(
            [prefixed],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()

    @property
    def active_model_info(self) -> dict:
        """Gibt Informationen über das aktive Modell zurück."""
        return {
            "name":       self.active_name,
            "model_id":   self.model_id,
            "dimensions": self.dimensions,
            "device":     self.device,
            "license":    self.model_cfg.get("license", "unbekannt"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Interne Hilfsmethoden
    # ─────────────────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Lazy-Loading: Modell wird erst beim ersten Aufruf geladen."""
        if self._model is None:
            logger.info(
                "Lade Embedding-Modell",
                model_id=self.model_id,
                device=self.device,
            )
            self._model = SentenceTransformer(
                self.model_id,
                device=self.device,
                trust_remote_code=False,
            )
            # Maximale Sequenzlänge aus Config setzen
            max_len = self.model_cfg.get("max_seq_length", 512)
            self._model.max_seq_length = max_len

            logger.info(
                "Embedding-Modell geladen",
                model_id=self.model_id,
                max_seq_length=max_len,
            )
        return self._model

    def _resolve_device(self, device: str) -> str:
        """Löst 'auto' auf: cuda wenn verfügbar, sonst cpu."""
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _hash(text: str) -> str:
        """MD5-Hash eines Textes für den Cache-Key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
