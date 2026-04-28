# services/ingest/app/services/chunker.py
#
# Chunking-Router: Erkennt die Dokumentklasse (A/B/C) und wendet
# die passende Chunking-Strategie an.
#
# Klasse A – Normative Rechtstexte:    §-basiertes hierarchisches Chunking
# Klasse B – Strukturierte Fach-Dok.:  Kapitel-basiertes hierarchisches Chunking
# Klasse C – Unstrukturierte Texte:    Semantisches Sliding-Window-Chunking
#
# Jeder Chunk trägt das gemeinsame Metadaten-Schema aus dem Architekturkonzept.

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
import structlog

from app.services.parser import DocumentStructure

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-Datenmodell (gemeinsames Schema für alle Klassen)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    Einzelner Chunk – entspricht dem gemeinsamen Metadaten-Schema
    aus Abschnitt 4.2 des Architekturkonzepts.
    """
    chunk_id:           str
    doc_id:             str
    doc_class:          str                     # A | B | C

    # Inhalt
    text:               str
    token_count:        int = 0

    # Klasse A
    norm_reference:     Optional[str] = None
    cross_references:   list[str] = field(default_factory=list)

    # Klasse B
    section_path:       Optional[str] = None
    heading_breadcrumb: Optional[str] = None
    requirement_id:     Optional[str] = None

    # Gemeinsam
    chunk_type:         Optional[str] = None    # tatbestand|rechtsfolge|definition|...
    hierarchy_level:    int = 0
    parent_chunk_id:    Optional[str] = None
    overlap_with_prev:  float = 0.0
    confidence_weight:  float = 1.0             # A=1.0 | B=0.85 | C=0.65

    # Versionierung (aus Dokument-Metadaten)
    version:            Optional[str] = None
    valid_from:         Optional[str] = None

    # Embedding wird später vom Embedder befüllt
    embedding:          Optional[list[float]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktion: Token-Schätzung
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Einfache Token-Schätzung: ~1 Token pro 4 Zeichen (deutsch).
    Für Produktionsbetrieb durch tiktoken ersetzen.
    """
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Klasse A: §-basiertes hierarchisches Chunking
# ─────────────────────────────────────────────────────────────────────────────

class ClassAChunker:
    """
    Normative Rechtstexte (Gesetze, Verordnungen).

    Hierarchie: §-Paragraph (Parent) → Absatz (Child) → Satz (optional)
    Token-Limits: Parent max. 1024 | Child max. 256 | Satz max. 128
    Legaldefinitionen und Verweise werden als eigene Chunks extrahiert.
    """

    # §-Paragraph Erkennung am Zeilenanfang.
    # Wir splitten am Zeilenumbruch VOR dem §-Zeichen – das ist robuster
    # als ein Lookahead mit ^ in re.split (Python-Einschränkung).
    PARA_SPLIT = re.compile(
        r'\n(?=[ \t]*§[ \t]*\d+)',
    )
    # Absatz-Erkennung: (1) (2) (3) oder Abs. 1
    ABSATZ_SPLIT = re.compile(
        r'(?=^\s*(?:\(\d+\)|Abs\.\s*\d+))',
        re.MULTILINE
    )
    # Legaldefinition
    DEFINITION_PATTERN = re.compile(
        r'im\s+Sinne\s+(?:dieses\s+Gesetzes|dieser\s+Verordnung|des\s+§\s*\d+)',
        re.IGNORECASE
    )
    # Verweis-Erkennung
    VERWEIS_PATTERN = re.compile(
        r'§\s*\d+\w*(?:\s+Abs\.\s*\d+)?(?:\s+Satz\s*\d+)?',
        re.IGNORECASE
    )
    # Normtyp-Erkennung (einfache regelbasierte Vorstufe zu spaCy in M2)
    NORMTYP_PATTERNS = {
        "MUST":       re.compile(r'\b(?:muss|müssen|ist\s+zu|sind\s+zu|hat\s+zu|haben\s+zu)\b', re.I),
        "MAY":        re.compile(r'\b(?:kann|können|darf|dürfen|ist\s+berechtigt)\b', re.I),
        "MUST_NOT":   re.compile(r'\b(?:darf\s+nicht|dürfen\s+nicht|ist\s+untersagt|sind\s+untersagt)\b', re.I),
        "DEF":        re.compile(r'\bim\s+Sinne\b', re.I),
        "EXCEPT":     re.compile(r'\b(?:es\s+sei\s+denn|sofern\s+nicht|ausgenommen)\b', re.I),
        "DEADLINE":   re.compile(r'\b(?:binnen|spätestens|unverzüglich|innerhalb\s+von)\b', re.I),
        "COMPETENCE": re.compile(r'\b(?:zuständig\s+ist|obliegt|ist\s+zuständig)\b', re.I),
    }

    TOKEN_LIMIT_PARENT = 1024
    TOKEN_LIMIT_CHILD  = 256
    TOKEN_LIMIT_SATZ   = 128

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: object,
    ) -> list[Chunk]:
        chunks = []

        # §-Paragraphen trennen
        paragraphs = self.PARA_SPLIT.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        for para_text in paragraphs:
            # Normreferenz aus erstem Satz extrahieren
            norm_ref = self._extract_norm_reference(para_text)
            if not norm_ref:
                continue  # Kein §-Paragraph → überspringen

            parent_id = str(uuid.uuid4())

            # Parent-Chunk: gesamter §-Paragraph
            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                doc_class="A",
                text=self._truncate(para_text, self.TOKEN_LIMIT_PARENT),
                token_count=estimate_tokens(para_text),
                norm_reference=norm_ref,
                cross_references=self._extract_cross_references(para_text, norm_ref),
                chunk_type="tatbestand",
                hierarchy_level=1,
                confidence_weight=1.0,
                version=getattr(metadata, "version", None),
                valid_from=str(metadata.valid_from) if getattr(metadata, "valid_from", None) else None,
            )
            chunks.append(parent_chunk)

            # Child-Chunks: Absätze
            absaetze = self.ABSATZ_SPLIT.split(para_text)
            absaetze = [a.strip() for a in absaetze if a.strip() and len(a.strip()) > 20]

            # Erste Teilmenge ist oft der Einleitungssatz vor (1) → überspringen
            # wenn er identisch mit dem Parent-Text beginnt
            if absaetze and absaetze[0].startswith(para_text[:30].strip()):
                absaetze = absaetze[1:] if len(absaetze) > 1 else []

            # Wenn keine Absätze erkannt: Satzweise aufteilen
            if not absaetze:
                absaetze = self._split_by_sentences(para_text, self.TOKEN_LIMIT_CHILD)
                # Ersten Satz überspringen wenn er die §-Überschrift ist
                if absaetze and estimate_tokens(absaetze[0]) < 15:
                    absaetze = absaetze[1:]

            for absatz_text in absaetze:
                # Legaldefinitionen als eigene Chunks
                if self.DEFINITION_PATTERN.search(absatz_text):
                    def_chunk = Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        doc_class="A",
                        text=self._truncate(absatz_text, self.TOKEN_LIMIT_CHILD),
                        token_count=estimate_tokens(absatz_text),
                        norm_reference=norm_ref,
                        chunk_type="definition",
                        hierarchy_level=2,
                        parent_chunk_id=parent_id,
                        confidence_weight=1.0,
                        version=getattr(metadata, "version", None),
                    )
                    chunks.append(def_chunk)
                    continue

                chunk_type = self._classify_normtyp(absatz_text)

                child_chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    doc_class="A",
                    text=self._truncate(absatz_text, self.TOKEN_LIMIT_CHILD),
                    token_count=estimate_tokens(absatz_text),
                    norm_reference=norm_ref,
                    chunk_type=chunk_type,
                    hierarchy_level=2,
                    parent_chunk_id=parent_id,
                    confidence_weight=1.0,
                    version=getattr(metadata, "version", None),
                )

                # Sehr langer Absatz → Satz-Ebene
                if child_chunk.token_count > self.TOKEN_LIMIT_CHILD:
                    saetze = self._split_by_sentences(absatz_text, self.TOKEN_LIMIT_SATZ)
                    for satz in saetze:
                        chunks.append(Chunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            doc_class="A",
                            text=satz,
                            token_count=estimate_tokens(satz),
                            norm_reference=norm_ref,
                            chunk_type=chunk_type,
                            hierarchy_level=3,
                            parent_chunk_id=parent_id,
                            confidence_weight=1.0,
                        ))
                else:
                    chunks.append(child_chunk)

        return chunks

    def _extract_norm_reference(self, text: str) -> Optional[str]:
        """
        Extrahiert die Normreferenz aus dem Paragraphen-Text.

        Zwei Formate werden unterstützt:
          Format 1: "§ 1 Anwendungsbereich"  – Titel auf gleicher Zeile
          Format 2: "§ 1\nAnwendungsbereich" – Titel auf nächster Zeile (häufig in PDFs)
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return None

        first_line = lines[0]

        # § vorhanden?
        match = re.match(r'(§\s*\d+\w*)((?:\s+\w+)*)', first_line)
        if not match:
            return None

        para_num = match.group(1).strip()   # z.B. "§ 1"
        inline_title = match.group(2).strip()  # z.B. "Anwendungsbereich" oder ""

        if inline_title:
            # Format 1: Titel auf gleicher Zeile
            return f"{para_num} {inline_title}"
        elif len(lines) > 1:
            # Format 2: Titel auf nächster Zeile – nur wenn kurz und kein Absatzmarker
            next_line = lines[1]
            if (len(next_line) < 80
                    and not re.match(r'^\(\d+\)', next_line)
                    and not next_line.startswith("§")):
                return f"{para_num} {next_line}"
        return para_num

    def _extract_cross_references(self, text: str, own_ref: str) -> list[str]:
        """Findet §-Verweise im Text (außer dem eigenen)."""
        refs = self.VERWEIS_PATTERN.findall(text)
        return list({r for r in refs if r.strip() != own_ref})[:10]

    def _classify_normtyp(self, text: str) -> str:
        """Einfache Normtyp-Klassifikation (wird in M2 durch spaCy ersetzt)."""
        for normtyp, pattern in self.NORMTYP_PATTERNS.items():
            if pattern.search(text):
                return normtyp.lower()
        return "tatbestand"

    def _split_by_sentences(self, text: str, max_tokens: int) -> list[str]:
        """Teilt Text an Satzgrenzen auf."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], []
        current_tokens = 0
        for sent in sentences:
            t = estimate_tokens(sent)
            if current_tokens + t > max_tokens and current:
                chunks.append(' '.join(current))
                current, current_tokens = [], 0
            current.append(sent)
            current_tokens += t
        if current:
            chunks.append(' '.join(current))
        return chunks

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Kürzt Text auf max_tokens (zeichenbasiert)."""
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text


# ─────────────────────────────────────────────────────────────────────────────
# Klasse B: Kapitel-basiertes hierarchisches Chunking
# ─────────────────────────────────────────────────────────────────────────────

class ClassBChunker:
    """
    Strukturierte Fachdokumente (Fachkonzepte, Lastenhefte, Standards).

    Hierarchie: Kapitel (Parent) → Unterkapitel (Child)
    Token-Limits: Parent 128 (Titel+Einleitung) | Child max. 512
    Overlap: 15% zwischen benachbarten Unterkapiteln
    Besonderheiten: heading_breadcrumb, Tabellen-Chunks, requirement_id
    """

    # Nummerierte Überschrift: 1. / 1.1 / 1.1.1
    HEADING_PATTERN = re.compile(
        r'^(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{3,}.*)?$',
        re.MULTILINE
    )
    # Anforderungs-ID: A-001, REQ-042, ANF-007
    REQUIREMENT_PATTERN = re.compile(
        r'\b([A-Z]{1,4}-\d{3,})\b'
    )
    # Tabellen-Erkennung (Tika gibt Tabellen oft mit | oder Tab-Zeichen aus)
    TABLE_PATTERN = re.compile(
        r'(?:(?:[^\n]+\|[^\n]+\n){2,}|(?:[^\n]+\t[^\n]+\n){3,})'
    )

    TOKEN_LIMIT_PARENT    = 128
    TOKEN_LIMIT_CHILD     = 512
    TOKEN_LIMIT_ANFORDING = 128
    OVERLAP_RATIO         = 0.15

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: object,
        headings: list[str],
    ) -> list[Chunk]:
        chunks = []

        # Dokument in Abschnitte trennen
        sections = self._split_into_sections(text)

        breadcrumb_stack: list[tuple[str, str]] = []  # (section_path, heading)
        prev_child_text = ""

        for section_path, heading, section_text in sections:
            # Breadcrumb-Stack aktualisieren
            breadcrumb_stack = self._update_breadcrumb(
                breadcrumb_stack, section_path, heading
            )
            breadcrumb = " > ".join(h for _, h in breadcrumb_stack)

            parent_id = str(uuid.uuid4())
            depth = section_path.count('.') + 1

            # Parent-Chunk: Kapitel-Titel + erste Sätze als Kontext-Anker
            intro = self._extract_intro(section_text, self.TOKEN_LIMIT_PARENT)
            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                doc_class="B",
                text=f"{heading}\n{intro}".strip(),
                token_count=estimate_tokens(intro),
                section_path=section_path,
                heading_breadcrumb=breadcrumb,
                chunk_type="anforderung" if depth > 1 else "tatbestand",
                hierarchy_level=depth,
                confidence_weight=0.85,
                version=getattr(metadata, "version", None),
            )
            chunks.append(parent_chunk)

            # Tabellen als eigene Chunks
            tables = self.TABLE_PATTERN.findall(section_text)
            remaining_text = self.TABLE_PATTERN.sub('', section_text)

            for table_text in tables:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    doc_class="B",
                    text=table_text.strip(),
                    token_count=estimate_tokens(table_text),
                    section_path=section_path,
                    heading_breadcrumb=breadcrumb,
                    chunk_type="tabelle",
                    hierarchy_level=depth + 1,
                    parent_chunk_id=parent_id,
                    confidence_weight=0.85,
                ))

            # Child-Chunks mit 15% Overlap
            child_chunks = self._split_with_overlap(
                remaining_text,
                self.TOKEN_LIMIT_CHILD,
                self.OVERLAP_RATIO,
                prev_child_text,
            )

            for i, child_text in enumerate(child_chunks):
                # Anforderungs-IDs extrahieren
                req_ids = self.REQUIREMENT_PATTERN.findall(child_text)
                req_id = req_ids[0] if req_ids else None

                overlap = self.OVERLAP_RATIO if i > 0 else 0.0

                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    doc_class="B",
                    text=child_text,
                    token_count=estimate_tokens(child_text),
                    section_path=section_path,
                    heading_breadcrumb=breadcrumb,
                    requirement_id=req_id,
                    chunk_type="anforderung" if req_id else "tatbestand",
                    hierarchy_level=depth + 1,
                    parent_chunk_id=parent_id,
                    overlap_with_prev=overlap,
                    confidence_weight=0.85,
                ))

            # Letzten Child-Text für Overlap-Berechnung merken
            if child_chunks:
                prev_child_text = child_chunks[-1]

        return chunks

    def _split_into_sections(
        self, text: str
    ) -> list[tuple[str, str, str]]:
        """
        Teilt den Text an nummerierten Überschriften auf.
        Gibt Liste von (section_path, heading, text) zurück.
        """
        sections = []
        matches = list(self.HEADING_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            section_path = match.group(1)
            heading = match.group(2).strip()
            # Text bis zur nächsten Überschrift
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text or heading:
                sections.append((section_path, heading, section_text))

        # Wenn keine Überschriften gefunden: gesamten Text als einen Abschnitt
        if not sections:
            sections = [("1", "Inhalt", text)]

        return sections

    def _update_breadcrumb(
        self,
        stack: list[tuple[str, str]],
        section_path: str,
        heading: str,
    ) -> list[tuple[str, str]]:
        """Aktualisiert den Breadcrumb-Stack basierend auf der Gliederungstiefe."""
        depth = len(section_path.split('.'))
        stack = [(p, h) for p, h in stack if len(p.split('.')) < depth]
        stack.append((section_path, heading[:30]))  # Max 30 Zeichen
        return stack

    def _extract_intro(self, text: str, max_tokens: int) -> str:
        """Extrahiert die ersten Sätze eines Abschnitts als Kontext-Anker."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        intro, tokens = [], 0
        for sent in sentences:
            t = estimate_tokens(sent)
            if tokens + t > max_tokens:
                break
            intro.append(sent)
            tokens += t
        return ' '.join(intro)

    def _split_with_overlap(
        self,
        text: str,
        max_tokens: int,
        overlap_ratio: float,
        prev_text: str,
    ) -> list[str]:
        """
        Teilt Text in Chunks mit Overlap zum vorherigen Chunk.
        Overlap wird als letzte N Tokens des vorherigen Chunks vorangestellt.
        """
        if not text.strip():
            return []

        words = text.split()
        overlap_words = int(max_tokens * overlap_ratio)
        step = max_tokens - overlap_words
        chunks = []

        # Overlap vom vorherigen Abschnitt
        prev_words = prev_text.split()
        overlap_prefix = prev_words[-overlap_words:] if prev_words else []

        i = 0
        while i < len(words):
            chunk_words = (overlap_prefix if i == 0 else []) + words[i:i + step]
            chunks.append(' '.join(chunk_words))
            i += step

        return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Klasse C: Semantisches Sliding-Window-Chunking
# ─────────────────────────────────────────────────────────────────────────────

class ClassCChunker:
    """
    Unstrukturierte Ergänzungstexte (FAQs, Handreichungen, Auslegungshinweise).

    Methode: Sentence-Splitter mit Sliding-Window
    Token-Bereich: min. 3 Sätze / max. 384 Token
    Overlap: 20% Sliding-Window
    confidence_weight: 0.65 (niedrigste Priorität)
    """

    TOKEN_MIN     = 50    # Minimum ~3 kurze Sätze
    TOKEN_MAX     = 384
    OVERLAP_RATIO = 0.20

    # Satzgrenzen (deutsch)
    SENTENCE_SPLIT = re.compile(
        r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])'
    )

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: object,
    ) -> list[Chunk]:
        chunks = []

        sentences = self.SENTENCE_SPLIT.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        overlap_count = max(1, int(len(sentences) * self.OVERLAP_RATIO))
        window_size   = self._estimate_window_size(sentences)

        i = 0
        while i < len(sentences):
            window = sentences[i:i + window_size]
            chunk_text = ' '.join(window)

            token_count = estimate_tokens(chunk_text)

            # Zu kurz: nächste Sätze hinzunehmen
            if token_count < self.TOKEN_MIN and i + window_size < len(sentences):
                window_size += 1
                continue

            if chunk_text.strip():
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    doc_class="C",
                    text=chunk_text,
                    token_count=token_count,
                    chunk_type="tatbestand",
                    hierarchy_level=1,
                    overlap_with_prev=self.OVERLAP_RATIO if i > 0 else 0.0,
                    confidence_weight=0.65,
                    version=getattr(metadata, "version", None),
                ))

            i += max(1, window_size - overlap_count)
            window_size = self._estimate_window_size(sentences)

        return chunks

    def _estimate_window_size(self, sentences: list[str]) -> int:
        """Schätzt sinnvolle Fenstergröße basierend auf durchschn. Satzlänge."""
        if not sentences:
            return 5
        avg_tokens = sum(estimate_tokens(s) for s in sentences[:20]) / min(20, len(sentences))
        if avg_tokens == 0:
            return 5
        return max(3, min(10, int(self.TOKEN_MAX / avg_tokens)))


# ─────────────────────────────────────────────────────────────────────────────
# Chunking-Router
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingRouter:
    """
    Haupteinstiegspunkt für das Chunking.

    Nimmt ParseResult entgegen, erkennt die Dokumentklasse (oder übernimmt
    den doc_class_hint des Parsers) und delegiert an die passende Strategie.
    """

    def __init__(self):
        self.chunker_a = ClassAChunker()
        self.chunker_b = ClassBChunker()
        self.chunker_c = ClassCChunker()

    def route_and_chunk(
        self,
        text: str,
        structure: DocumentStructure,
        doc_id: str,
        metadata: object,
        doc_class_override: Optional[str] = None,
    ) -> list[Chunk]:
        """
        Hauptmethode: Routing + Chunking.

        Args:
            text:               Bereinigter Volltext aus dem Parser
            structure:          DocumentStructure aus dem Parser
            doc_id:             UUID des Quelldokuments
            metadata:           DocumentMetadata aus dem Ingest-Request
            doc_class_override: Optionale manuelle Klassen-Überschreibung

        Returns:
            Liste von Chunk-Objekten mit vollständigem Metadaten-Schema
        """
        # Dokumentklasse bestimmen
        doc_class = doc_class_override or self._determine_class(structure)

        log = logger.bind(doc_id=doc_id, doc_class=doc_class)
        log.info("Chunking-Router: Klasse bestimmt")

        # Chunking delegieren
        if doc_class == "A":
            chunks = self.chunker_a.chunk(
                text=text,
                doc_id=doc_id,
                metadata=metadata,
            )
        elif doc_class == "B":
            chunks = self.chunker_b.chunk(
                text=text,
                doc_id=doc_id,
                metadata=metadata,
                headings=structure.headings,
            )
        else:  # C
            chunks = self.chunker_c.chunk(
                text=text,
                doc_id=doc_id,
                metadata=metadata,
            )

        # Token-Counts berechnen
        for chunk in chunks:
            if chunk.token_count == 0:
                chunk.token_count = estimate_tokens(chunk.text)

        # Leere Chunks entfernen
        chunks = [c for c in chunks if c.text.strip()]

        log.info(
            "Chunking abgeschlossen",
            chunk_count=len(chunks),
            doc_class=doc_class,
        )

        return chunks

    def _determine_class(self, structure: DocumentStructure) -> str:
        """
        Regelbasierte Kaskade (identisch mit Parser-Logik,
        aber auf DocumentStructure-Objekt operierend).
        """
        has_strong_b = structure.has_toc or structure.heading_count >= 5

        if (structure.has_paragraph_markers
                and structure.paragraph_count >= 3
                and not has_strong_b):
            return "A"
        elif structure.has_heading_hierarchy or structure.has_toc:
            return "B"
        else:
            return "C"
