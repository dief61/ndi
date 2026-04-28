# services/ingest/app/services/parser.py
#
# Tika-Parser: Nimmt Rohdokument-Bytes entgegen, schickt sie an den
# Apache Tika Server und gibt ein ParseResult-Objekt zurück.
#
# ParseResult enthält:
#   - text:       Volltext (bereinigt)
#   - structure:  Erkannte Strukturmerkmale (§-Marker, Headings, ToC)
#   - metadata:   Tika-Metadaten (Content-Type, Autor, Datum etc.)
#   - doc_class_hint: Vorschlag für den Chunking-Router (A / B / C)

import re
import httpx
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from app.core.config import settings

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Datenmodell: Ergebnis des Parsings
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentStructure:
    """Erkannte Strukturmerkmale eines Dokuments."""
    has_paragraph_markers: bool = False   # §-Zeichen gefunden
    has_heading_hierarchy: bool = False   # H1/H2/H3 oder 1.1.2-Nummerierung
    has_toc: bool = False                 # Inhaltsverzeichnis erkannt
    paragraph_count: int = 0             # Anzahl §-Paragraphen
    heading_count: int = 0               # Anzahl Überschriften
    headings: list = field(default_factory=list)  # Extrahierte Überschriften


@dataclass
class ParseResult:
    """Vollständiges Ergebnis des Tika-Parsings."""
    text: str                            # Bereinigter Volltext
    structure: DocumentStructure         # Strukturmerkmale
    metadata: dict                       # Rohe Tika-Metadaten
    content_type: str                    # z.B. application/pdf
    doc_class_hint: str                  # "A", "B" oder "C"
    char_count: int = 0
    word_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Tika-Parser
# ─────────────────────────────────────────────────────────────────────────────

class TikaParser:
    """
    Sendet Dokumente an den Apache Tika Server und verarbeitet die Antwort.

    Tika-Endpunkte:
      POST /tika          → Nur Text (text/plain)
      PUT  /meta          → Nur Metadaten (application/json)
      PUT  /rmeta/text    → Text + Metadaten kombiniert (application/json)

    Wir nutzen /rmeta/text, weil wir beides in einem Aufruf bekommen.
    """

    # Regex-Muster für Strukturerkennung
    #
    # Klasse A: §-Paragraph ist STRUKTURGEBEND wenn er am Zeilenanfang steht
    # und eine eigenständige Überschrift bildet (z.B. "§ 1 Anwendungsbereich").
    # Bloße Textreferenzen ("gemäß § 3 Abs. 1") stehen mitten im Satz.
    PARAGRAPH_PATTERN = re.compile(
        r'^\s*§\s*\d+\w*\s+\S',           # § am Zeilenanfang + Folgetext
        re.MULTILINE
    )
    # Klasse B: Nummerierte Überschriften – vollständige Zeile erfassen
    HEADING_PATTERN = re.compile(
        r'^(\d+\.\d+(?:\.\d+)*\s+.+)$'   # 1.1 Vollständiger Titel
        r'|^([A-ZÄÖÜ][A-ZÄÖÜ\s]{4,})$',       # GROSSBUCHSTABEN ÜBERSCHRIFT
        re.MULTILINE
    )
    # ToC-Erkennung: nur wenn die GESAMTE Zeile dem Muster entspricht.
    # "inhalt" als Substring würde in dt. Rechtstexten viele Fehlklassifikationen
    # erzeugen (z.B. "Inhalt der Erlaubnis", "inhaltlich").
    TOC_PATTERN = re.compile(
        r'(?i)^\s*(?:inhaltsverzeichnis|table\s+of\s+contents)\s*$',
        re.MULTILINE
    )

    def __init__(self):
        self.tika_url = settings.tika_server_url
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def parse(self, content: bytes, filename: str) -> ParseResult:
        """
        Hauptmethode: Dokument an Tika senden, Ergebnis aufbereiten.

        Args:
            content:  Rohe Datei-Bytes
            filename: Originalname (für Content-Type-Erkennung)

        Returns:
            ParseResult mit Text, Struktur, Metadaten und doc_class_hint
        """
        log = logger.bind(filename=filename, size_bytes=len(content))
        log.info("Tika-Parsing gestartet")

        content_type = self._guess_content_type(filename)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # /rmeta/text liefert Liste von JSON-Objekten (ein Objekt pro Seite/Anhang)
            response = await client.put(
                f"{self.tika_url}/rmeta/text",
                content=content,
                headers={
                    "Content-Type": content_type,
                    "Accept": "application/json",
                    "X-Tika-OCRLanguage": "deu+eng",   # OCR-Sprache für PDFs
                },
            )
            response.raise_for_status()

        tika_result = response.json()

        # Tika gibt eine Liste zurück – wir mergen alle Texte
        raw_text, metadata = self._extract_from_tika_result(tika_result)

        # Text bereinigen
        clean_text = self._clean_text(raw_text)

        # Struktur analysieren
        structure = self._analyze_structure(clean_text)

        # Dokumentklasse vorschlagen
        doc_class_hint = self._determine_doc_class(structure)

        result = ParseResult(
            text=clean_text,
            structure=structure,
            metadata=metadata,
            content_type=content_type,
            doc_class_hint=doc_class_hint,
            char_count=len(clean_text),
            word_count=len(clean_text.split()),
        )

        log.info(
            "Tika-Parsing abgeschlossen",
            doc_class_hint=doc_class_hint,
            char_count=result.char_count,
            word_count=result.word_count,
            paragraph_count=structure.paragraph_count,
            heading_count=structure.heading_count,
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Interne Hilfsmethoden
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_from_tika_result(self, tika_result: list) -> tuple[str, dict]:
        """
        Tika /rmeta/text gibt eine Liste von Objekten zurück.
        Jedes Objekt hat einen 'X-TIKA:content'-Schlüssel mit dem Text
        und weitere Schlüssel mit Metadaten.

        Wir mergen alle Texte und nehmen die Metadaten des ersten Objekts.
        """
        texts = []
        metadata = {}

        for i, item in enumerate(tika_result):
            # Text extrahieren
            text = item.get("X-TIKA:content", "")
            if text:
                texts.append(text.strip())

            # Metadaten nur vom ersten Objekt (Hauptdokument)
            if i == 0:
                metadata = {
                    k: v for k, v in item.items()
                    if k != "X-TIKA:content"
                }

        return "\n\n".join(texts), metadata

    def _clean_text(self, text: str) -> str:
        """
        Bereinigt den Rohtext von Tika:
        - Übermäßige Leerzeilen reduzieren
        - Steuerzeichen entfernen
        - Bindestriche normalisieren (Trenn- vs. Gedankenstrich)
        - Geschützte Leerzeichen normalisieren
        """
        if not text:
            return ""

        # Steuerzeichen entfernen (außer \n und \t)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Geschützte Leerzeichen und andere Unicode-Spaces normalisieren
        text = re.sub(r'[\u00a0\u2000-\u200b\u202f\u205f\u3000]', ' ', text)

        # Trennstriche am Zeilenende auflösen (z.B. "Verwal-\ntung" → "Verwaltung")
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # Mehrfache Leerzeichen auf eines reduzieren
        text = re.sub(r'[ \t]{2,}', ' ', text)

        # Mehr als 2 aufeinanderfolgende Leerzeilen auf 2 reduzieren
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Leerzeichen am Zeilenanfang/-ende entfernen
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(lines)

        return text.strip()

    def _analyze_structure(self, text: str) -> DocumentStructure:
        """
        Analysiert den bereinigten Text auf Strukturmerkmale.
        Ergebnis steuert den Chunking-Router.
        """
        structure = DocumentStructure()

        # §-Marker prüfen (Klasse A)
        paragraphs = self.PARAGRAPH_PATTERN.findall(text)
        structure.has_paragraph_markers = len(paragraphs) >= 2
        structure.paragraph_count = len(paragraphs)

        # Heading-Hierarchie prüfen (Klasse B)
        # findall gibt Tupel zurück wenn der Regex Gruppen hat → zu Strings flatten
        raw_headings = self.HEADING_PATTERN.findall(text)
        headings = [
            next(g for g in match if g)   # erstes nicht-leeres Gruppen-Element
            for match in raw_headings
        ]
        structure.has_heading_hierarchy = len(headings) >= 2
        structure.heading_count = len(headings)
        structure.headings = headings[:50]  # Max 50 für Metadaten

        # Inhaltsverzeichnis prüfen (stärkes Klasse-B-Signal)
        structure.has_toc = bool(self.TOC_PATTERN.search(text[:5000]))

        return structure

    def _determine_doc_class(self, structure: DocumentStructure) -> str:
        """
        Chunking-Router-Vorschlag basierend auf Strukturmerkmalen.

        Kaskade (wie im Architekturkonzept):
          1. §-Marker strukturgebend (>= 3) UND keine starken B-Signale → Klasse A
          2. Heading-Hierarchie oder Inhaltsverzeichnis vorhanden → Klasse B
          3. Fallback → Klasse C

        Wichtig: Ein Dokument mit Inhaltsverzeichnis und vielen Überschriften
        ist Klasse B – auch wenn es §-Referenzen im Fließtext enthält.
        Nur echte Gesetzestexte (§ als Strukturelement) sind Klasse A.
        """
        has_strong_b_signal = structure.has_toc or structure.heading_count >= 5

        # Klasse A: §-Paragraphen sind strukturgebend UND kein starkes B-Signal
        if (structure.has_paragraph_markers
                and structure.paragraph_count >= 3
                and not has_strong_b_signal):
            return "A"

        # Klasse B: Heading-Hierarchie oder Inhaltsverzeichnis
        elif structure.has_heading_hierarchy or structure.has_toc:
            return "B"

        # Klasse C: Kein stabiles Strukturgerüst
        else:
            return "C"

    def _guess_content_type(self, filename: str) -> str:
        """
        Content-Type aus Dateiname ableiten.
        Tika kann den Typ selbst erkennen, aber ein korrekter Hint
        verbessert die Parsing-Qualität.
        """
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        mapping = {
            "pdf":  "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document",
            "doc":  "application/msword",
            "html": "text/html",
            "htm":  "text/html",
            "txt":  "text/plain",
            "rtf":  "application/rtf",
        }
        return mapping.get(ext, "application/octet-stream")
