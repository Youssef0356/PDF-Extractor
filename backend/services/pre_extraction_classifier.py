"""
Pre-extraction classifier for industrial instrumentation PDFs.
Runs BEFORE the LLM on the full document text.

PURPOSE
-------
Local Ollama LLMs hallucinate on long operating manuals because:
  1. The document contains many topics (flow measurement examples,
     volume flow setup wizards, etc.) that appear more often than
     the actual primary measurement type stated on page 1.
  2. The LLM chunks are small and lose global context.
  3. The model self-reports 0.9–1.0 confidence regardless.

This module solves this by extracting the HIGH-CONFIDENCE fields
deterministically from the full text using weighted keyword scoring
and regex patterns.  Results are injected into extracted_data with
confidence=1.0 BEFORE extract_all_fields() is called, so the LLM
never touches these fields.

FIELDS HANDLED HERE (never sent to LLM if matched):
  - typeMesure   (most critical — drives code, technologie, etc.)
  - code         (LT, FT, PT, TT, AT…)
  - technologie  (Radar, Electromagnétique, Coriolis, etc.)
  - marque       (Siemens, KROHNE, E+H, etc.)
  - signalSortie (4-20mA, etc.)
  - hart         (true/false)
  - alimentation (24VDC, boucle, etc.)
  - indiceIP     (IP65, IP66, IP67…)
  - températureProcess (only if ° symbol present)
  - certificats  (ATEX, IECEx, SIL…)

FIELDS ALWAYS LEFT TO LLM:
  - plageMesureMin/Max/Unite  (user fills these anyway)
  - nombreFils                (usually obvious)
  - precision                 (needs verbatim quote)
  - seuil / sortieTOR         (context-dependent)
  - matériauMembrane          (needs specific table context)
"""

from __future__ import annotations
import re
from typing import Any

# ---------------------------------------------------------------------------
# Type Mesure — weighted keyword scoring
# The winner must beat second place by MARGIN to be accepted.
# This prevents "flow" in a table of contents beating "level measurement"
# in the description body.
# ---------------------------------------------------------------------------

_TYPE_MESURE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "Niveau": [
        # Strong signals — product description, principle of operation
        (r'level\s+measurement', 10.0),
        (r'mesure\s+de\s+niveau', 10.0),
        (r'continuous\s+level', 8.0),
        (r'niveau\s+continu', 8.0),
        (r'non.contact.*level', 8.0),
        (r'radar.*level', 7.0),
        (r'level.*radar', 7.0),
        (r'level.*transmitter', 6.0),
        (r'transmetteur.*niveau', 6.0),
        (r'LR\d+', 5.0),          # SITRANS LR = Level Radar
        (r'\bLT[-\s]\d+', 5.0),   # ISA tag LT-xxx
        (r'\blevel\b', 1.0),       # generic — low weight
        (r'\bniveau\b', 1.0),
    ],
    "Débit": [
        (r'flow\s+measurement', 10.0),
        (r'mesure\s+de\s+débit', 10.0),
        (r'flow\s+meter', 8.0),
        (r'débitmètre', 8.0),
        (r'electromagnetic.*flow', 7.0),
        (r'coriolis.*flow', 7.0),
        (r'vortex.*flow', 7.0),
        (r'\bFT[-\s]\d+', 5.0),
        (r'\bflow\b', 0.5),        # very low — "flow" appears in menus/TOC
        (r'\bdébit\b', 0.5),
    ],
    "Pression": [
        (r'pressure\s+measurement', 10.0),
        (r'mesure\s+de\s+pression', 10.0),
        (r'pressure\s+transmitter', 8.0),
        (r'transmetteur\s+de\s+pression', 8.0),
        (r'\bPT[-\s]\d+', 5.0),
        (r'\bpressure\b', 1.0),
        (r'\bpression\b', 1.0),
    ],
    "Température": [
        (r'temperature\s+measurement', 10.0),
        (r'mesure\s+de\s+température', 10.0),
        (r'temperature\s+transmitter', 8.0),
        (r'transmetteur\s+de\s+température', 8.0),
        (r'\bTT[-\s]\d+', 5.0),
        (r'\btemperature\b', 0.5),
    ],
    "Analyse procédé": [
        (r'process\s+analysis', 10.0),
        (r'analyse\s+procédé', 10.0),
        (r'\bpH\s+measurement', 8.0),
        (r'conductivity\s+measurement', 8.0),
        (r'oxygen\s+measurement', 8.0),
        (r'\bAT[-\s]\d+', 5.0),
    ],
}

_TYPE_MESURE_MARGIN = 3.0   # winner score must exceed 2nd by this much


def _classify_type_mesure(text: str) -> tuple[str, float] | None:
    """
    Returns (typeMesure, confidence) or None if not confident enough.
    confidence is normalised to 0–1 based on score gap.
    """
    text_lower = text.lower()
    
    # Priority zone: first 2000 characters (typically cover page/title defining the instrument)
    priority_zone = text_lower[:2000]
    
    scores: dict[str, float] = {}

    for type_name, patterns in _TYPE_MESURE_KEYWORDS.items():
        total = 0.0
        for pattern, weight in patterns:
            # Full text baseline matches
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            total += len(matches) * weight
            
            # Boost if found in priority zone
            priority_matches = re.findall(pattern, priority_zone, re.IGNORECASE)
            total += len(priority_matches) * weight * 3.0  # 3x boost for priority zone mentions
        scores[type_name] = total

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return None

    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = best_score - second_score

    if gap < _TYPE_MESURE_MARGIN:
        # Ambiguous — but if best is clearly dominant (>3x second), trust it
        if second_score > 0 and best_score / second_score < 2.0:
            return None

    # Normalise confidence: gap of 10+ → 1.0, gap of 3 → 0.70
    conf = min(1.0, 0.60 + (gap / 30.0))
    return best_name, conf


# ---------------------------------------------------------------------------
# Code — derived from typeMesure + ISA tag scan
# ---------------------------------------------------------------------------

_CODE_FROM_TYPE = {
    "Niveau":          "LT",
    "Débit":           "FT",
    "Pression":        "PT",
    "Température":     "TT",
    "Analyse procédé": "AT",
}

_ISA_TAG_SCAN = re.compile(
    # Must be uppercase to avoid matching "at 4 mA", "pi 3.14", etc.
    r'\b(FT|FI|FQ|FS|FSH|FSL'
    r'|LT|LI|LG|LS|LSH|LSL'
    r'|PT|PI|PG|PS|PDT|PDI'
    r'|TT|TI|TG|TS|TSH|TSL'
    r'|AT|AI'
    r')\s*[-_]?\s*\d+'
    # No re.IGNORECASE — uppercase only
)


def _classify_code(text: str, type_mesure: str | None) -> tuple[str, float] | None:
    # 1. Explicit ISA tag in text (highest confidence)
    match = _ISA_TAG_SCAN.search(text)
    if match:
        tag = match.group(1).upper()
        return tag, 1.0

    # 2. Derive from typeMesure
    if type_mesure and type_mesure in _CODE_FROM_TYPE:
        return _CODE_FROM_TYPE[type_mesure], 0.90

    return None


# ---------------------------------------------------------------------------
# Technologie — keyword patterns per type
# ---------------------------------------------------------------------------

_TECHNOLOGY_PATTERNS: list[tuple[str, str, float]] = [
    # (pattern, technologie_value, weight)
    (r'W.band|80\s*GHz|frequency.modulated\s+radar|FMCW|radar\s+transmitter', "Radar", 10.0),
    (r'guided\s+wave\s+radar|GWR|TDR\s+radar', "Radar guidé", 9.0),
    (r'ultrasonic\s+level|ultrason.*niveau', "Ultrason", 9.0),
    (r'electromagnetic.*flow|débitmètre.*électromagn', "Electromagnétique", 9.0),
    (r'coriolis', "Coriolis", 9.0),
    (r'vortex', "Vortex", 9.0),
    (r'differential\s+pressure|pression\s+différentielle', "Pression différentielle", 9.0),
    (r'capacitive|capacitif', "Capacitif", 8.0),
    (r'hydrostatic|hydrostatique', "À pression hydrostatique", 8.0),
    (r'radiometric|gamma\s+ray|gammastrahlung', "Radiométrique (gamma)", 8.0),
    (r'turbine\s+meter|turbine.*flow', "À turbine", 8.0),
    (r'thermocouple|thermopile|pt100|pt1000|rtd\b', "Thermocouple/RTD", 8.0),
]


def _classify_technologie(text: str) -> tuple[str, float] | None:
    text_lower = text.lower()
    hits: list[tuple[str, float]] = []

    for pattern, tech_name, weight in _TECHNOLOGY_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            hits.append((tech_name, len(matches) * weight))

    if not hits:
        return None

    hits.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = hits[0]
    conf = min(1.0, 0.70 + best_score / 50.0)
    return best_name, conf


# ---------------------------------------------------------------------------
# Marque — brand detection
# ---------------------------------------------------------------------------

_BRAND_PATTERNS: list[tuple[str, str]] = [
    (r'\bSiemens\b', "Siemens"),
    (r'\bKROHNE\b', "KROHNE"),
    (r'\bEndress\s*\+?\s*Hauser\b|E\+H\b', "Endress+Hauser"),
    (r'\bRosemount\b', "Emerson (Rosemount)"),
    (r'\bEmerson\b(?!.*Rosemount)', "Emerson"),
    (r'\bABB\b', "ABB"),
    (r'\bYokogawa\b', "Yokogawa"),
    (r'\bFoxboro\b', "Foxboro"),
    (r'\bWIKA\b', "WIKA"),
    (r'\bVEGA\b', "VEGA"),
    (r'\bBaumer\b', "Baumer"),
    (r'\bSICK\b', "SICK"),
    (r'\bTurck\b', "Turck"),
    (r'\bDanfoss\b', "Danfoss"),
    (r'\bSamson\b', "Samson"),
    (r'\bFlowserve\b', "Flowserve"),
    (r'\bBürkert\b', "Bürkert"),
    (r'\bASCO\b', "ASCO"),
    (r'\bFesto\b', "Festo"),
    (r'\bRotork\b', "Rotork"),
    (r'\bHoneywell\b', "Honeywell"),
    (r'\bSensirion\b', "Sensirion"),
    (r'\bGemü\b', "Gemü"),
    (r'\bPepperl\s*\+?\s*Fuchs\b', "Pepperl+Fuchs"),
]


def _classify_marque(text: str) -> tuple[str, float] | None:
    # Search in first 500 chars and last 200 chars (header/footer) first
    priority_zone = text[:500] + text[-200:]
    for pattern, brand in _BRAND_PATTERNS:
        if re.search(pattern, priority_zone, re.IGNORECASE):
            return brand, 1.0
    # Full text fallback
    for pattern, brand in _BRAND_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return brand, 0.90
    return None


# ---------------------------------------------------------------------------
# Signal sortie
# ---------------------------------------------------------------------------

_SIGNAL_PATTERNS: list[tuple[str, str]] = [
    (r'4\s*[\.…]\s*20\s*mA|4-20\s*mA', "4-20mA"),
    (r'0\s*[\.…]\s*20\s*mA|0-20\s*mA', "0-20mA"),
    (r'0\s*[\.…]\s*10\s*V|0-10\s*V', "0-10V"),
    (r'0\s*[\.…]\s*5\s*V|0-5\s*V', "0-5V"),
]


def _classify_signal(text: str) -> tuple[str, float] | None:
    for pattern, value in _SIGNAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return value, 0.95
    return None


# ---------------------------------------------------------------------------
# HART
# ---------------------------------------------------------------------------

def _classify_hart(text: str) -> tuple[bool, float] | None:
    if re.search(r'\bHART\b', text):
        return True, 1.0
    return None


# ---------------------------------------------------------------------------
# Alimentation (power supply)
# ---------------------------------------------------------------------------

_ALIM_PATTERNS: list[tuple[str, str, float]] = [
    (r'loop[\s-]powered|boucle\s+de\s+courant|2[\s-]wire', "boucle", 1.0),
    (r'24\s*V\s*DC|24\s*VDC', "24VDC", 0.95),
    (r'24\s*V\s*AC|24\s*VAC', "24VAC", 0.95),
    (r'220\s*V\s*AC|230\s*V\s*AC', "220VAC", 0.95),
    (r'12\s*[\.…]\s*35\s*V\s*DC|9\s*[\.…]\s*35\s*V\s*DC', "12-35VDC", 0.90),
    (r'85\s*[\.…]\s*264\s*V\s*AC', "85-264VAC", 0.90),
]


def _classify_alimentation(text: str) -> tuple[str, float] | None:
    for pattern, value, conf in _ALIM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return value, conf
    return None


# ---------------------------------------------------------------------------
# IP rating
# ---------------------------------------------------------------------------

_IP_PATTERN = re.compile(r'IP\s*(6[5-9]|[7-9]\d)\b', re.IGNORECASE)
_NEMA_PATTERN = re.compile(r'NEMA\s*(\w+)', re.IGNORECASE)


def _classify_ip(text: str) -> tuple[str, float] | None:
    # Collect all IP ratings found and take the best
    hits = _IP_PATTERN.findall(text)
    if hits:
        # Return the "worst" (lowest) protection that still appears — 
        # most conservative and most commonly the official rating
        ratings = sorted(set(f"IP{h}" for h in hits))
        # If IP66 and IP67 both present → "IP66/IP67"
        if len(ratings) > 1:
            return "/".join(ratings), 0.95
        return ratings[0], 0.95
    nema = _NEMA_PATTERN.search(text)
    if nema:
        return f"NEMA{nema.group(1).upper()}", 0.90
    return None


# ---------------------------------------------------------------------------
# Process temperature (requires ° symbol)
# ---------------------------------------------------------------------------

_TEMP_PROCESS_PATTERNS = [
    re.compile(
        r'[Pp]rocess\s+temperature\s*[-:]\s*([-+]?\d[\d,. ]*\.{2,3}[-+]?\d[\d,. ]*°[CcFf])',
        re.IGNORECASE
    ),
    re.compile(
        r'[Tt]empérature\s+du\s+produit\s*[-:]\s*([-+]?\d[\d,. ]*\.{2,3}[-+]?\d[\d,. ]*°[CcFf])',
        re.IGNORECASE
    ),
    re.compile(
        r'[Pp]rocess\s+temperature\s+([-+]?\d[\d,.]*\s*[\.…]+\s*[-+]?\d[\d,.]*\s*°[CcFf])',
        re.IGNORECASE
    ),
]


def _classify_temperature_process(text: str) -> tuple[str, float] | None:
    for pattern in _TEMP_PROCESS_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip(), 0.95
    return None


# ---------------------------------------------------------------------------
# Certifications
# ---------------------------------------------------------------------------

_CERT_PATTERNS: list[tuple[str, str]] = [
    (r'\bATEX\b', "ATEX"),
    (r'\bIECEx\b', "IECEx"),
    (r'\bSIL\s*2\b', "SIL 2"),
    (r'\bSIL\s*3\b', "SIL 3"),
    (r'\b(?<!\w)FM\b(?!\w)', "FM"),
    (r'\bCSA\b', "CSA"),
    (r'\bUL\b', "UL"),
]


def _classify_certificats(text: str) -> tuple[list[str], float] | None:
    found = []
    for pattern, cert in _CERT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(cert)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for c in found:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    if unique:
        return unique, 0.95
    return [], 0.80   # explicitly empty is still a valid result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Fields that the pre-classifier always handles — LLM skips these if filled
CLASSIFIER_OWNED_FIELDS = {
    "typeMesure", "code", "technologie", "marque",
    "signalSortie", "hart", "alimentation", "indiceIP",
    "températureProcess", "certificats",
}


def run_pre_extraction(full_text: str) -> dict[str, dict[str, Any]]:
    """
    Run deterministic classification on the full document text.

    Returns a dict:
      field_name -> {"value": ..., "confidence": float, "source": "classifier"}

    These results should be merged into extracted_data BEFORE the LLM runs,
    and should be stored in confidence_data with the returned confidence.

    The LLM extraction loop should skip any field already in extracted_data.
    """
    results: dict[str, dict[str, Any]] = {}

    def _store(field: str, result: tuple | None):
        if result is None:
            return
        value, conf = result
        results[field] = {"value": value, "confidence": conf, "source": "classifier"}

    _store("typeMesure", _classify_type_mesure(full_text))

    # Code depends on typeMesure result
    type_mesure = results.get("typeMesure", {}).get("value")
    _store("code", _classify_code(full_text, type_mesure))

    _store("technologie",        _classify_technologie(full_text))
    _store("marque",             _classify_marque(full_text))
    _store("signalSortie",       _classify_signal(full_text))
    _store("hart",               _classify_hart(full_text))
    _store("alimentation",       _classify_alimentation(full_text))
    _store("indiceIP",           _classify_ip(full_text))
    _store("températureProcess", _classify_temperature_process(full_text))

    cert_result = _classify_certificats(full_text)
    if cert_result is not None:
        value, conf = cert_result
        results["certificats"] = {"value": value, "confidence": conf, "source": "classifier"}

    return results
