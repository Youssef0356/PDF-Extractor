"""
Regex-based extractor for deterministic fields.
Scans the full PDF text for well-known patterns, producing high-confidence
results BEFORE the LLM is invoked.  Fields not matched are left for the LLM.
"""
import re
from typing import Optional

# ---------------------------------------------------------------------------
#  Known manufacturer brands (order = priority; first match wins)
# ---------------------------------------------------------------------------
_BRANDS = [
    "Siemens", "KROHNE", "Endress+Hauser", "Endress\\+Hauser",
    "ABB", "Emerson", "Yokogawa", "Honeywell", "Schneider Electric",
    "Schneider", "Vega", "Pepperl+Fuchs", "Pepperl\\+Fuchs",
    "Danfoss", "Bürkert", "Burkert", "Festo", "IFM", "ifm",
    "Wika", "WIKA", "Bosch Rexroth", "Rosemount", "Fisher",
    "Foxboro", "Samson", "Azbil", "Fuji Electric", "Hitachi",
    "Mitsubishi", "Phoenix Contact", "Turck", "Sick", "Balluff",
    "Banner", "Keyence", "Hach", "Mettler Toledo", "Fluke",
]

# Pre-compile brand regex (case-insensitive, word-boundary)
_BRAND_PATTERNS: list[tuple[str, re.Pattern]] = []
for _b in _BRANDS:
    # Normalise display name (remove regex escapes for the canonical name)
    _canonical = _b.replace("\\+", "+")
    # Build pattern: allow the brand to appear at a word boundary
    _pat = re.compile(r"\b" + re.escape(_canonical) + r"\b", re.IGNORECASE)
    _BRAND_PATTERNS.append((_canonical, _pat))


# ---------------------------------------------------------------------------
#  Model patterns (keyed by canonical brand name)
# ---------------------------------------------------------------------------
_MODEL_PATTERNS_BY_BRAND: dict[str, list[re.Pattern]] = {
    "Siemens": [
        re.compile(r"\b(SIMATIC\s+HMI\s+KTP\d{3,4}\s+BASIC)\b", re.IGNORECASE),
        re.compile(r"\b(SIMATIC\s+HMI\b[^\n]{0,60}?\bKTP\d{3,4}\s+BASIC)\b", re.IGNORECASE),
        re.compile(r"\b(SITRANS\s+[A-Z]\d{2,4}(?:/[A-Z]\d{2,4})*)\b", re.IGNORECASE),
        re.compile(r"\b(SIPART\s+PS\d+)\b", re.IGNORECASE),
        re.compile(r"\b(SIMATIC\s+S7-\d{3,4})\b", re.IGNORECASE),
        re.compile(r"\b(CPU\s*\d{3,4}[A-Z]?\s*[A-Z]{2}/[A-Z]{2}/[A-Z]{2})\b", re.IGNORECASE),
        re.compile(r"\b(CPU\s*\d{3,4}[A-Z]?)\b", re.IGNORECASE),
        re.compile(r"\b(SIMATIC\s+\w+(?:\s*[-/]\s*\w+)*)\b", re.IGNORECASE),
    ],
    "KROHNE": [
        re.compile(r"\b(H250\s*[A-Z]?\d*(?:\s*[A-Z]\d+)?)\b", re.IGNORECASE),
        re.compile(r"\b(OPTIFLUX\s*\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(OPTISONIC\s*\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(OPTITEMP\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(OPTIMASS\s*\w+)\b", re.IGNORECASE),
    ],
    "Endress+Hauser": [
        re.compile(r"\b(Promag\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Proline\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Cerabar\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Deltabar\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Liquiline\s*\w+)\b", re.IGNORECASE),
    ],
}

# Generic model fallback: "Model: <value>" or "Type: <value>"
_GENERIC_MODEL_PATTERNS = [
    re.compile(r"(?:Model|Type|Modèle)\s*[:=]\s*([A-Z0-9][A-Z0-9\s\-/\.]{2,30})", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
#  Reference / order-number patterns
# ---------------------------------------------------------------------------
_REFERENCE_PATTERNS = [
    # Siemens 7MF order numbers
    re.compile(r"\b(7MF\s*\d[\d\s\-\.]{5,})\b"),
    # Siemens HMI order numbers like: 6AV2123-2GB03-0AX0
    re.compile(r"\b(6AV\d{4}[-\s]?[0-9A-Z]{5}[-\s]?[0-9A-Z]{4}[-\s]?[0-9A-Z]{4})\b", re.IGNORECASE),
    # Siemens S7 order numbers like: 6ES72141AG400XB0
    re.compile(r"\b(6ES7[0-9A-Z]{8,})\b", re.IGNORECASE),
    # Generic "PA nnn" style
    re.compile(r"\b(PA\s*\d{3,}[\d\.\-\s]*)\b"),
    # Generic "Order No." / "Part No." / "Article No." / "Ref." patterns
    re.compile(
        r"(?:Order\s*(?:No\.?|Number)|Part\s*(?:No\.?|Number)|Article\s*(?:No\.?|Number)|Ref(?:erence)?(?:\s*No\.? )?)\s*[:=]?\s*"
        r"([A-Z0-9][\w\-\./ ]{4,30}(?:\d|[-/.]))",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
#  Tag / reperage patterns
# ---------------------------------------------------------------------------
_TAG_PATTERNS = [
    # ISA-style tags: FT-0201.07, PT-1234, LT-001, TT-100, etc.
    re.compile(r"\b([FPTL][TIRC][\-_]\d{2,}(?:\.\d+)?)\b"),
    # Extended ISA: FIC, FCV, etc.
    re.compile(r"\b([A-Z]{2,4}[\-_]\d{3,}(?:\.\d+)?)\b"),
    # "Tag: <value>" or "Repere: <value>"
    re.compile(r"(?:Tag|Repere|Repérage|Label)\s*[:=]\s*([A-Z0-9][\w\-\.]{3,})", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
#  Small helper
# ---------------------------------------------------------------------------
def _first_match(patterns: list[re.Pattern], text: str) -> Optional[re.Match]:
    """Return the first match from a list of compiled patterns."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m
    return None


def _looks_like_plc_cpu_datasheet(text: str) -> bool:
    """Heuristic: detect PLC/CPU documents to avoid misclassifying as a measurement instrument."""
    t = (text or "").lower()
    keywords = [
        "simatic s7",
        "s7-1200",
        "cpu 12",
        "plc",
        "programmable logic controller",
        "digital input",
        "digital output",
        "analog input",
        "analog output",
    ]
    return any(k in t for k in keywords)


def _all_matches(patterns: list[re.Pattern], text: str) -> list[re.Match]:
    """Return all matches from a list of compiled patterns."""
    matches = []
    for pat in patterns:
        matches.extend(pat.finditer(text))
    return matches


# ---------------------------------------------------------------------------
#  Per-field extractors
# ---------------------------------------------------------------------------

def _extract_type_signal(text: str) -> Optional[dict]:
    """Match output signal type: 4-20mA, 0-20mA, 0-10V, 0-5V."""
    patterns = [
        (re.compile(r"4\s*[\.\-…–]+\s*20\s*mA", re.IGNORECASE), "4-20mA"),
        (re.compile(r"0\s*[\.\-…–]+\s*20\s*mA", re.IGNORECASE), "0-20mA"),
        (re.compile(r"0\s*[\.\-…–]+\s*10\s*V\b", re.IGNORECASE), "0-10V"),
        (re.compile(r"0\s*[\.\-…–]+\s*5\s*V\b", re.IGNORECASE), "0-5V"),
    ]
    for pat, canonical in patterns:
        m = pat.search(text)
        if m:
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


def _extract_communication(text: str) -> Optional[dict]:
    """Match communication protocol keywords."""
    def _is_negated(match_start: int) -> bool:
        # Look slightly before the match for explicit negations like "PROFIBUS: Non".
        start = max(0, match_start - 40)
        window = text[start:match_start].lower()
        return any(tok in window for tok in [": non", " non", ": no", " no", "=0", ": 0"])

    protocols = [
        (re.compile(r"\bPROFIBUS\s+DP\b", re.IGNORECASE), "PROFIBUS DP"),
        (re.compile(r"\bModbus\s+TCP\b", re.IGNORECASE), "Modbus TCP"),
        (re.compile(r"\bModbus\s+RTU\b", re.IGNORECASE), "Modbus RTU"),
        (re.compile(r"\bHART\b"), "HART"),
    ]
    for pat, canonical in protocols:
        m = pat.search(text)
        if m:
            if _is_negated(m.start()):
                continue
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}

    # If the device clearly indicates Ethernet/PROFINET, map to "Autre" (schema doesn't list PROFINET).
    for pat in [
        re.compile(r"\bPROFINET\b", re.IGNORECASE),
        re.compile(r"\bIndustrial\s+Ethernet\b", re.IGNORECASE),
        re.compile(r"\bEthernet\b", re.IGNORECASE),
    ]:
        m = pat.search(text)
        if m and not _is_negated(m.start()):
            return {"value": "Autre", "confidence": 0.9, "quote": m.group(0).strip(), "source": "regex"}
    return None


def _extract_alimentation(text: str) -> Optional[dict]:
    """Match power supply voltage."""
    patterns = [
        (re.compile(r"\b24\s*V\s*DC\b", re.IGNORECASE), "24V DC"),
        (re.compile(r"\b220\s*V\s*AC\b", re.IGNORECASE), "220V AC"),
        (re.compile(r"\b230\s*V\s*AC\b", re.IGNORECASE), "230V AC"),
        (re.compile(r"\b12\s*V\s*DC\b", re.IGNORECASE), "12V DC"),
    ]
    for pat, canonical in patterns:
        m = pat.search(text)
        if m:
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


def _extract_nb_fils(text: str) -> Optional[dict]:
    """Match wire count: 2-wire, 4-wire, etc."""
    patterns = [
        (re.compile(r"\b2[\-\s]*wire\b", re.IGNORECASE), "2"),
        (re.compile(r"\btwo[\-\s]*wire\b", re.IGNORECASE), "2"),
        (re.compile(r"\b4[\-\s]*wire\b", re.IGNORECASE), "4"),
        (re.compile(r"\bfour[\-\s]*wire\b", re.IGNORECASE), "4"),
        (re.compile(r"\b3[\-\s]*wire\b", re.IGNORECASE), "3"),
        (re.compile(r"\bthree[\-\s]*wire\b", re.IGNORECASE), "3"),
    ]
    for pat, canonical in patterns:
        m = pat.search(text)
        if m:
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


def _extract_marque(text: str) -> Optional[dict]:
    """Match known manufacturer brand names."""
    for canonical, pat in _BRAND_PATTERNS:
        m = pat.search(text)
        if m:
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


def _extract_modele(text: str, brand: Optional[str] = None) -> Optional[dict]:
    """Match model name/number, optionally using brand-specific patterns."""
    # Try brand-specific patterns first
    if brand:
        brand_patterns = _MODEL_PATTERNS_BY_BRAND.get(brand, [])
        for pat in brand_patterns:
            m = pat.search(text)
            if m:
                value = m.group(1).strip()
                # Clean up extra whitespace
                value = re.sub(r"\s+", " ", value)
                return {"value": value, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}

    # Try all brand-specific patterns as a fallback
    for _, patterns in _MODEL_PATTERNS_BY_BRAND.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                value = m.group(1).strip()
                value = re.sub(r"\s+", " ", value)
                return {"value": value, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}

    # Generic fallback
    for pat in _GENERIC_MODEL_PATTERNS:
        m = pat.search(text)
        if m:
            value = m.group(1).strip()
            value = re.sub(r"\s+", " ", value)
            return {"value": value, "confidence": 0.8, "quote": m.group(0).strip(), "source": "regex"}

    return None


def _extract_reference(text: str) -> Optional[dict]:
    """Match product reference / order number / part number."""
    for pat in _REFERENCE_PATTERNS:
        m = pat.search(text)
        if m:
            value = m.group(1).strip()
            # Clean up whitespace in references
            value = re.sub(r"\s+", " ", value)
            # Strip common trailing artifacts from PDF headers/footers.
            value = re.sub(r"\s+(?:page|p\.|seite)\s*\d+\b", "", value, flags=re.IGNORECASE).strip()

            if not _looks_like_reference(value):
                continue

            return {"value": value, "confidence": 0.9, "quote": m.group(0).strip(), "source": "regex"}
    return None


def _extract_equipment_name(text: str) -> Optional[dict]:
    """Match a human-readable equipment designation/name from common label patterns."""
    patterns = [
        re.compile(r"(?:Equipment|Device|Instrument|Product)\s*Name\s*[:=]\s*([^\n]{3,80})", re.IGNORECASE),
        re.compile(r"(?:Designation|Désignation)\s*[:=]\s*([^\n]{3,80})", re.IGNORECASE),
        re.compile(r"(?:Instrument)\s*[:=]\s*([^\n]{3,80})", re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            value = m.group(1).strip()
            value = re.sub(r"\s+", " ", value)
            # Avoid obviously-bad captures.
            if len(value) < 3:
                continue
            return {"value": value, "confidence": 0.85, "quote": m.group(0).strip(), "source": "regex"}
    return None


def _extract_categorie(text: str) -> Optional[dict]:
    """Extract equipment category ONLY if explicitly labeled in the document."""
    patterns = [
        re.compile(
            r"(?:Cat(?:égorie|egorie)|Category|Type\s+d['’]?instrument)\s*[:=]\s*(Transmetteur|Actionneur|Autre)",
            re.IGNORECASE,
        ),
    ]

    for pat in patterns:
        m = pat.search(text)
        if m:
            value = m.group(1)
            # Normalize capitalization to match allowed values.
            canonical = value[0].upper() + value[1:].lower()
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}

    return None


def _extract_type_mesure(text: str) -> Optional[dict]:
    """Infer measurement type from document keywords."""
    text_lower = text.lower()

    scores = {
        "Pression": (
            len(re.findall(r"\bpressure\b", text_lower))
            + len(re.findall(r"\bpression\b", text_lower))
            + len(re.findall(r"\bdifferential\s+pressure\b", text_lower))
            + len(re.findall(r"\bgauge\s+pressure\b", text_lower))
        ),
        "Debit": (
            len(re.findall(r"\bflow\b", text_lower))
            + len(re.findall(r"\bdebit\b", text_lower))
            + len(re.findall(r"\bdébit\b", text_lower))
            + len(re.findall(r"\bflowmeter\b", text_lower))
            + len(re.findall(r"\bflow\s*meter\b", text_lower))
        ),
        "Niveau": (
            len(re.findall(r"\blevel\b", text_lower))
            + len(re.findall(r"\bniveau\b", text_lower))
            + len(re.findall(r"\blevel\s+transmitt", text_lower))
        ),
        "Temperature": (
            len(re.findall(r"\btemperature\s+(?:transmitt|sensor|measur)\b", text_lower))
            + len(re.findall(r"\btempérature\b", text_lower))
            + len(re.findall(r"\bthermocouple\b", text_lower))
            + len(re.findall(r"\bRTD\b", text))  # case-sensitive
        ),
    }

    # Find the dominant measurement type
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score >= 3:
        return {"value": best_type, "confidence": 0.85, "quote": f"{best_type.lower()} keywords found {best_score}x", "source": "regex"}
    return None


def _extract_plage_mesure(text: str) -> Optional[dict]:
    """
    Match measurement range patterns like:
    - "40 ... 1200 L/h"
    - "0.01 ... 1 bar"
    - "min ... max unit"
    """
    # Pattern: number (separator) number (unit)
    range_pat = re.compile(
        r"(\d+(?:\.\d+)?)\s*"              # min value
        r"(?:\.{2,3}|…|–|\-|to)\s*"        # separator
        r"(\d+(?:\.\d+)?)\s*"              # max value
        r"(L/h|m³/h|m3/h|bar|mbar|kPa|MPa|psi|°C|°F|mm|m|Pa|inH2O)\b",  # unit
        re.IGNORECASE,
    )

    matches = list(range_pat.finditer(text))
    if not matches:
        return None

    # Pick the first range that appears in a "measuring range" or "span" context
    for m in matches:
        start = max(0, m.start() - 200)
        context = text[start:m.start()].lower()
        if any(kw in context for kw in [
            "measuring range", "span", "plage", "range",
            "measuring span", "étendue", "mesure",
        ]):
            try:
                min_val = float(m.group(1))
                max_val = float(m.group(2))
                unite = m.group(3)
                return {
                    "value": {"min": min_val, "max": max_val, "unite": unite},
                    "confidence": 0.85,
                    "quote": m.group(0).strip(),
                    "source": "regex",
                }
            except ValueError:
                continue

    return None


def _extract_reperage(text: str) -> Optional[dict]:
    """Match instrument tag/label patterns."""
    for pat in _TAG_PATTERNS:
        m = pat.search(text)
        if m:
            value = m.group(1) if pat.groups else m.group(0)
            return {"value": value.strip(), "confidence": 0.9, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def extract_with_regex(full_text: str) -> dict[str, dict]:
    """
    Scan the full PDF text with regex patterns for all deterministic fields.

    Returns:
        Dict mapping field_name -> {"value", "confidence", "quote", "source"}
        for each field that was confidently matched.
        Fields NOT matched are simply omitted from the result.
    """
    results: dict[str, dict] = {}

    # --- Direct pattern matches (high priority) ---
    r = _extract_type_signal(full_text)
    if r:
        results["typeSignal"] = r

    r = _extract_communication(full_text)
    if r:
        results["communication"] = r

    r = _extract_alimentation(full_text)
    if r:
        results["alimentation"] = r

    r = _extract_nb_fils(full_text)
    if r:
        results["nbFils"] = r

    r = _extract_marque(full_text)
    if r:
        results["marque"] = r

    # Model extraction benefits from knowing the brand
    brand = results.get("marque", {}).get("value")
    r = _extract_modele(full_text, brand=brand)
    if r:
        results["modele"] = r

    r = _extract_reference(full_text)
    if r:
        results["reference"] = r

    # If we have a Siemens order number, lock brand/model to Siemens/HMI patterns
    # to avoid picking up "compatible" brands/models elsewhere in the document.
    ref_val = (results.get("reference", {}) or {}).get("value")
    if isinstance(ref_val, str):
        ref_norm = re.sub(r"\s+", "", ref_val).upper()
        if ref_norm.startswith("6AV") or ref_norm.startswith("6ES7") or ref_norm.startswith("7MF"):
            results["marque"] = {"value": "Siemens", "confidence": 1.0, "quote": ref_val, "source": "regex"}
            hm = re.search(r"\bKTP\d{3,4}\s+BASIC\b", full_text, re.IGNORECASE)
            if hm:
                results["modele"] = {
                    "value": f"SIMATIC HMI {hm.group(0).strip()}",
                    "confidence": 0.9,
                    "quote": hm.group(0).strip(),
                    "source": "regex",
                }

    r = _extract_equipment_name(full_text)
    if r:
        results["equipmentName"] = r

    r = _extract_reperage(full_text)
    if r:
        results["reperage"] = r

    # --- Keyword-frequency inference (lower priority) ---
    # IMPORTANT: For PLC/CPU datasheets, these heuristics are a frequent source of
    # semantic errors (e.g. timing ranges interpreted as measurement ranges).
    if not _looks_like_plc_cpu_datasheet(full_text):
        r = _extract_categorie(full_text)
        if r:
            results["categorie"] = r

        r = _extract_type_mesure(full_text)
        if r:
            results["typeMesure"] = r

        r = _extract_plage_mesure(full_text)
        if r:
            results["plageMesure"] = r

    return results
