"""
V2 value normalization and alias resolution.

Fixes in this version:
- plageMesureUnite: rejects pure-number values (LLM was copying the max value instead of the unit)
- températureProcess: rejects values without a degree symbol (e.g. "Autre: 3,23" → null)
- nombreFils: passes int values through as-is (LLM correctly returns bare integers like 4)
- All string values stripped of leading/trailing whitespace before matching
- "Autre: X" pattern: preserves original value in suffix for downstream display
"""

import re
from typing import Any
from models.field_options import (
    CATEGORY_OPTIONS,
    TYPE_MESURE_OPTIONS,
    TYPE_ACTIONNEUR_OPTIONS,
    ALL_CODES,
    ALL_TECHNOLOGIES,
    SIGNAL_SORTIE_OPTIONS,
    ALIMENTATION_OPTIONS,
    COMMUNICATION_OPTIONS,
    NOMBRE_FILS_OPTIONS,
    MARQUE_OPTIONS,
    MATERIAU_MEMBRANE_OPTIONS,
    FIRST_LETTER_TO_TYPE,
)

# ISA 5.1 tag pattern (FT-101, PT-302, etc.)
ISA_TAG_PATTERN = re.compile(
    r'\b(FT|FI|FQ|FS|FSH|FSL'
    r'|LT|LI|LG|LS|LSH|LSL'
    r'|PT|PI|PG|PS|PDT|PDI'
    r'|TT|TI|TG|TS|TSH|TSL'
    r'|AT|AI|pHT|O2T|COT|CO2T'
    r'|CV|PCV|FCV|LCV|TCV'
    r'|XV|MOV|AOV|SOV|SDV'
    r'|ACT|CYL|HCY|MTR|VSD'
    r'|PIC|TIC|FIC|LIC|PLC|DCS'
    r'|PSV|PRV|BDV|DMP|FIL'
    r')\s*[-_]?\s*\d+',
    re.IGNORECASE
)

# Regex: detects a temperature range string like "-40...+300°C" or "-20 to 85°C"
_TEMP_RANGE_PATTERN = re.compile(
    r'[-+]?\d[\d,.]* *(\.{2,3}|to|à|~|/) *[-+]?\d[\d,.]*.*°[CcFfKk]'
    r'|[-+]?\d[\d,.]* *°[CcFfKk] *(\.{2,3}|to|à|~|/) *[-+]?\d[\d,.]*.*°[CcFfKk]',
    re.IGNORECASE
)
# Simpler: anything with a degree sign and at least one digit is probably OK
_HAS_DEGREE = re.compile(r'\d.*°[CcFfKk]|°[CcFfKk].*\d', re.IGNORECASE)

# Regex: detects a pure number or number with unit that is NOT a temperature
_PURE_NUMBER = re.compile(r'^[-+]?\d[\d,. ]*$')


def normalize_value_v2(field_name: str, value: Any) -> Any:
    """Canonicalize extracted values for V2 schema."""
    if value is None:
        return None

    # ── Numeric fields — always convert first ─────────────────────────────
    if field_name in ("plageMesureMin", "plageMesureMax", "seuil", "courseMM", "forceN", "pressionAlimentationBar"):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            val_clean = value.strip().replace(",", ".")
            match = re.search(r"[-+]?\d*\.?\d+", val_clean)
            if match:
                try:
                    return float(match.group())
                except (ValueError, TypeError):
                    pass
        return None

    # ── Boolean fields ────────────────────────────────────────────────────
    if field_name in ("hart", "sortieTOR"):
        if isinstance(value, bool):
            return value
        val_s = str(value).strip().lower()
        if val_s in ("true", "yes", "oui", "1", "vrai"):
            return True
        if val_s in ("false", "no", "non", "0", "faux"):
            return False
        return None

    # ── nombreFils — schema stores int (2, 3, 4, 5) ──────────────────────
    # The LLM correctly returns bare integers (e.g. 4). Just pass them through.
    # Also handle string variants like "4 fils", "4-wire" for robustness.
    if field_name == "nombreFils":
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            match = re.search(r'\b([2-5])\b', value.strip())
            if match:
                return int(match.group(1))
        return value

    # ── plageMesureUnite — must be a real unit, not a number ─────────────
    if field_name == "plageMesureUnite":
        if not isinstance(value, str):
            return None
        v = value.strip()
        # Reject if it looks like a pure number (LLM copying max value)
        if _PURE_NUMBER.match(v):
            return None
        # Reject "Autre: <number>"
        if re.match(r'^[Aa]utre\s*:\s*[-\d.,\s]+$', v):
            return None
        return v if v else None

    # ── températureProcess — must look like a real temperature ───────────
    if field_name == "températureProcess":
        if not isinstance(value, str):
            return None
        v = value.strip()
        # Check if it has a degree sign with a digit nearby → valid
        if _HAS_DEGREE.search(v):
            return v
        # Also accept "Autre: -40...+85°C" style if degree present
        if "autre" in v.lower() and _HAS_DEGREE.search(v):
            # Extract just the temperature part
            after_colon = v.split(":", 1)[-1].strip()
            return after_colon if after_colon else v
        # Reject: "Autre: 3,23" or pure numbers or random text without °
        return None

    # ── From here on, value must be a string ─────────────────────────────
    if not isinstance(value, str):
        return value

    v = value.strip()
    if not v:
        return None
    vl = v.lower()

    # Handle "Autre: <original>" — preserve the original text in the suffix
    original_extracted: str | None = None
    if re.match(r'^[Aa]utre\s*:', v):
        original_extracted = v.split(":", 1)[-1].strip()
        v = "Autre"
        vl = "autre"

    # ── CATEGORY ─────────────────────────────────────────────────────────
    if field_name == "category":
        for opt in CATEGORY_OPTIONS:
            if opt.lower() in vl:
                return opt
        return "Autre"

    # ── TYPE MESURE ───────────────────────────────────────────────────────
    if field_name == "typeMesure":
        aliases = {
            "pression": "Pression", "pressure": "Pression", "druck": "Pression",
            "debit": "Débit", "débit": "Débit", "flow": "Débit", "durchfluss": "Débit",
            "niveau": "Niveau", "level": "Niveau", "füllstand": "Niveau",
            "temperature": "Température", "température": "Température", "temperatur": "Température",
            "analyse": "Analyse procédé", "analyzer": "Analyse procédé",
        }
        for k, target in aliases.items():
            if k in vl:
                return target
        return "Autre"

    # ── CODE ──────────────────────────────────────────────────────────────
    if field_name == "code":
        v_upper = v.upper().strip()
        if v_upper in ALL_CODES:
            return v_upper
        for c in sorted(ALL_CODES, key=len, reverse=True):
            if c in v_upper:
                return c
        return "Autre"

    # ── TECHNOLOGIE ───────────────────────────────────────────────────────
    if field_name == "technologie":
        for t in ALL_TECHNOLOGIES:
            if t.lower() in vl:
                return t
        if original_extracted:
            return f"Autre: {original_extracted}"
        return "Autre"

    # ── SIGNAL SORTIE ─────────────────────────────────────────────────────
    if field_name == "signalSortie":
        # Normalize spaces and separators
        v_norm = v.replace(" ", "").replace("...", "-").replace("…", "-").replace(",", ".").lower()
        for opt in SIGNAL_SORTIE_OPTIONS:
            if opt.replace(" ", "").lower() == v_norm:
                return opt
        for opt in SIGNAL_SORTIE_OPTIONS:
            if opt.replace(" ", "").lower() in v_norm:
                return opt
        if original_extracted:
            return f"Autre: {original_extracted}"
        return "Autre"

    # ── ALIMENTATION ──────────────────────────────────────────────────────
    if field_name == "alimentation":
        if any(x in vl for x in ("loop", "boucle", "2 fils", "2-wire", "2wire")):
            return "boucle"
        if "24" in vl and "dc" in vl:
            return "24VDC"
        if "24" in vl and "ac" in vl:
            return "24VAC"
        if "12" in vl and "30" in vl:
            return "12-30VDC"
        if "85" in vl and "264" in vl:
            return "85-264VAC"
        if "220" in vl:
            return "220VAC"
        for opt in ALIMENTATION_OPTIONS:
            if opt.lower() in vl:
                return opt
        if original_extracted:
            return f"Autre: {original_extracted}"
        return "Autre"

    # ── COMMUNICATION ─────────────────────────────────────────────────────
    if field_name == "communication":
        if any(x in vl for x in ("non", "aucun", "none", "no ", "pas de")):
            return "non"
        # HART check first (also used as a separate boolean field)
        for opt in COMMUNICATION_OPTIONS:
            if opt.lower() in vl:
                return opt
        if original_extracted:
            return f"Autre: {original_extracted}"
        return "Autre"

    # ── MARQUE ────────────────────────────────────────────────────────────
    if field_name == "marque":
        for opt in MARQUE_OPTIONS:
            if opt == "Emerson (Rosemount)":
                if "emerson" in vl or "rosemount" in vl:
                    return opt
            elif opt == "Emerson (Fisher)":
                if "fisher" in vl:
                    return opt
            elif opt.lower() in vl:
                return opt
        if original_extracted:
            return f"Autre: {original_extracted}"
        return v  # Return as-is for unknown brands (open field)

    # ── MATERIAU MEMBRANE ─────────────────────────────────────────────────
    if field_name == "matériauMembrane":
        for opt in MATERIAU_MEMBRANE_OPTIONS:
            if opt.lower() in vl:
                return opt
        if original_extracted:
            return f"Autre: {original_extracted}"
        return "Autre"

    # ── INDICE IP ─────────────────────────────────────────────────────────
    if field_name == "indiceIP":
        # Look for IP## pattern
        ip_match = re.search(r'IP\s*(\d{2,3}[A-Z]?)', v, re.IGNORECASE)
        if ip_match:
            return f"IP{ip_match.group(1).upper()}"
        nema_match = re.search(r'NEMA\s*(\w+)', v, re.IGNORECASE)
        if nema_match:
            return f"NEMA{nema_match.group(1).upper()}"
        return v

    # ── TYPE MESURE ACTIONNEUR ────────────────────────────────────────────
    if field_name == "typeActionneur":
        for opt in TYPE_ACTIONNEUR_OPTIONS:
            if opt.lower() in vl:
                return opt
        return "Autre"

    # ── certificats — handled as list upstream, but normalize individual items
    if field_name == "certificats":
        # This field is a list — individual items are normalized by the extractor
        return v

    # ── Fallback for open text fields (precision, plageMesureUnite, etc.) ─
    if original_extracted:
        return f"Autre: {original_extracted}"
    return v


def detect_isa_tag_info(text: str) -> dict | None:
    """Extract category, code, and type from an ISA tag if found."""
    match = ISA_TAG_PATTERN.search(text)
    if not match:
        return None

    tag = match.group(1).upper()
    first_letter = tag[0]

    type_mesure = FIRST_LETTER_TO_TYPE.get(first_letter)
    category = "Actionneur" if first_letter in ("C", "X", "M", "D", "H", "V", "B") else "Transmetteur/Capteur"

    return {
        "category": category,
        "code": tag,
        "typeMesure": type_mesure,
        "source": "isa_tag"
    }