"""
V2 value normalization and alias resolution.
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


def normalize_value_v2(field_name: str, value: Any) -> Any:
    """Canonicalize extracted values for V2 schema."""
    if value is None:
        return None

    # Handle "Autre: custom text" from LLM
    original_extracted = None
    if isinstance(value, str) and value.startswith("Autre:"):
        original_extracted = value.replace("Autre:", "").strip()
        value = "Autre"

    # Numeric conversion for specific fields (ALWAYS TRY THIS FIRST)
    if field_name in ("plageMesureMin", "plageMesureMax", "seuil", "courseMM", "forceN", "pressionAlimentationBar"):
        # If it's a string, try to extract the first number found
        if isinstance(value, str):
            # Replace comma with dot for float parsing
            val_clean = value.replace(",", ".")
            # Extract digits, dot, and minus sign
            match = re.search(r"[-+]?\d*\.?\d+", val_clean)
            if match:
                try:
                    return float(match.group())
                except (ValueError, TypeError):
                    pass
        elif isinstance(value, (int, float)):
            return float(value)
        return None

    if not isinstance(value, str):
        # Handle boolean conversion for hart/sortieTOR if they come as strings
        if field_name in ("hart", "sortieTOR"):
            if isinstance(value, bool): return value
            val_s = str(value).lower()
            if val_s in ("true", "yes", "oui", "1"): return True
            if val_s in ("false", "no", "non", "0"): return False
        
        # Handle numeric conversion for nombreFils
        if field_name == "nombreFils":
            try:
                val = int(value)
                return val if val in NOMBRE_FILS_OPTIONS else None
            except (ValueError, TypeError):
                return None
        
        return value

    v = value.strip()
    vl = v.lower()

    # Re-check boolean strings for field_name == "hart" or "sortieTOR"
    if field_name in ("hart", "sortieTOR"):
        if vl in ("true", "yes", "oui", "1"): return True
        if vl in ("false", "no", "non", "0"): return False

    # --- CATEGORY ---
    if field_name == "category":
        for opt in CATEGORY_OPTIONS:
            if opt.lower() in vl: return opt
        return "Autre"

    # --- TYPE MESURE ---
    if field_name == "typeMesure":
        # Check aliases
        aliases = {
            "pression": "Pression", "pressure": "Pression", "druck": "Pression",
            "debit": "Débit", "débit": "Débit", "flow": "Débit", "durchfluss": "Débit",
            "niveau": "Niveau", "level": "Niveau", "füllstand": "Niveau",
            "temperature": "Température", "température": "Température", "temperatur": "Température",
            "analyse": "Analyse procédé", "analyzer": "Analyse procédé",
        }
        for k, target in aliases.items():
            if k in vl: return target
        return "Autre"

    # --- CODE ---
    if field_name == "code":
        v_upper = v.upper()
        if v_upper in ALL_CODES: return v_upper
        # Try finding anywhere in string
        for c in ALL_CODES:
            if c in v_upper: return c
        return "Autre"

    # --- TECHNOLOGIE ---
    if field_name == "technologie":
        for t in ALL_TECHNOLOGIES:
            if t.lower() in vl: return t
        return f"Autre: {original_extracted}" if original_extracted else "Autre"

    # --- SIGNAL ---
    if field_name == "signalSortie":
        v_norm = v.replace(" ", "").replace("...", "-").replace("…", "-").lower()
        for opt in SIGNAL_SORTIE_OPTIONS:
            if opt.replace(" ", "").lower() in v_norm: return opt
        return f"Autre: {original_extracted}" if original_extracted else "Autre"

    # --- ALIMENTATION ---
    if field_name == "alimentation":
        if any(x in vl for x in ("loop", "boucle", "2 fils", "2-wire")): return "boucle"
        if "24" in vl and "dc" in vl: return "24VDC"
        if "24" in vl and "ac" in vl: return "24VAC"
        if "220" in vl: return "220VAC"
        for opt in ALIMENTATION_OPTIONS:
            if opt.lower() in vl: return opt
        return f"Autre: {original_extracted}" if original_extracted else "Autre"

    # --- COMMUNICATION ---
    if field_name == "communication":
        if "non" in vl or "aucun" in vl or "none" in vl: return "non"
        for opt in COMMUNICATION_OPTIONS:
            if opt.lower() in vl: return opt
        return f"Autre: {original_extracted}" if original_extracted else "Autre"

    # --- MARQUE ---
    if field_name == "marque":
        for opt in MARQUE_OPTIONS:
            # Handle Emerson/Rosemount alias
            if opt == "Emerson (Rosemount)":
                if "emerson" in vl or "rosemount" in vl: return opt
            if opt.lower() in vl: return opt
        return f"Autre: {original_extracted}" if original_extracted else "Autre"

    # --- MATERIAU MEMBRANE ---
    if field_name == "matériauMembrane":
        for opt in MATERIAU_MEMBRANE_OPTIONS:
            if opt.lower() in vl: return opt
        return f"Autre: {original_extracted}" if original_extracted else "Autre"

    # Final "Autre" formatting for open strings
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
    
    # Map first letter to typeMesure
    type_mesure = FIRST_LETTER_TO_TYPE.get(first_letter)
    
    # Map first letter to category
    category = "Actionneur" if first_letter in ("C", "X", "M", "D", "H", "V", "B") else "Transmetteur/Capteur"
    
    return {
        "category": category,
        "code": tag,
        "typeMesure": type_mesure,
        "source": "isa_tag"
    }
