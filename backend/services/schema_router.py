"""
Router to identify equipment category and type from document text.
"""

from services.normalize_v2 import detect_isa_tag_info, ISA_TAG_PATTERN

def detect_schema_context(text: str) -> dict:
    """
    Analyzes document text to determine:
    - category (Transmetteur/Capteur or Actionneur)
    - typeMesure or typeActionneur
    - code (ISA tag)
    """
    t = (text or "").lower()
    
    # 1. Look for ISA tag — strongest signal
    isa_info = detect_isa_tag_info(text)
    if isa_info:
        return isa_info

    # 2. Keyword scoring fallback
    actionneur_keywords = [
        "vanne", "valve", "actionneur", "actuator", "cylinder", "vérin",
        "positioner", "positionneur", "control valve", "xv", "mov", "aov", "sov",
        "pic", "tic", "fic", "lic", "plc", "dcs"
    ]
    
    transmitter_keywords = [
        "transmetteur", "transmitter", "capteur", "sensor", "transmitter",
        "pt", "ft", "tt", "lt", "at", "transmitter", "indicateur", "indicator",
        "mesure de", "measurement", "débitmètre", "flowmeter"
    ]

    actionneur_score = sum(1 for k in actionneur_keywords if k in t)
    transmitter_score = sum(1 for k in transmitter_keywords if k in t)

    if actionneur_score > transmitter_score and actionneur_score >= 2:
        return {
            "category": "Actionneur",
            "source": "keywords"
        }
    
    if transmitter_score >= 2:
        return {
            "category": "Transmetteur/Capteur",
            "source": "keywords"
        }

    return {
        "category": "Transmetteur/Capteur", # Default
        "source": "default"
    }
