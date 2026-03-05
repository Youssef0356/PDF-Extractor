"""Document type classifier.

Provides lightweight heuristics to identify broad document categories so extraction
can be constrained to the relevant schema.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentContext:
    doc_type: str
    confidence: float
    rationale: str


def classify_document(full_text: str) -> DocumentContext:
    t = (full_text or "").lower()

    hmi_keywords = [
        "hmi",
        "instruction sheet",
        "touchscreen",
        "tft lcd",
        "panel type",
        "resolution",
        "backlight",
        "screen",
        "operation temperature",
        "storage temperature",
        "mounting",
        "usb host",
        "usb slave",
        "ethernet",
        "rs-232",
        "rs-485",
    ]

    instr_keywords = [
        "transmitter",
        "transmetteur",
        "flowmeter",
        "débitmètre",
        "pressur",
        "pression",
        "level transmitter",
        "temperature transmitter",
        "hart",
        "4-20",
        "measuring range",
        "plage de mesure",
        "rangeability",
    ]

    hmi_score = sum(1 for k in hmi_keywords if k in t)
    instr_score = sum(1 for k in instr_keywords if k in t)

    if hmi_score >= max(3, instr_score + 2):
        return DocumentContext(
            doc_type="hmi_manual",
            confidence=min(1.0, 0.25 + (hmi_score / 20.0)),
            rationale=f"hmi_score={hmi_score}, instrument_score={instr_score}",
        )

    if instr_score >= max(3, hmi_score + 2):
        return DocumentContext(
            doc_type="instrument_datasheet",
            confidence=min(1.0, 0.25 + (instr_score / 20.0)),
            rationale=f"instrument_score={instr_score}, hmi_score={hmi_score}",
        )

    return DocumentContext(
        doc_type="unknown",
        confidence=0.2,
        rationale=f"hmi_score={hmi_score}, instrument_score={instr_score}",
    )
