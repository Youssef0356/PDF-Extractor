"""
LLM extraction service.
Uses Ollama (Qwen2.5) to extract structured data from document chunks.
"""
import json
import ollama as ollama_client
import re
import time
from datetime import datetime

from config import ACCURACY_FIRST, TEXT_MODEL
from models.schema import FIELD_DESCRIPTIONS, EquipmentSchema
from services.document_classifier import DocumentContext


def _parse_llm_json(raw: str) -> dict:
    """Parse the model response into a dict.

    The model sometimes wraps JSON in markdown fences or emits extra text.
    This function extracts the first top-level JSON object and parses it.
    """
    if raw is None:
        return {"value": None, "confidence": 0.0, "quote": None}

    s = str(raw).strip()
    if not s:
        return {"value": None, "confidence": 0.0, "quote": None}

    # Strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # Extract first JSON object by brace matching
    start = s.find("{")
    if start == -1:
        return {"value": None, "confidence": 0.0, "quote": None}

    depth = 0
    end = None
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    payload = s[start:end] if end is not None else s[start:]
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else {"value": None, "confidence": 0.0, "quote": None}
    except Exception:
        return {"value": None, "confidence": 0.0, "quote": None}


# ---------------------------------------------------------------------------
# Per-field prompt rules + few-shot examples
# ---------------------------------------------------------------------------

FIELD_PROMPT_RULES: dict[str, str] = {
    "equipmentName": (
        'Extract the commercial equipment designation as a short human-readable name (product/family + type if present).\n'
        'Prefer the exact name as printed on the cover/title/header (e.g. "SITRANS P320", "OPTIFLUX 2000", "SIMATIC S7-1200", "LOGO! 8").\n'
        'If the document is about a device that is primarily identified by a type number, include the brand when available (e.g. "SAMSON Type 3271").\n'
        'NEVER return: plant tags (e.g. FT-101), page/section numbers (e.g. "4-4"), or pure numbers (e.g. "3271").\n'
        'NEVER return part/order/article numbers (patterns: 6AV…, 6ES7…, 7MF…, long alphanumeric codes with many digits).\n'
        'If you cannot form a human-readable designation from the excerpts → null.\n'
        "GOOD: 'SAMSON Type 3271' | GOOD: 'SITRANS P320' | BAD: '3271' | BAD: 'FT-101' | BAD: '6ES7 153-1AA00-0XB0'"
    ),

    "reference": (
        'Extract the manufacturer part number / order number / article number '
        '(e.g. "6ES7 315-2EH14-0AB0", "7MF4433-1DA02", "FMR51-AAACBMJF2A4").\n'
        'This is the alphanumeric code used to order the product — NOT the commercial name.\n'
        "GOOD: '6ES7 315-2EH14-0AB0' | BAD: 'SIMATIC S7-300', 'transmitter'"
    ),

    "fabricant": (
        'Extract the manufacturer / brand name as printed in the document '
        '(e.g. "Siemens", "Endress+Hauser", "Krohne", "Yokogawa", "ABB", "Emerson").\n'
        'Return only the company name, not a product family.\n'
        "GOOD: 'Endress+Hauser' | BAD: 'SITRANS', 'Series 3000'"
    ),

    "marque": (
        'Extract the manufacturer / brand name (company) as printed in the document (e.g. "KROHNE", "Siemens").\n'
        'NEVER return numbers (page numbers, section numbers, years, order codes).\n'
        'If the only candidates are numbers or codes, return null.'
    ),

    "modele": (
        'Extract the model designation (e.g. "H250 M40", "SITRANS P320").\n'
        'NEVER return pure numbers like "18" or "7.7" (these are almost always page/section numbers).\n'
        'If multiple model strings appear, prefer the one repeated in headers/cover (often near the top of pages).'
    ),

    "categorie": (
        'Classify the equipment into the SINGLE most specific category that matches the document.\n'
        'Return EXACTLY one of the allowed categorie values (case and spacing must match).\n'
        '\n'
        '=== MEASUREMENT INSTRUMENTS ===\n'
        '  Pressure transmitter\n'
        '  Pressure gauge\n'
        '  Differential pressure transmitter\n'
        '  Temperature transmitter\n'
        '  Temperature sensor\n'
        '  Temperature indicator\n'
        '  Flowmeter\n'
        '  Electromagnetic flowmeter\n'
        '  Coriolis flowmeter\n'
        '  Vortex flowmeter\n'
        '  Ultrasonic flowmeter\n'
        '  Variable area flowmeter\n'
        '  Turbine flowmeter\n'
        '  Positive displacement meter\n'
        '  Pitot tube / annubar\n'
        '  Orifice plate\n'
        '  Level transmitter\n'
        '  Level switch\n'
        '  Radar level sensor\n'
        '  Ultrasonic level sensor\n'
        '  Float level switch\n'
        '  Capacitive level sensor\n'
        '  Vibrating fork / tuning fork\n'
        '  Density meter\n'
        '  Viscosity meter\n'
        '\n'
        '=== ANALYTICAL INSTRUMENTS ===\n'
        '  pH sensor / transmitter\n'
        '  Conductivity sensor\n'
        '  Dissolved oxygen sensor\n'
        '  Turbidity sensor\n'
        '  Gas analyzer\n'
        '  Flame detector\n'
        '  Gas detector\n'
        '  Moisture analyzer\n'
        '  Particle counter\n'
        '  Chromatograph\n'
        '\n'
        '=== FINAL CONTROL ELEMENTS ===\n'
        '  Control valve\n'
        '  Valve positioner\n'
        '  Pneumatic actuator\n'
        '  Electric actuator\n'
        '  Solenoid valve\n'
        '  Safety relief valve\n'
        '  Butterfly valve\n'
        '  Pump\n'
        '  Variable speed drive\n'
        '\n'
        '=== AUTOMATION & CONTROL ===\n'
        '  PLC\n'
        '  DCS controller\n'
        '  PID controller\n'
        '  Remote I/O\n'
        '  Safety system (SIS)\n'
        '\n'
        '=== HUMAN INTERFACE & DISPLAY ===\n'
        '  HMI panel\n'
        '  Field indicator\n'
        '  Paperless recorder\n'
        '  SCADA / RTU\n'
        '\n'
        '=== POWER & SIGNAL CONDITIONING ===\n'
        '  Signal isolator / barrier\n'
        '  Power supply\n'
        '  Signal converter\n'
        '\n'
        '=== OTHER ===\n'
        '  Weighing system\n'
        '  Vibration sensor\n'
        '  Position sensor\n'
        '  Speed sensor\n'
        '  Torque sensor\n'
        '  Noise / acoustic sensor\n'
        '  Autre\n'
        '\n'
        'RULES:\n'
        '  - Pick the MOST SPECIFIC match, not a generic one.\n'
        '  - A rotameter → "Variable area flowmeter", not "Flowmeter".\n'
        '  - A PT100 without transmitter → "Temperature sensor", not "Temperature transmitter".\n'
        '  - A valve + positioner + actuator as one assembly → "Control valve".\n'
        'GOOD: "Variable area flowmeter" | BAD: "Flowmeter", "Measurement device", "Sensor"'
    ),

    "typeMesure": (
        'Extract the physical quantity being measured. Use ONE of: Pression, Debit, Niveau, Temperature, Autre.\n'
        'Match regardless of language:\n'
        '  flow/débit/durchfluss → Debit\n'
        '  pressure/pression/druck → Pression\n'
        '  level/niveau/füllstand → Niveau\n'
        '  temperature/température/temperatur → Temperature\n'
        'A flow meter (débitmètre) always measures Debit.\n'
        "GOOD: 'Debit' | BAD: 'flow measurement', 'débit de liquides'"
    ),

    "technologie": (
        'Extract the measurement/actuation technology principle EXACTLY as described in the document.\n'
        'Do not translate or reformat — return a short French or bilingual label.\n'
        'Common examples: Electromagnetique, Piezo-resistif, Ultrasonique, Coriolis, Vortex, '
        'Radar, Capacitif, Pneumatique, Hydraulique, Section variable (flotteur), TFT Tactile, Autre.\n'
        'If the actuator uses compressed air / air supply / "air" / "pneumatique" → technologie = "Pneumatique".\n'
        'For variable-area / rotameter / float-tube devices → "Section variable (flotteur)"\n'
        'NEVER return numbers (page/section numbers like "18"). If the principle is not stated, return null.\n'
        "GOOD: 'Section variable (flotteur)', 'Electromagnetique', 'Coriolis' | BAD: 'float principle', 'variable area flowmeter'"
    ),

    "typeSignal": (
        'Extract the output signal type (e.g. "4-20mA", "0-20mA", "0-10V", "HART", "Profibus PA").\n'
        'Normalise to compact form: "4-20mA", "0-10V", etc.\n'
        'If multiple signals exist, return the PRIMARY/analog output.\n'
        "GOOD: '4-20mA' | BAD: '4 to 20 milliampere', 'analog output'"
    ),

    "plageMesure": (
        'Extract the measuring range as {min, max, unite}.\n'
        'Rules:\n'
        '  - For numeric ranges: min and max are numbers, unite is the unit string.\n'
        '  - CRITICAL: Ignore rangeability / turndown ratios (e.g. "100 : 1", "10:1"). DO NOT extract min=1, max=100. Return null if no true measurement range (like 0 to 10 m3/h) is given.\n'
        '  - Never use "1" or "ratio" as a unit. Units must be physical (e.g., m3/h, bar, °C).\n'
        '  - Ignore: time delays (ms/µs), supply voltages, and ratios.\n'
        '  - The quote MUST contain the unit AND the numeric bounds you return (e.g. "10 ... 100 l/h").\n'
        'BAD: {"min": 1, "max": 100, "unite": "1"} ← Ratios are NOT ranges.\n'
        'BAD: {"min": 0, "max": 20, "unite": "ms"} ← this is a time delay'
    ),

    "alimentation": (
        'Extract the supply voltage/power specification as printed '
        '(e.g. "24V DC", "220V AC", "10.5…30V DC", "85…264V AC").\n'
        'Normalise: Vdc→V DC, VAC→V AC.\n'
        "GOOD: '24V DC' | BAD: 'loop powered', 'external supply'"
    ),

    "nbFils": (
        'Extract the number of wires for the electrical connection (e.g. "2 fils", "4 fils").\n'
        '2-wire/2 fils/loop-powered → "2 fils" | 4-wire/4 fils → "4 fils".\n'
        "GOOD: '2 fils' | BAD: '2-wire transmitter', 'loop'"
    ),

    "communication": (
        'Extract the digital communication protocol if present '
        '(e.g. "HART", "PROFIBUS DP", "Profibus PA", "Foundation Fieldbus", "Profinet", '
        '"Modbus RTU", "Modbus TCP", "RS-485", "Ethernet").\n'
        'If the device has no digital protocol (analog only), return null.\n'
        "GOOD: 'HART' | BAD: '4-20mA with HART option', 'serial'"
    ),

    "classeProtection": (
        'Extract the IP/NEMA protection class exactly as written (e.g. "IP67", "IP68", "NEMA 4X").\n'
        'If multiple IP ratings appear, return the highest one.\n'
        "GOOD: 'IP67' | BAD: 'dust-proof', 'weatherproof'"
    ),

    "classificationZone": (
        'Extract the hazardous area / ATEX / IECEx classification as printed '
        '(e.g. "II 2G Ex d IIC T6", "ATEX Zone 1", "IECEx Zone 0").\n'
        'Return null if the device is not certified for hazardous areas.\n'
        "GOOD: 'II 2G Ex d IIC T6' | BAD: 'approved', 'certified'"
    ),

    "reperage": (
        'Extract the tag number / instrument tag used on the P&ID or plant '
        '(e.g. "FT-101", "PT-202A", "LT_003").\n'
        'This is a site-specific identifier, NOT a manufacturer model number.\n'
        "GOOD: 'FT-101' | BAD: '6ES7 315', 'SITRANS P320'"
    ),

    "dateCalibration": (
        'Extract the CALIBRATION INTERVAL (how often), NOT a specific calendar date.\n'
        'Return format: "<N> mois" (e.g. "3 mois", "12 mois", "6 mois").\n'
        'Convert: quarterly→"3 mois", monthly→"1 mois", annually→"12 mois", every 2 years→"24 mois".\n'
        'If only a table of past calibration dates appears with no stated interval, compute the gap between dates.\n'
        'Do NOT return certificate numbers (e.g. IEC 60770-2) or specific dates.\n'
        "GOOD: '12 mois' | BAD: '2024-03-15', 'IEC 60770-2', 'annual calibration required'"
    ),

    "sortiesAlarme": (
        'Extract alarm/relay outputs as a list ONLY if alarm names, thresholds, AND units are ALL explicitly stated.\n'
        'Each object: {"nomAlarme": "<name>", "typeAlarme": "<Haut|Bas|Défaut>", '
        '"seuilAlarme": <number>, "uniteAlarme": "<unit>", "relaisAssocie": "<relay>"}\n'
        'ANTI-HALLUCINATION — nomAlarme and seuilAlarme must be explicit in the document, never guessed.\n'
        'DO NOT extract fault signal currents (e.g., "Courant de signalisation", 1.0mA, 3.6 mA, 22 mA) as alarm outputs. Alarms must be physical relay or switch outputs, not 4-20mA loop fault states.\n'
        'If the document only mentions detector type (e.g. "NAMUR") with no thresholds → return null.\n'
        'GOOD: [{"nomAlarme": "High Flow", "typeAlarme": "Haut", "seuilAlarme": 100, "uniteAlarme": "m3/h", "relaisAssocie": "R1"}]\n'
        'BAD: [{"nomAlarme": "Courant de signalisation", "seuilAlarme": 3.0, "uniteAlarme": "mA"}] ← Incorrect fault current'
    ),
}


def _build_extraction_prompt(
    field_name: str,
    chunks: list[dict],
    doc_ctx: DocumentContext | None = None,
) -> str:
    """Build a focused, example-driven prompt for a single field."""
    field_info = FIELD_DESCRIPTIONS.get(field_name, {})
    description = field_info.get("description", field_name)
    allowed = field_info.get("allowed_values")

    context = "\n---\n".join(c["text"] for c in chunks)

    allowed_str = ""
    if allowed:
        if field_name == "categorie":
            allowed_str = (
                f"\nAllowed values (must choose EXACTLY one): {', '.join(allowed)}. "
                "If none fits, return null."
            )
        else:
            allowed_str = (
                f"\nPreferred values (use one of these if it matches the document): {', '.join(allowed)}. "
                "If the document has a value not in this list, return the exact document value."
            )

    doc_ctx_str = ""
    if doc_ctx:
        doc_ctx_str = (
            f"\nDocument type: {doc_ctx.doc_type} "
            f"(confidence: {doc_ctx.confidence:.2f}) — {doc_ctx.rationale}\n"
        )

    field_rule = FIELD_PROMPT_RULES.get(field_name, "")
    field_rule_block = f"\nField rules:\n{field_rule}\n" if field_rule else ""

    # Response format hint
    if field_name == "plageMesure":
        fmt = '{"value": {"min": <number|null>, "max": <number|null>, "unite": "<unit|null>"}, "confidence": 0.0-1.0, "quote": "<verbatim text>"}'
    elif field_name == "sortiesAlarme":
        fmt = '{"value": [{"nomAlarme": "...", "typeAlarme": "...", "seuilAlarme": <number>, "uniteAlarme": "...", "relaisAssocie": "..."}], "confidence": 0.0-1.0, "quote": "<verbatim text>"}'
    else:
        fmt = '{"value": "<extracted value or null>", "confidence": 0.0-1.0, "quote": "<verbatim text or null>"}'

    return f"""You are an industrial equipment datasheet analyzer.
{doc_ctx_str}
Extract the field "{field_name}" from the document excerpts below.
Description: {description}{allowed_str}
{field_rule_block}
Document excerpts:
\"\"\"
{context}
\"\"\"

RULES:
- Only extract if the value is EXPLICITLY stated or unambiguously implied.
- "quote" must be a verbatim substring copied from the excerpts above — no paraphrasing, no ellipses.
- If the value is not found, return {{"value": null, "confidence": 0.0, "quote": null}}.
- Respond with ONLY a JSON object. No explanation, no markdown.

Response format: {fmt}

JSON:"""


def _build_strict_json_prompt(field_name: str, chunks: list[dict], doc_ctx: DocumentContext | None = None) -> str:
    base = _build_extraction_prompt(field_name, chunks, doc_ctx=doc_ctx)
    return base + "\n\nSTRICT: Output MUST start with '{{' and end with '}}'. No other characters."


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_field(field_name: str, chunks: list[dict], doc_ctx: DocumentContext | None = None) -> dict:
    if not _field_applicable(field_name, doc_ctx) or not chunks:
        return {"value": None, "confidence": 0.0, "quote": None}

    prompt = _build_extraction_prompt(field_name, chunks, doc_ctx=doc_ctx)
    messages = [
        {
            "role": "system",
            "content": "You are a precise data extractor. Always respond with valid JSON only. No markdown, no explanation.",
        },
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            try:
                response = ollama_client.chat(
                    model=TEXT_MODEL,
                    messages=messages,
                    options={"temperature": 0.1},
                    format="json",
                    keep_alive="10m",
                )
            except TypeError:
                response = ollama_client.chat(
                    model=TEXT_MODEL,
                    messages=messages,
                    options={"temperature": 0.1},
                    keep_alive="10m",
                )

            raw = response["message"]["content"].strip()
            result = _parse_llm_json(raw)

            if attempt < 3 and result.get("value") is None and not (raw.lstrip().startswith("{") and raw.rstrip().endswith("}")):
                prompt = _build_strict_json_prompt(field_name, chunks, doc_ctx=doc_ctx)
                raise ValueError("LLM did not return JSON; retrying with strict prompt")

            result.setdefault("_raw", raw[:1200])
            result.setdefault("quote", None)
            result.setdefault("confidence", 0.0)
            result.setdefault("value", None)
            return result

        except Exception as e:
            last_error = e
            print(f"[LLM] Error '{field_name}' (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(2 * attempt)

    print(f"[LLM] Giving up '{field_name}': {last_error}")
    return {"value": None, "confidence": 0.0, "quote": None, "_error": str(last_error) if last_error else None}


# ---------------------------------------------------------------------------
# Validation helpers  (lean — trust the LLM + quote anchor)
# ---------------------------------------------------------------------------
def _field_applicable(field_name: str, doc_ctx: DocumentContext | None) -> bool:
    return True  # Extend with doc_type gates if needed


def _quote_in_context(quote: str | None, chunks: list[dict]) -> bool:
    if not quote:
        return False

    # Some datasheets legitimately use ellipsis to denote ranges (e.g. "0,2…1,0 bar").
    # Only reject ellipsis when it looks like truncation rather than a numeric range.
    qraw = quote
    if "..." in qraw or "…" in qraw:
        if not re.search(r"\d\s*(?:\.\.\.|…)\s*\d", qraw):
            return False
    context = "\n---\n".join(c.get("text", "") for c in chunks)
    q = quote.strip()
    if q in context:
        return True

    # Normalised whitespace + punctuation fallback
    def _norm(s: str) -> str:
        s = s.replace("…", "...").replace("–", "-").replace("−", "-")
        s = s.replace("\u00a0", " ")
        return re.sub(r"\s+", " ", s).strip().lower()

    return bool(q) and _norm(q) in _norm(context)


def _quote_optional(field_name: str) -> bool:
    return field_name in {"categorie", "typeMesure", "technologie"}


def _value_supported_by_quote(value, quote: str | None) -> bool:
    """Lightweight lexical check: extracted value should appear in the quote."""
    if value is None:
        return True
    if not quote:
        return False
    if isinstance(value, str):
        v = value.strip().lower()
        q = quote.lower()
        # Try direct substring, then compact (strip non-alnum)
        if v in q:
            return True
        vc = re.sub(r"[^a-z0-9]", "", v)
        qc = re.sub(r"[^a-z0-9]", "", q)
        return bool(vc) and vc in qc
    # For dicts/lists just trust quote presence
    return True


def _normalize_value(field_name: str, value):
    """Minimal normalization — canonical casing and alias resolution only."""
    if value is None:
        return None

    # Unwrap single-key dicts the model sometimes returns
    if isinstance(value, dict) and field_name not in ("plageMesure", "sortiesAlarme"):
        if field_name in value:
            value = value[field_name]
        elif len(value) == 1:
            value = next(iter(value.values()))

    if not isinstance(value, str):
        return value

    v = value.strip()
    vl = v.lower()

    ALIAS: dict[str, dict[str, str]] = {
        "typeMesure": {
            "pression": "Pression", "pressure": "Pression", "druck": "Pression",
            "debit": "Debit", "débit": "Debit", "flow": "Debit", "durchfluss": "Debit",
            "niveau": "Niveau", "level": "Niveau", "füllstand": "Niveau",
            "temperature": "Temperature", "température": "Temperature", "temperatur": "Temperature",
            "autre": "Autre", "other": "Autre",
        },
        "communication": {
            "profibus dp": "PROFIBUS DP", "profibus": "PROFIBUS DP",
            "profibus pa": "Profibus PA",
            "foundation fieldbus": "Foundation Fieldbus", "fieldbus": "Foundation Fieldbus",
            "profinet": "Profinet", "ethernet": "Ethernet",
            "rs-232": "RS-232", "rs232": "RS-232",
            "rs-485": "RS-485", "rs485": "RS-485",
            "modbus tcp": "Modbus TCP", "modbus tcp/ip": "Modbus TCP/IP",
            "modbus rtu": "Modbus RTU", "hart": "HART",
        },
        "technologie": {
            "électromagnétique": "Electromagnetique", "electromagnetique": "Electromagnetique", "electromagnetic": "Electromagnetique",
            "magnetique": "Magnetique", "magnétique": "Magnetique",
            "hydraulique": "Hydraulique", "pneumatique": "Pneumatique",
            "numérique": "Numerique", "numerique": "Numerique", "digital": "Numerique",
            "piezo-resistif": "Piezo-resistif", "piézo-résistif": "Piezo-resistif",
            "electronique": "Electronique", "électronique": "Electronique",
            "tft tactile": "TFT Tactile", "autre": "Autre", "other": "Autre",
        },
        "nbFils": {
            "2 fils": "2 fils", "2-wire": "2 fils", "two-wire": "2 fils", "2wire": "2 fils", "2": "2 fils",
            "4 fils": "4 fils", "4-wire": "4 fils", "four-wire": "4 fils", "4wire": "4 fils", "4": "4 fils",
        },
    }

    if field_name in ALIAS:
        mapped = ALIAS[field_name].get(vl)
        if mapped:
            return mapped

    # typeSignal normalisation
    if field_name == "typeSignal":
        v = v.replace("…", "-").replace("–", "-").replace("...", "-")
        compact = re.sub(r"\s+", "", v).lower()
        MAP = {"4-20ma": "4-20mA", "4-20": "4-20mA", "0-20ma": "0-20mA",
               "0-10v": "0-10V", "0-5v": "0-5V"}
        return MAP.get(compact, v)

    # alimentation normalisation
    if field_name == "alimentation":
        v = v.replace("Vdc", "V DC").replace("VDC", "V DC").replace("Vac", "V AC").replace("VAC", "V AC")
        compact = re.sub(r"\s+", "", v).lower()
        if compact == "24vdc": return "24V DC"
        if compact == "220vac": return "220V AC"
        return v

    # dateCalibration: ensure "<N> mois" format
    if field_name == "dateCalibration":
        m = re.search(r"\b(\d{1,3})\s*(mois|month|months|an|ans|year|years|jour|day|days)\b", vl)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if unit in {"an", "ans", "year", "years"}: n *= 12
            if unit in {"jour", "day", "days"}: return f"{n} jours"
            return f"{n} mois"
        ADVERBS = {
            "trimestriel": "3 mois", "quarterly": "3 mois",
            "mensuel": "1 mois", "monthly": "1 mois",
            "annuel": "12 mois", "annually": "12 mois", "yearly": "12 mois",
        }
        for kw, canonical in ADVERBS.items():
            if kw in vl:
                return canonical
        return v

    return v


def _is_expected_type(field_name: str, value) -> bool:
    if value is None: return True
    if field_name == "plageMesure": return isinstance(value, dict)
    if field_name == "sortiesAlarme": return isinstance(value, list)
    return isinstance(value, str)


def _is_numericish_string(s: str) -> bool:
    v = (s or "").strip()
    if not v:
        return False
    # Allow common separators but require at least one digit and no letters.
    has_digit = any(ch.isdigit() for ch in v)
    has_alpha = any(ch.isalpha() for ch in v)
    return has_digit and not has_alpha


def _field_sanity_check(field_name: str, value, quote: str | None) -> bool:
    """Extra guardrails for common failure modes (page numbers, ratios, etc.)."""
    if value is None:
        return True

    if field_name in {"equipmentName", "marque", "modele", "technologie"}:
        if isinstance(value, str) and _is_numericish_string(value):
            return False

    if field_name == "plageMesure" and isinstance(value, dict):
        q = (quote or "").lower()
        unite = str(value.get("unite") or "").strip().lower()
        vmin = value.get("min")
        vmax = value.get("max")

        # Reject the classic ratio confusion: "10 : 1", "100:1", etc.
        if ":" in q:
            # Only accept if the quote also contains the returned unit and both bounds.
            if unite and unite not in q:
                return False
            if vmin is not None and str(vmin) not in (quote or ""):
                return False
            if vmax is not None and str(vmax) not in (quote or ""):
                return False

    return True


# Fields where the allowed_values list is a soft hint, not a hard constraint.
# The LLM may return valid values not in the list (e.g. "Débitmètre" for categorie).
_OPEN_FIELDS = {"technologie", "typeMesure", "communication"}


def _is_value_allowed(field_name: str, value) -> bool:
    if not ACCURACY_FIRST: return True
    if field_name in _OPEN_FIELDS: return True   # open-ended: don't block valid extractions
    allowed = FIELD_DESCRIPTIONS.get(field_name, {}).get("allowed_values")
    if value is None or not allowed: return True
    return isinstance(value, str) and value in allowed


def _apply_cross_field_guards(extracted: dict) -> dict:
    out = dict(extracted)

    # equipmentName must not look like an order number
    eq = out.get("equipmentName")
    ref = out.get("reference")
    if isinstance(eq, str):
        is_ref_pattern = bool(re.match(r"^(6AV|6ES7|7MF)\b", eq, re.I)) or (
            bool(re.fullmatch(r"[A-Z0-9\-_/\.]{8,}", eq.upper()))
            and any(c.isdigit() for c in eq)
            and any(c.isalpha() for c in eq)
        )
        if is_ref_pattern or (isinstance(ref, str) and eq.strip() == ref.strip()):
            out.pop("equipmentName", None)

        # equipmentName must not be just a page/section number or a pure numeric token
        if _is_numericish_string(eq):
            out.pop("equipmentName", None)

    # plageMesure unit must match typeMesure (pressure → pressure unit)
    if out.get("typeMesure") == "Pression" and isinstance(out.get("plageMesure"), dict):
        unite = (out["plageMesure"].get("unite") or "").strip().lower()
        if unite not in {"bar", "pa", "psi", "kpa", "mpa", "mbar", "mmhg", "mmh2o", "inhg"}:
            out.pop("plageMesure", None)

    # plageMesure unit must not be a ratio
    if isinstance(out.get("plageMesure"), dict):
        unite = (out["plageMesure"].get("unite") or "").strip()
        if ":" in unite or unite.replace(".", "").isdigit():
            out.pop("plageMesure", None)

    return out


# ---------------------------------------------------------------------------
# Main extraction entry points
# ---------------------------------------------------------------------------

def _run_llm_fields(
    llm_fields: dict[str, list[dict]],
    field_chunks: dict[str, list[dict]],
    min_confidence: float,
    doc_ctx: DocumentContext | None,
) -> tuple[dict, dict]:
    """Extract fields via LLM in parallel; return (extracted, meta)."""
    from concurrent.futures import ThreadPoolExecutor
    from config import LLM_MAX_WORKERS

    extracted: dict = {}
    meta: dict = {}

    def process(field_name, chunks):
        t0 = time.time()
        print(f"  [LLM] START: {field_name} ({len(chunks)} chunks)…")
        result = extract_field(field_name, chunks, doc_ctx=doc_ctx)
        print(f"  [LLM] FINISH: {field_name} in {time.time()-t0:.1f}s")
        return field_name, result

    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
        for field_name, result in (f.result() for f in [executor.submit(process, n, c) for n, c in llm_fields.items()]):
            value = _normalize_value(field_name, result.get("value"))
            confidence = float(result.get("confidence") or 0.0)
            quote = result.get("quote")

            quote_supported = _value_supported_by_quote(value, quote)
            quote_in_context = _quote_in_context(quote, field_chunks.get(field_name, []))
            if _quote_optional(field_name) and not quote:
                quote_supported = True
                quote_in_context = True

            checks = {
                "non_null": value is not None,
                "min_confidence": confidence >= min_confidence,
                "expected_type": _is_expected_type(field_name, value),
                "allowed_value": _is_value_allowed(field_name, value),
                "quote_supported": quote_supported,
                "quote_in_context": quote_in_context,
                "sane_value": _field_sanity_check(field_name, value, quote),
            }
            ok = all(checks.values())
            rejection = next((k for k, v in checks.items() if not v), None) if not ok else None

            meta[field_name] = {
                "accepted": ok, "confidence": confidence, "value": value if ok else None,
                "quote": quote, "source": "llm", "checks": checks,
                "rejection_reason": rejection, "raw": result.get("_raw"), "error": result.get("_error"),
            }
            if ok:
                extracted[field_name] = value
                print(f"  [OK]   {field_name} = {value!r}")
            else:
                print(f"  [MISS] {field_name} (reason: {rejection}, conf: {confidence:.2f})")

    return extracted, meta


def extract_all_fields_with_meta(
    field_chunks: dict[str, list[dict]],
    min_confidence: float = 0.3,
    regex_results: dict[str, dict] | None = None,
    doc_ctx: DocumentContext | None = None,
    collection_name: str | None = None,
) -> tuple[EquipmentSchema, dict[str, dict]]:
    if regex_results is None:
        regex_results = {}

    extracted: dict = {}
    meta: dict = {}
    llm_fields: dict = {}

    # Phase 1 — regex results
    for field_name, chunks in field_chunks.items():
        if doc_ctx is not None and not _field_applicable(field_name, doc_ctx):
            meta[field_name] = {"accepted": False, "confidence": 0.0, "value": None,
                                "quote": None, "source": "doc_type_gate",
                                "rejection_reason": "inapplicable_for_doc_type"}
            continue

        if field_name in regex_results:
            rx = regex_results[field_name]
            value = _normalize_value(field_name, rx.get("value"))
            confidence = float(rx.get("confidence") or 1.0)
            quote = rx.get("quote")

            quote_supported = _value_supported_by_quote(value, quote)
            quote_in_context = _quote_in_context(quote, chunks or [])
            if _quote_optional(field_name) and not quote:
                quote_supported = True
                quote_in_context = True

            checks = {
                "non_null": value is not None,
                "min_confidence": confidence >= min_confidence,
                "expected_type": _is_expected_type(field_name, value),
                "allowed_value": _is_value_allowed(field_name, value),
                "quote_supported": quote_supported,
                "quote_in_context": quote_in_context,
            }
            ok = all(checks.values())
            meta[field_name] = {
                "accepted": ok, "confidence": confidence, "value": value if ok else None,
                "quote": quote, "source": "regex", "checks": checks,
                "rejection_reason": next((k for k, v in checks.items() if not v), None) if not ok else None,
            }
            if ok:
                extracted[field_name] = value
                print(f"  [REGEX] {field_name} = {value!r}")
            else:
                llm_fields[field_name] = chunks  # fallback to LLM
        else:
            llm_fields[field_name] = chunks

    # Phase 2 — LLM
    if llm_fields:
        llm_extracted, llm_meta = _run_llm_fields(llm_fields, field_chunks, min_confidence, doc_ctx)
        extracted.update(llm_extracted)
        meta.update(llm_meta)

    # Phase 3 — cross-field guards
    extracted = _apply_cross_field_guards(extracted)

    # Phase 4 — re-retrieval verification (DISABLED: causes false negatives dropping valid fields)
    # extracted = _post_extract_reretrieval_verification(extracted, field_chunks=field_chunks, collection_name=collection_name)

    # Build schema
    schema_data = {k: v for k, v in extracted.items()}
    return EquipmentSchema(**schema_data), meta


# ---------------------------------------------------------------------------
# Re-retrieval verification
# ---------------------------------------------------------------------------

def _normalize_for_query(field_name: str, value) -> str:
    if value is None: return ""
    if isinstance(value, str): return value
    if field_name == "plageMesure" and isinstance(value, dict):
        return f"{value.get('min')} {value.get('unite')} {value.get('max')}"
    if field_name == "sortiesAlarme" and isinstance(value, list):
        return " ".join(str(a.get("nomAlarme", "")) for a in value if isinstance(a, dict))
    return str(value)


def _top_result_supports_value(field_name: str, value, top_chunk: dict | None) -> bool:
    if value is None or not top_chunk: return bool(value is None)
    text = top_chunk.get("text", "")
    return _value_supported_by_quote(value, text)


def _post_extract_reretrieval_verification(
    extracted: dict,
    field_chunks: dict | None = None,
    collection_name: str | None = None,
) -> dict:
    try:
        from services.vector_store import query_chunks
    except Exception:
        return extracted

    doc_id = None
    if field_chunks:
        for chunks in field_chunks.values():
            for c in (chunks or []):
                doc_id = (c.get("metadata") or {}).get("doc_id")
                if doc_id: break
            if doc_id: break

    where = {"doc_id": doc_id} if doc_id else None
    verified: dict = {}

    for field_name, value in extracted.items():
        qv = _normalize_for_query(field_name, value).strip()
        if not qv: continue
        results = query_chunks(f"What is {qv}?", n_results=1, where=where, collection_name=collection_name)
        top = results[0] if results else None
        if _top_result_supports_value(field_name, value, top):
            verified[field_name] = value
        else:
            print(f"  [VERIFY FAIL] {field_name}='{value}' not supported by re-retrieval")

    return verified