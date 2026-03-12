"""
LLM extraction service.
Uses Ollama (Qwen2.5) to extract structured data from document chunks.
"""
import json
import ollama as ollama_client
import re
import time
import unicodedata
from datetime import datetime

from config import ACCURACY_FIRST, TEXT_MODEL, ENABLE_RERETRIEVAL_VERIFICATION
from models.schema import FIELD_DESCRIPTIONS, EquipmentSchema
from services.document_classifier import DocumentContext


# ---------------------------------------------------------------------------
# Per-field prompt rules + few-shot examples
# ---------------------------------------------------------------------------

FIELD_PROMPT_RULES: dict[str, str] = {
    "equipmentName": (
        'Extract the PRIMARY commercial product name — the main device this document is about.\n'
        'It appears on the COVER PAGE or DOCUMENT TITLE, usually as the largest text or document heading.\n'
        'Examples: "H250 M40", "SITRANS P320", "OPTIFLUX 2000", "SIMATIC S7-1200".\n'
        'RULES:\n'
        '  - If the title says "H250 M40 – Débitmètres à section variable", return "H250 M40".\n'
        '  - Sub-components or modules mentioned inside the document (e.g. "ESK4", "FSK4") are NOT the equipment name.\n'
        '  - NEVER return part/order numbers (6AV…, 6ES7…, 7MF…, alphanumeric codes with hyphens).\n'
        '  - If the cover shows both a product family and a specific model, return the specific model.\n'
        "GOOD: 'H250 M40', 'SITRANS P320' | BAD: 'ESK4' (module name), '6ES7 153-1AA00-0XB0' (order number)"
    ),

    "reference": (
        'Extract the MAIN PRODUCT manufacturer part number / order number.\n'
        'It must be the order code for the primary instrument described in this document.\n'
        'Siemens examples: "7MF4433-1DA02", "7ML1830-1DP", "6NH3112-3BA00-0"\n'
        'RULES:\n'
        '  - Must appear in the ordering section, title page, or ordering data table\n'
        '  - Do NOT extract accessory codes, accessory part numbers, or power supply references\n'
        '  - "7ML534" appearing in a power/accessory section is NOT the main reference\n'
        '  - If no clear order number is found, return null — do not guess\n'
        "GOOD: '7MF4433-1DA02' (from ordering section) | BAD: '7ML534' (accessory reference)"
    ),

    "categorie": (
        'Classify the equipment into ONE of: Transmetteur, Débitmètre, Capteurs, Actionneur, Automate, IHM, Autre.\n'
        'Rules:\n'
        '  - Radar level transmitters, level transmitters, distance/level sensors with 4-20mA output → Transmetteur\n'
        '  - Flow meters (débitmètres, variable area, Coriolis, vortex, electromagnetic flow) → Débitmètre\n'
        '  - Pressure/level/temp transmitters with 4-20mA output → Transmetteur\n'
        '  - PLCs/CPUs/controllers → Automate\n'
        '  - Touchscreens/HMI panels → IHM\n'
        '  - Passive sensors WITHOUT signal conditioning → Capteurs\n'
        '  - Valves/positioners/actuators → Actionneur\n'
        'IMPORTANT: If the document title or description contains "transmitter" or "transmetteur", classify as Transmetteur.\n'
        'For the quote, use the EXACT verbatim fragment from the source language (e.g. "transmitter", "level sensor", "débitmètre").\n'
        'The "value" must be one of the French labels above, but the "quote" must be from the document.\n'
        "GOOD: 'Transmetteur' with quote 'radar level transmitter' | BAD: 'flowmeter', 'capteur de débit'"
    ),

    "typeMesure": (
        'Extract the PRIMARY physical quantity this instrument DIRECTLY measures.\n'
        'Allowed values: Debit, Niveau, Pression, Temperature, Autre\n'
        'CRITICAL RULES:\n'
        '  - A RADAR sensor measures DISTANCE or LEVEL → return "Niveau"\n'
        '  - An ultrasonic level sensor → "Niveau"\n'
        '  - Even if level can be CONVERTED to flow (via weir/flume formulas), the direct measurement is STILL "Niveau"\n'
        '  - Only return "Debit" if the instrument DIRECTLY outputs a flow value (e.g. electromagnetic flowmeter)\n'
        '  - Pressure transmitter → "Pression"\n'
        '  - Thermocouple / RTD → "Temperature"\n'
        'GOOD: "Niveau" (for radar, ultrasonic, float, capacitive level sensors)\n'
        'BAD: "Debit" for a radar that can be configured for open-channel flow measurement'
    ),

    "technologie": (
        'Extract the measurement/actuation technology principle EXACTLY as described in the document.\n'
        'Do not translate or reformat — return a short French or bilingual label.\n'
        'Common examples: Electromagnetique, Piezo-resistif, Ultrasonique, Coriolis, Vortex, '
        'Radar, Capacitif, Pneumatique, Hydraulique, Section variable (flotteur), TFT Tactile, Autre.\n'
        'For variable-area / rotameter / float-tube devices → "Section variable (flotteur)"\n'
        "GOOD: 'Section variable (flotteur)', 'Electromagnetique', 'Coriolis' | BAD: 'float principle', 'variable area flowmeter'"
    ),

    "typeSignal": (
        'Extract the output signal type.\n'
        'Look for sections titled: "Output signal", "Electrical output", "Signal output", "Communication".\n'
        'Allowed values: 4-20mA, 0-20mA, 0-10V, 0-5V, 24V DC (DI/DO), Autre\n'
        'Rules:\n'
        '  - "4 ... 20 mA", "4…20mA", "4 to 20 mA", "4-20 mA" → all mean "4-20mA"\n'
        '  - If document shows "4 … 20 mA/HART" → return "4-20mA" (HART is the protocol, not the signal type)\n'
        '  - For the quote, use the shortest verbatim fragment containing the signal range (e.g. "4 … 20 mA")\n'
        'GOOD: "4-20mA" | BAD: "4-20mA with HART", "loop current"'
    ),

    "plageMesure": (
        'Extract the FLOW/PRESSURE/LEVEL measuring range as {min, max, unite}.\n'
        'CRITICAL RULES:\n'
        '  1. For a FLOW METER (débitmètre), the range MUST use flow units: L/h, m³/h, kg/h, t/h, gpm.\n'
        '     NEVER return a temperature range (°C, °F, K) for a flowmeter — those are fluid temp LIMITS.\n'
        '  2. For ratio/turndown (e.g. "10:1", "100:1") → return null, never a numeric range.\n'
        '  3. Ignore: time delays (ms/µs), supply voltages, temperature operating limits for flow devices.\n'
        '  4. If no numeric min/max range with CORRECT units is found → return null.\n'
        'GOOD: {"min": 40, "max": 1200, "unite": "L/h"} ← flowmeter range\n'
        'GOOD: {"min": 0, "max": 10, "unite": "bar"} ← pressure transmitter range\n'
        'BAD: {"min": -196, "max": 300, "unite": "°C"} ← fluid temperature limit ≠ measuring range\n'
        'BAD: {"min": 10, "max": 100, "unite": "1"} ← ratio misread as range'
    ),

    "modele": (
        'Extract the model name or designation from the document title or ordering section.\n'
        'This is usually the product name without manufacturer prefix.\n'
        'Examples: "H250 M40", "SITRANS P320", "OPTIFLUX 2000".\n'
        'RULES:\n'
        '  - Look for "Type", "Model", "Type designation" labels.\n'
        '  - The title often contains both manufacturer and model.\n'
        '  - Do NOT return order numbers (6AV…, 7MF…) or phone numbers.\n'
        '  - Do NOT return website URLs, homepage text, or marketing slogans.\n'
        '  - If the title says "Temperature Sensor NTC M12", the model is "Temperature Sensor NTC M12".\n'
        "GOOD: 'H250 M40', 'SITRANS P320', 'Temperature Sensor NTC M12' | BAD: 'Bosch Data Logging at our homepage' (URL), '6ES7 315' (order number)"
    ),

    "alimentation": (
        'Extract the supply voltage/power specification.\n'
        'Look for sections titled: "Voltage supply", "Power supply", "Operating voltage", "Alimentation".\n'
        'CRITICAL: Only extract VOLTAGE values (V, VDC, VAC, V DC, V AC).\n'
        'Valid formats: "24V DC", "12…35 V DC", "9…35 V DC", "85…264V AC"\n'
        'NEVER extract:\n'
        '  - "Loop powered" or "2 fils" (these belong to nbFils, not alimentation)\n'
        '  - Product/order numbers (e.g. "7ML534", "6ES7…", anything with letters+digits like a part number)\n'
        '  - Phone numbers, addresses, model names\n'
        '  - Power consumption in watts (W) or current in mA alone\n'
        'If the document shows a range like "at 4 mA: 12…35 V DC / at 20 mA: 9…35 V DC", return "9…35 V DC" (the minimum range).\n'
        "GOOD: '12…35 V DC', '9-35 V DC' | BAD: 'Loop powered', '7ML534' (product code)"
    ),

    "nbFils": (
        'Extract or INFER the number of wires for electrical connection (wiring topology).\n'
        'Allowed values: "2 fils", "4 fils", "Autre (3,5,...)"\n'
        'Direct evidence: look for "fils", "wire", "2-wire", "4-wire", "2 fils", "4 fils", "loop powered", "2-Leiter", "4-Leiter".\n'
        'INFERENCE RULES:\n'
        '  - If the document says "Loop powered", "auto-alimenté par la boucle", or "alimenté par la boucle" → return "2 fils".\n'
        '  - 4-20mA loop powered → "2 fils" (power and signal on same 2 wires).\n'
        '  - 4-20mA with separate power supply (e.g. 230V AC or separate 24V supply terminals) → "4 fils".\n'
        '  - If document states operating voltage 9-35V DC at 4-20mA and no separate power terminal mentioned → "2 fils".\n'
        'For the quote: use the shortest phrase that supports this (e.g. "loop powered", "2-wire", or the voltage/current spec).\n'
        "GOOD: '2 fils' with quote 'loop powered' | BAD: '2 fils' with quote '4...20 mA' if 'loop powered' was available."
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
        'STRICT ANTI-HALLUCINATION RULES — return null unless ALL of these are met:\n'
        '  1. nomAlarme: the exact alarm label must appear word-for-word in the document.\n'
        '  2. seuilAlarme: a specific numeric threshold value must be stated next to the alarm name.\n'
        '  3. uniteAlarme: the unit of that threshold must be stated.\n'
        'If ANY of these three are absent → return null. Do NOT infer or guess any of them.\n'
        'NAMUR detectors, MIN/MAX contacts, or relay outputs mentioned WITHOUT a setpoint value → null.\n'
        'GOOD: [{"nomAlarme": "High Flow", "typeAlarme": "Haut", "seuilAlarme": 100, "uniteAlarme": "m3/h", "relaisAssocie": "R1"}]\n'
        'BAD: [{"nomAlarme": "MIN 1", "seuilAlarme": 1.0, "uniteAlarme": "mA"}] ← "MIN 1" not in document; 1.0 mA invented'
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
- "quote" must be a VERBATIM substring copied character-for-character from the excerpts above.
  • Do NOT translate — if the document is in French, the quote must be in French.
  • Do NOT paraphrase, summarize, or rephrase any part of the quote.
  • Do NOT use ellipses (...) or (…) anywhere in the quote.
  • The quote must appear exactly as-is in the excerpts (same words, same order, same language).
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
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            try:
                response = ollama_client.chat(
                    model=TEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1},
                    format="json",
                    keep_alive="10m",
                )
            except TypeError:
                response = ollama_client.chat(
                    model=TEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
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
    """Check if a field should be automatically extracted."""
    # User request: exclude reference and plage de mesure from automatic determination
    if field_name in ("reference", "plageMesure"):
        return False
    return True


from difflib import SequenceMatcher

def _normalize_encoding(s: str) -> str:
    """Normalize common encoding/OCR differences (ellipsis, quotes, spaces)."""
    if not s:
        return ""
    # Ellipsis normalization
    s = s.replace("…", "...").replace("..", ".")
    # Quote normalization
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Hyphen/Dash normalization
    s = s.replace("–", "-").replace("—", "-")
    # Whitespace normalization (handle non-breaking spaces, tabs, etc.)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _quote_in_context(quote: str | None, chunks: list[dict], threshold: float = 0.85) -> bool:
    """Check if the quote exists in chunks using fuzzy matching to handle encoding/whitespace issues."""
    if not quote:
        return False
    
    q_norm = _normalize_encoding(quote).lower()
    if not q_norm:
        return False

    context = "\n---\n".join(c.get("text", "") for c in chunks)
    context_norm = _normalize_encoding(context).lower()
    
    # 1. Exact match (fast path)
    if q_norm in context_norm:
        return True
    
    # 2. Fuzzy sliding window match (handles minor encoding variations)
    # Optimization: only check window if q_norm is long enough
    if len(q_norm) < 10:
        return q_norm in context_norm

    # Check a bit wider than q_norm to catch nearby variations
    window_size = len(q_norm)
    for i in range(len(context_norm) - window_size + 1):
        window = context_norm[i:i + window_size]
        # Use SequenceMatcher for fuzzy ratio
        ratio = SequenceMatcher(None, q_norm, window).ratio()
        if ratio >= threshold:
            return True
        
    return False


def _compute_confidence(
    *,
    checks: dict,
    chunks: list[dict] | None,
    source: str,
    model_confidence: float | None = None,
) -> float:
    if not checks.get("non_null", True):
        return 0.0

    best_distance: float | None = None
    if chunks:
        for c in chunks:
            d = c.get("distance")
            if isinstance(d, (int, float)):
                best_distance = float(d) if best_distance is None else min(best_distance, float(d))

    if best_distance is None:
        retrieval_score = 0.5
    else:
        retrieval_score = 1.0 - max(0.0, min(1.0, best_distance))

    score = retrieval_score
    if not checks.get("quote_in_context", True):
        score *= 0.35
    if not checks.get("quote_supported", True):
        score *= 0.50
    if not checks.get("expected_type", True):
        score *= 0.25
    if not checks.get("allowed_value", True):
        score *= 0.65

    if source == "regex" and all(checks.values()):
        score = max(score, 0.90)

    if isinstance(model_confidence, (int, float)):
        mc = max(0.0, min(1.0, float(model_confidence)))
        score = 0.90 * score + 0.10 * mc

    return max(0.0, min(1.0, float(score)))


def _strip_accents(s: str) -> str:
    """Remove diacritics/accents for fuzzy matching (é→e, ü→u, etc.)."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def llm_verify_coherence(value, quote, field_name: str = "") -> bool:
    """Use a small, fast LLM call to verify if a quote supports an extracted value across languages."""
    if not value or not quote:
        return False

    prompt = f"""
    Verify if the following text excerpt (quote) supports the extracted information (value) for the field "{field_name}".
    The document might be in English, German, or French, but the value is usually normalized to French or a technical standard.
    
    Field: {field_name}
    Extracted Value: {value}
    Quote from document: "{quote}"
    
    Does the quote logically support the extracted value? (e.g., "radar" supports "Niveau", "4...20mA" supports "4-20mA", "loop powered" supports "2 fils").
    Answer ONLY with 'YES' or 'NO'.
    """
    
    try:
        response = ollama_client.chat(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
            keep_alive="5m"
        )
        answer = response["message"]["content"].strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"  [COHERENCE] Error: {e}")
        return True # Fallback to true to avoid aggressive rejection on API error

def _value_supported_by_quote(value, quote: str | None, field_name: str = "") -> bool:
    """Check if the value is supported by the quote (direct lexical or LLM-based coherence)."""
    if value is None:
        return True
    if not quote:
        return False

    # 1. Direct lexical support (Fast path)
    q = _normalize_encoding(quote).lower()
    if isinstance(value, str):
        v = _normalize_encoding(value).lower()
        if v in q:
            return True
            
        # Accent-normalized comparison
        v_stripped = _strip_accents(v)
        q_stripped = _strip_accents(q)
        if v_stripped in q_stripped:
            return True

    # 2. LLM-based semantic coherence check (Handles cross-language and inference)
    return llm_verify_coherence(value, quote, field_name)


FIELD_MIN_CONFIDENCE_OVERRIDE = {
    "typeSignal": 0.25,   # table dots vs ellipsis causes quote mismatches
    "alimentation": 0.25, # same issue with voltage tables
    "categorie": 0.25,    # short single-word classification
}

def _get_min_confidence(field_name: str, default_min: float) -> float:
    return FIELD_MIN_CONFIDENCE_OVERRIDE.get(field_name, default_min)

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

    # Reject alimentation if it looks like a part number (e.g. 7ML534, 6ES7...)
    if field_name == "alimentation" and v:
        if re.match(r'^[0-9A-Z]{3,}\d{3,}', v.replace(' ', '')):
            return None

    ALIAS: dict[str, dict[str, str]] = {
        "categorie": {
            "transmitter": "Transmetteur", "transmetteur": "Transmetteur",
            "débitmètre": "Débitmètre", "debitmetre": "Débitmètre", "flowmeter": "Débitmètre",
            "flow meter": "Débitmètre", "compteur": "Débitmètre", "rotameter": "Débitmètre",
            "capteur": "Capteurs", "capteurs": "Capteurs", "sensor": "Capteurs", "sensors": "Capteurs",
            "actionneur": "Actionneur", "actuator": "Actionneur",
            "automate": "Automate", "plc": "Automate", "cpu": "Automate",
            "ihm": "IHM", "hmi": "IHM",
            "autre": "Autre", "other": "Autre",
        },
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
            # Electromagnetic
            "électromagnétique": "Electromagnetique", "electromagnetique": "Electromagnetique",
            "electromagnetic": "Electromagnetique", "magnétique": "Magnetique", "magnetique": "Magnetique",
            # Variable area / float / rotameter
            "section variable (flotteur)": "Section variable (flotteur)",
            "section variable": "Section variable (flotteur)",
            "flotteur": "Section variable (flotteur)",
            "float": "Section variable (flotteur)",
            "variable area": "Section variable (flotteur)",
            "rotameter": "Section variable (flotteur)",
            "variable-area": "Section variable (flotteur)",
            "à flotteur": "Section variable (flotteur)",
            "a flotteur": "Section variable (flotteur)",
            "schwebekörper": "Section variable (flotteur)",
            # Ultrasonic
            "ultrasonique": "Ultrasonique", "ultrasonic": "Ultrasonique", "ultraschall": "Ultrasonique",
            # Coriolis / Vortex / Radar / Capacitive
            "coriolis": "Coriolis",
            "vortex": "Vortex",
            "radar": "Radar",
            "capacitif": "Capacitif", "capacitive": "Capacitif",
            # Others
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
        MAP = {
            "4-20ma": "4-20mA", 
            "4-20": "4-20mA", 
            "4-20ma/hart": "4-20mA",
            "420ma/hart": "4-20mA",
            "420mahrt": "4-20mA",
            "0-20ma": "0-20mA",
            "0-10v": "0-10V", 
            "0-5v": "0-5V"
        }
        return MAP.get(compact, v)

    # alimentation normalisation
    if field_name == "alimentation":
        # French: CC = courant continu (DC), CA = courant alternatif (AC)
        v = v.replace("V CC", "V DC").replace("V CA", "V AC")
        v = v.replace("Vdc", "V DC").replace("VDC", "V DC")
        v = v.replace("Vac", "V AC").replace("VAC", "V AC")
        # Normalise "14 V DC et 30 V DC" → "14-30V DC" (range written in French prose)
        range_m = re.match(r"(\d+(?:[.,]\d+)?)\s*V\s*DC\s+(?:et|à|to|and|-)\s*(\d+(?:[.,]\d+)?)\s*V\s*DC", v, re.IGNORECASE)
        if range_m:
            return f"{range_m.group(1)}-{range_m.group(2)}V DC"
        range_m = re.match(r"(\d+(?:[.,]\d+)?)\s*V\s*AC\s+(?:et|à|to|and|-)\s*(\d+(?:[.,]\d+)?)\s*V\s*AC", v, re.IGNORECASE)
        if range_m:
            return f"{range_m.group(1)}-{range_m.group(2)}V AC"
        compact = re.sub(r"\s+", "", v).lower()
        if compact == "24vdc": return "24V DC"
        if compact == "220vac": return "220V AC"
        
        # Issue 1 & 2: If alimentation contains wiring topology, it should be remapped, not kept here.
        # We return it as-is for now, and remap in _apply_cross_field_guards.
        # However, if it ONLY contains wiring topology and NO voltage, we might want to flag it.
        WIRING_PATTERNS = [r"loop\s*powered", r"2\s*fils?", r"2\s*wires?", r"4\s*fils?", r"4\s*wires?"]
        if any(re.search(p, vl) for p in WIRING_PATTERNS) and not any(c.isdigit() for c in v if c not in "24"):
            # If it matches wiring and doesn't look like a voltage (except maybe the '2' or '4' from 2-wire)
            pass 

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


# Fields where the allowed_values list is a soft hint, not a hard constraint.
# The LLM may return valid values not in the list (e.g. "Débitmètre" for categorie).
_OPEN_FIELDS = {"categorie", "technologie", "typeMesure", "communication"}

def _is_value_allowed(field_name: str, value) -> bool:
    if not ACCURACY_FIRST: return True
    if field_name in _OPEN_FIELDS: return True   # open-ended: don't block valid extractions
    allowed = FIELD_DESCRIPTIONS.get(field_name, {}).get("allowed_values")
    if value is None or not allowed: return True
    return isinstance(value, str) and value in allowed


def _parse_llm_json(raw: str) -> dict:
    if not raw:
        return {"value": None, "confidence": 0.0, "quote": None}

    # Strip markdown fences
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```"))

    def _first_json_obj(text: str) -> str | None:
        start = text.find("{")
        if start < 0: return None
        depth = in_str = esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc: esc = False; continue
                if ch == "\\": esc = True; continue
                if ch == '"': in_str = False
                continue
            if ch == '"': in_str = True; continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return text[start:i + 1]
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        extracted = _first_json_obj(raw)
        if extracted:
            try: parsed = json.loads(extracted)
            except json.JSONDecodeError:
                print(f"[LLM] JSON parse failed: {raw[:240]!r}")
                return {"value": None, "confidence": 0.0, "quote": None}
        else:
            print(f"[LLM] No JSON object found: {raw[:240]!r}")
            return {"value": None, "confidence": 0.0, "quote": None}

    if not isinstance(parsed, dict):
        return {"value": None, "confidence": 0.0, "quote": None}

    parsed.setdefault("value", None)
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("quote", None)
    return parsed


# ---------------------------------------------------------------------------
# Cross-field guard  (minimal — only clearly wrong cases)
# ---------------------------------------------------------------------------

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

    # plageMesure unit must match typeMesure (pressure → pressure unit)
    if out.get("typeMesure") == "Pression" and isinstance(out.get("plageMesure"), dict):
        unite = (out["plageMesure"].get("unite") or "").strip().lower()
        if unite not in {"bar", "pa", "psi", "kpa", "mpa", "mbar", "mmhg", "mmh2o", "inhg"}:
            out.pop("plageMesure", None)

    # plageMesure unit must NOT be temperature for flow meters
    _FLOW_INDICATORS = {"Debit", "Débitmètre"}
    type_mesure = out.get("typeMesure", "")
    categorie = out.get("categorie", "")
    if (type_mesure in _FLOW_INDICATORS or categorie in _FLOW_INDICATORS) and isinstance(out.get("plageMesure"), dict):
        unite = (out["plageMesure"].get("unite") or "").strip().lower()
        if unite in {"°c", "°f", "k", "celsius", "fahrenheit", "kelvin"}:
            print(f"  [GUARD] Rejected plageMesure with temp unit '{unite}' for flow device")
            out.pop("plageMesure", None)

    # plageMesure unit must NOT be temperature for pressure devices
    if type_mesure == "Pression" and isinstance(out.get("plageMesure"), dict):
        unite = (out["plageMesure"].get("unite") or "").strip().lower()
        if unite in {"°c", "°f", "k", "celsius", "fahrenheit", "kelvin"}:
            print(f"  [GUARD] Rejected plageMesure with temp unit '{unite}' for pressure device")
            out.pop("plageMesure", None)

    # Issue 1 & 2: alimentation to nbFils remapping
    alim = out.get("alimentation")
    if isinstance(alim, str):
        alim_l = alim.lower()
        wiring_found = None
        if any(x in alim_l for x in ["loop powered", "boucle", "2 wire", "2-wire", "2 fils", "2.fil"]):
            wiring_found = "2 fils"
        elif any(x in alim_l for x in ["4 wire", "4-wire", "4 fils", "4.fil"]):
            wiring_found = "4 fils"
        
        if wiring_found:
            # If nbFils is missing, populate it
            if not out.get("nbFils"):
                out["nbFils"] = wiring_found
                print(f"  [GUARD] Remapped '{alim}' from alimentation to nbFils='{wiring_found}'")
            
            # IMPROVED: Only clear alimentation if it contains NO voltage information.
            # Look for 2-3 digits followed by V (e.g., 24V, 230V, 12...35 V)
            has_voltage = bool(re.search(r'\d{1,3}(?:\s*[-.]{1,3}\s*\d{1,3})?\s*V', alim, re.I))
            
            # If it's just "Loop powered (2 fils)", and has no voltage, clear it.
            if not has_voltage:
                out.pop("alimentation", None)
                print(f"  [GUARD] Cleared alimentation '{alim}' (no voltage detected, only wiring topology)")

    # Issue 3: nomAlarme validation
    sorties = out.get("sortiesAlarme")
    if isinstance(sorties, list):
        valid_sorties = []
        for s in sorties:
            nom = s.get("nomAlarme")
            if not nom or not isinstance(nom, str):
                continue
            
            # Reject parameter descriptions or long sentences
            # Alarms are usually short: "High", "Low", "HH", "LL", "NAMUR", "Défaut", etc.
            nom_l = nom.lower()
            is_param_desc = any(x in nom_l for x in ["adjustable", "damping", "signal =", "input variable", "damping"])
            is_too_long = len(nom) > 30
            
            if is_param_desc or is_too_long:
                print(f"  [GUARD] Rejected alarm name: '{nom}' (looks like parameter description)")
                continue
            
            # Additional check: seuilAlarme 0.0 with uniteAlarme "s" is often a default/placeholder
            if s.get("seuilAlarme") == 0.0 and s.get("uniteAlarme") == "s":
                # Only keep if the name is very specific
                if "delay" not in nom_l:
                    print(f"  [GUARD] Rejected alarm: '{nom}' (placeholder values 0.0s)")
                    continue

            valid_sorties.append(s)
        
        if valid_sorties:
            out["sortiesAlarme"] = valid_sorties
        else:
            out.pop("sortiesAlarme", None)

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
            model_confidence = float(result.get("confidence") or 0.0)
            quote = result.get("quote")

            checks = {
                "non_null": value is not None,
                # Use model_confidence for the acceptance gate, not the blended score
                "min_confidence": model_confidence >= _get_min_confidence(field_name, min_confidence),
                "expected_type": _is_expected_type(field_name, value),
                "allowed_value": _is_value_allowed(field_name, value),
                "quote_supported": _value_supported_by_quote(value, quote, field_name),
                "quote_in_context": _quote_in_context(quote, field_chunks.get(field_name, []), field_name),
            }

            computed_confidence = _compute_confidence(
                checks=checks,
                chunks=field_chunks.get(field_name, []),
                source="llm",
                model_confidence=model_confidence,
            )
            ok = all(checks.values())
            rejection = next((k for k, v in checks.items() if not v), None) if not ok else None

            meta[field_name] = {
                "accepted": ok,
                "confidence": computed_confidence,
                "model_confidence": model_confidence,
                "value": value if ok else None,
                "quote": quote, "source": "llm", "checks": checks,
                "rejection_reason": rejection, "raw": result.get("_raw"), "error": result.get("_error"),
            }
            if ok:
                extracted[field_name] = value
                print(f"  [OK]   {field_name} = {value!r}")
            else:
                print(f"  [MISS] {field_name} (reason: {rejection}, conf: {computed_confidence:.2f})")

    return extracted, meta


def merge_extractions(results: list[dict], min_confidence: float = 0.3) -> tuple[dict, dict]:
    """Merge extraction results from multiple documents (e.g. Datasheet + OI)."""
    if not results:
        return {}, {}

    merged_data = {}
    merged_meta = {}

    # Priority order for fields if multiple documents provide values
    # Generally, Datasheet (FI) > Operating Instructions (OI)
    # But we can also use confidence as a tie-breaker
    
    all_fields = set()
    for res in results:
        all_fields.update(res.get("meta", {}).keys())

    for field in all_fields:
        best_val = None
        best_meta = None
        max_conf = -1.0

        for res in results:
            meta = res.get("meta", {}).get(field)
            if not meta or not meta.get("accepted"):
                continue
            
            conf = meta.get("confidence", 0.0)
            
            # If we have a very high confidence match, prefer it
            if conf > max_conf:
                max_conf = conf
                best_val = meta.get("value")
                best_meta = meta
            elif conf == max_conf and conf > 0:
                # Tie-breaker: prefer non-null if existing is null (shouldn't happen with accepted=True)
                if best_val is None and meta.get("value") is not None:
                    best_val = meta.get("value")
                    best_meta = meta

        if best_meta:
            merged_data[field] = best_val
            merged_meta[field] = best_meta

    return merged_data, merged_meta

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
            model_confidence = float(rx.get("confidence") or 1.0)
            quote = rx.get("quote")

            checks = {
                "non_null": value is not None,
                "min_confidence": True,
                "expected_type": _is_expected_type(field_name, value),
                "allowed_value": _is_value_allowed(field_name, value),
                "quote_supported": _value_supported_by_quote(value, quote, field_name),
                "quote_in_context": _quote_in_context(quote, chunks or []),
            }
            computed_confidence = _compute_confidence(
                checks=checks,
                chunks=chunks or [],
                source="regex",
                model_confidence=model_confidence,
            )
            checks["min_confidence"] = computed_confidence >= min_confidence
            ok = all(checks.values())
            meta[field_name] = {
                "accepted": ok,
                "confidence": computed_confidence,
                "model_confidence": model_confidence,
                "value": value if ok else None,
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

    # Phase 4 — re-retrieval verification (disabled by default - causes false negatives)
    from config import ENABLE_RERETRIEVAL_VERIFICATION
    if ENABLE_RERETRIEVAL_VERIFICATION:
        extracted = _post_extract_reretrieval_verification(extracted, field_chunks=field_chunks, collection_name=collection_name)

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