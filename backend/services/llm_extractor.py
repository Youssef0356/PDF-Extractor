"""
LLM extraction service.
Uses Ollama (Qwen2.5) to extract structured data from document chunks.
"""
import json
import ollama as ollama_client
import time

from config import TEXT_MODEL, VISION_MODEL
from models.schema import FIELD_DESCRIPTIONS, EquipmentSchema


def _build_extraction_prompt(field_name: str, chunks: list[dict]) -> str:
    """Build a prompt for extracting a specific field value."""
    field_info = FIELD_DESCRIPTIONS.get(field_name, {})
    description = field_info.get("description", field_name)
    allowed = field_info.get("allowed_values")
    
    # Combine chunk texts
    context = "\n---\n".join([c["text"] for c in chunks])
    
    allowed_str = ""
    if allowed:
        allowed_str = f"\nThe value MUST be one of: {', '.join(allowed)}"
    
    prompt = f"""You are an industrial document analyzer specializing in equipment datasheets.

Given the following document excerpts, extract the value for the field "{field_name}".
Field description: {description}{allowed_str}

Document excerpts:
\"\"\"
{context}
\"\"\"

IMPORTANT RULES:
- Extract ONLY the value for "{field_name}" if it is EXPLICITLY stated in the excerpts.
- DO NOT guess, infer, or use general knowledge.
- If the exact value is not explicitly stated or is ambiguous, you MUST respond with null.
- For the "plageMesure" field, extract min, max, and unit separately.
- For "sortiesAlarme", extract all alarm entries as a list.
- Respond with ONLY valid JSON, no other text.
- If you return a non-null value, you MUST also return a "quote" copied verbatim from the excerpts that supports the value.

Response format:
{{"value": <extracted_value>, "confidence": <number between 0.0 and 1.0>, "quote": <string or null>}}

For plageMesure, use:
{{"value": {{"min": <number or null>, "max": <number or null>, "unite": "<unit or null>"}}, "confidence": <0.0-1.0>, "quote": <string or null>}}

For sortiesAlarme, use:
{{"value": [{{"nomAlarme": "<name>", "typeAlarme": "<type>", "seuilAlarme": <number>, "uniteAlarme": "<unit>", "relaisAssocie": "<relay>"}}], "confidence": <0.0-1.0>, "quote": <string or null>}}

Your JSON response:"""
    
    return prompt


def extract_field(field_name: str, chunks: list[dict]) -> dict:
    """
    Extract a single field value using the LLM.
    
    Args:
        field_name: The form field to extract.
        chunks: Relevant document chunks from semantic search.
        
    Returns:
        Dict with 'value' and 'confidence'.
    """
    if not chunks:
        return {"value": None, "confidence": 0.0, "quote": None}

    prompt = _build_extraction_prompt(field_name, chunks)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = ollama_client.chat(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
                keep_alive="10m",  # Keep model in VRAM for 10 minutes
            )

            raw = response["message"]["content"].strip()
            result = _parse_llm_json(raw)
            if "quote" not in result:
                result["quote"] = None
            if "confidence" not in result:
                result["confidence"] = 0.0
            if "value" not in result:
                result["value"] = None
            return result

        except Exception as e:
            last_error = e
            wait_s = 2 * attempt
            print(f"[LLM] Error extracting '{field_name}' (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(wait_s)

    print(f"[LLM] Giving up extracting '{field_name}' after 3 attempts: {last_error}")
    return {"value": None, "confidence": 0.0, "quote": None}


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling common formatting issues."""
    if not raw:
        return {"value": None, "confidence": 0.0, "quote": None}

    # Strip markdown code block markers if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
            except json.JSONDecodeError:
                return {"value": None, "confidence": 0.0, "quote": None}
        else:
            return {"value": None, "confidence": 0.0, "quote": None}

    if not isinstance(parsed, dict):
        return {"value": None, "confidence": 0.0, "quote": None}

    parsed.setdefault("value", None)
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("quote", None)
    return parsed


def _normalize_value(field_name: str, value):
    if value is None:
        return None

    # Some models may return a dict for scalar fields, e.g. {"categorie": "Transmetteur"}.
    # Coerce to the inner scalar when possible.
    if isinstance(value, dict):
        if field_name in value and not isinstance(value[field_name], (dict, list)):
            value = value[field_name]
        elif len(value) == 1:
            only_val = next(iter(value.values()))
            if not isinstance(only_val, (dict, list)):
                value = only_val

        # Some models return range-like objects for scalar string fields, e.g.
        # {"min": "SITRANS P320", "max": "SITRANS P320", "unit": "..."}.
        # If min/max are identical scalars, unwrap to that scalar.
        if isinstance(value, dict):
            min_v = value.get("min")
            max_v = value.get("max")
            if min_v is not None and max_v is not None and min_v == max_v and not isinstance(min_v, (dict, list)):
                value = min_v

    if field_name == "typeSignal" and isinstance(value, str):
        v = value.strip()
        v = v.replace("…", "-").replace("–", "-").replace("...", "-")
        compact = v.replace(" ", "").lower()
        if compact in {"4-20ma", "4-20"}:
            return "4-20mA"
        if compact in {"0-20ma", "0-20"}:
            return "0-20mA"
        if compact in {"0-10v", "0-10"}:
            return "0-10V"
        if compact in {"0-5v", "0-5"}:
            return "0-5V"
        return v

    if field_name == "alimentation" and isinstance(value, str):
        v = value.strip()
        v = v.replace("Vdc", "V DC").replace("VDC", "V DC")
        v = v.replace("Vac", "V AC").replace("VAC", "V AC")
        compact = v.replace(" ", "").lower()
        if compact == "24vdc":
            return "24V DC"
        if compact == "220vac":
            return "220V AC"
        return v

    if field_name == "nbFils":
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return value.strip()

    return value


def _is_value_allowed(field_name: str, value) -> bool:
    field_info = FIELD_DESCRIPTIONS.get(field_name, {})
    allowed = field_info.get("allowed_values")
    if value is None or not allowed:
        return True

    if isinstance(value, str):
        return value in allowed
    return False


def _is_expected_type(field_name: str, value) -> bool:
    if value is None:
        return True
    if field_name == "plageMesure":
        return isinstance(value, dict)
    if field_name == "sortiesAlarme":
        return isinstance(value, list)
    return isinstance(value, str)


def _quote_in_context(quote: str | None, chunks: list[dict]) -> bool:
    if not quote or not isinstance(quote, str) or not quote.strip():
        return False
    context = "\n---\n".join([(c.get("text") or "") for c in chunks])
    if not context:
        return False

    q = quote.strip()
    if q in context:
        return True
    return q.lower() in context.lower()


def _value_supported_by_quote(value, quote: str | None) -> bool:
    if value is None:
        return True
    if not quote or not isinstance(quote, str) or not quote.strip():
        return False

    # Pure lexical support check.
    # (Semantic correctness is handled separately in _passes_semantic_guard.)
    return _basic_value_supported_by_quote(value, quote)


def _basic_value_supported_by_quote(value, quote: str) -> bool:
    """Original lexical support check, extracted so we can add semantic guards on top."""
    if isinstance(value, str):
        v = value.strip().lower()
        q = quote.lower()
        if v and v in q:
            return True
        v_compact = "".join(ch for ch in v if ch.isalnum())
        q_compact = "".join(ch for ch in q if ch.isalnum())
        return bool(v_compact) and v_compact in q_compact
    return True


def _passes_semantic_guard(field_name: str, value, quote: str | None) -> bool:
    """Extra field-specific checks to prevent semantically wrong but text-supported values."""
    if value is None:
        return True

    q = (quote or "").strip()
    ql = q.lower()

    if field_name == "dateCalibration":
        # Accept only if quote contains calibration intent.
        # Reject common doc metadata.
        if any(k in ql for k in ["last modified", "modified", "revision", "rev.", "édition", "edition", "version", "published"]):
            return False
        return any(k in ql for k in ["calibration", "calibrated", "étalonn", "etalonn", "certificate", "certificat"])

    if field_name == "plageMesure":
        if not isinstance(value, dict):
            return False
        # Reject timing/delay contexts (ms/us) unless it's explicitly a measuring range.
        # PLC datasheets frequently contain "input delay" and similar.
        if any(k in ql for k in ["delay", "time", "cycle", "bit operation", "input delay", "output delay", "cpu", "reaction time"]):
            return False
        if any(u in ql for u in [" ms", "µs", " us"]):
            # If the quote is in time units, treat as non-measurement range.
            return False
        return True

    if field_name == "reperage":
        if not isinstance(value, str):
            return False
        # Reperage should look like a tag/label, not an electrical characteristic.
        # Reject if quote looks like current/voltage spec.
        if any(u in ql for u in ["ma", " a", " v", " ohm", "residual current", "current"]):
            return False
        return True

    return True


def extract_all_fields(field_chunks: dict[str, list[dict]], min_confidence: float = 0.3) -> EquipmentSchema:
    """
    Extract all form fields from the document in parallel.
    
    Args:
        field_chunks: Dict mapping field_name -> relevant chunks
                     (from semantic_search.search_all_fields).
        min_confidence: Minimum confidence threshold required to accept a field value.
        
    Returns:
        Populated EquipmentSchema.
    """
    from concurrent.futures import ThreadPoolExecutor
    from config import LLM_MAX_WORKERS
    
    extracted = {}
    
    def process_field(field_name, chunks):
        print(f"  [LLM] START: {field_name} (from {len(chunks)} chunks)...")
        start_time = time.time()
        result = extract_field(field_name, chunks)
        elapsed = time.time() - start_time
        print(f"  [LLM] FINISH: {field_name} in {elapsed:.1f}s")
        return field_name, result

    # Run extractions in parallel
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_field, name, chunks) 
            for name, chunks in field_chunks.items()
        ]
        
        for future in futures:
            field_name, result = future.result()
            value = _normalize_value(field_name, result.get("value"))
            confidence = result.get("confidence")
            if confidence is None:
                confidence = 0.0
            quote = result.get("quote")

            ok = (
                value is not None
                and confidence >= min_confidence
                and _is_value_allowed(field_name, value)
                and _value_supported_by_quote(value, quote)
                and _passes_semantic_guard(field_name, value, quote)
                and _quote_in_context(quote, field_chunks.get(field_name, []))
            )
            if ok:
                extracted[field_name] = value
                print(f"  [OK] {field_name} = {value}")
            else:
                print(f"  [MISS] {field_name} (conf: {confidence})")
    
    # Build the EquipmentSchema from extracted data
    schema_data = {}
    for field_name, value in extracted.items():
        if field_name == "plageMesure" and isinstance(value, dict):
            schema_data["plageMesure"] = value
        elif field_name == "sortiesAlarme" and isinstance(value, list):
            schema_data["sortiesAlarme"] = value
        else:
            schema_data[field_name] = value
    
    return EquipmentSchema(**schema_data)


def extract_all_fields_with_meta(
    field_chunks: dict[str, list[dict]],
    min_confidence: float = 0.3,
    regex_results: dict[str, dict] | None = None,
) -> tuple[EquipmentSchema, dict[str, dict]]:
    from concurrent.futures import ThreadPoolExecutor
    from config import LLM_MAX_WORKERS

    if regex_results is None:
        regex_results = {}

    extracted: dict[str, object] = {}
    meta: dict[str, dict] = {}

    # --- Phase 1: Accept regex-extracted fields immediately ---
    llm_fields: dict[str, list[dict]] = {}
    for field_name, chunks in field_chunks.items():
        if field_name in regex_results:
            rx = regex_results[field_name]
            value = rx.get("value")
            confidence = rx.get("confidence", 1.0)
            quote = rx.get("quote")
            extracted[field_name] = value
            meta[field_name] = {
                "accepted": True,
                "confidence": confidence,
                "value": value,
                "quote": quote,
                "source": "regex",
            }
            print(f"  [REGEX] {field_name} = {value}")
        else:
            llm_fields[field_name] = chunks

    # --- Phase 2: LLM extraction for remaining fields ---
    def process_field(field_name, chunks):
        print(f"  [LLM] START: {field_name} (from {len(chunks)} chunks)...")
        start_time = time.time()
        result = extract_field(field_name, chunks)
        elapsed = time.time() - start_time
        print(f"  [LLM] FINISH: {field_name} in {elapsed:.1f}s")
        return field_name, result

    if llm_fields:
        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_field, name, chunks)
                for name, chunks in llm_fields.items()
            ]

            for future in futures:
                field_name, result = future.result()
                value = _normalize_value(field_name, result.get("value"))
                confidence = result.get("confidence")
                if confidence is None:
                    confidence = 0.0
                quote = result.get("quote")

                ok = (
                    value is not None
                    and confidence >= min_confidence
                    and _is_expected_type(field_name, value)
                    and _is_value_allowed(field_name, value)
                    and _value_supported_by_quote(value, quote)
                    and _passes_semantic_guard(field_name, value, quote)
                    and _quote_in_context(quote, field_chunks.get(field_name, []))
                )

                meta[field_name] = {
                    "accepted": ok,
                    "confidence": confidence,
                    "value": value if ok else None,
                    "quote": quote,
                    "source": "llm",
                }

                if ok:
                    extracted[field_name] = value
                    print(f"  [OK] {field_name} = {value}")
                else:
                    print(f"  [MISS] {field_name} (conf: {confidence})")

    schema_data = {}
    for field_name, value in extracted.items():
        if field_name == "plageMesure" and isinstance(value, dict):
            schema_data["plageMesure"] = value
        elif field_name == "sortiesAlarme" and isinstance(value, list):
            schema_data["sortiesAlarme"] = value
        else:
            schema_data[field_name] = value

    return EquipmentSchema(**schema_data), meta
