"""
LLM extraction service.
Uses Ollama (Qwen2.5) to extract structured data from document chunks.
"""
import json
import ollama as ollama_client
import re
import time

from config import ACCURACY_FIRST, TEXT_MODEL, VISION_MODEL
from models.schema import FIELD_DESCRIPTIONS, EquipmentSchema
from services.document_classifier import DocumentContext


def _build_extraction_prompt(field_name: str, chunks: list[dict], doc_ctx: DocumentContext | None = None) -> str:
    """Build a prompt for extracting a specific field value."""
    field_info = FIELD_DESCRIPTIONS.get(field_name, {})
    description = field_info.get("description", field_name)
    allowed = field_info.get("allowed_values")
    
    # Combine chunk texts
    context = "\n---\n".join([c["text"] for c in chunks])
    
    allowed_str = ""
    if allowed:
        allowed_str = f"\nThe value MUST be one of: {', '.join(allowed)}"
    
    doc_ctx_str = ""
    if doc_ctx is not None:
        doc_ctx_str = (
            f"\nDocument context:\n"
            f"- doc_type: {doc_ctx.doc_type}\n"
            f"- classifier_confidence: {doc_ctx.confidence}\n"
            f"- rationale: {doc_ctx.rationale}\n"
        )

    equipment_name_rule = ""
    if field_name == "equipmentName":
        equipment_name_rule = (
            "- For \"equipmentName\": this is the COMMERCIAL product name only (human-readable). "
            "Do NOT return part numbers / order numbers / references (e.g., strings like 6AV..., 6ES7..., 7MF...). "
            "If you only see a reference/order number and no explicit commercial name, return null.\n"
        )

    prompt = f"""You are an industrial document analyzer specializing in equipment datasheets.
{doc_ctx_str}

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
- For the "categorie" field: extract it ONLY if the category word is explicitly written in the excerpts (e.g., "Transmetteur" / "Actionneur"). Otherwise return null.
{equipment_name_rule}- Apply domain constraints:
  - If typeMesure is Pression: units must be consistent with pressure (bar, Pa, psi, kPa, MPa, mbar). Never return flow units (m³/h, L/h).
  - If technologie is Ultrasonique: typeSignal is probably an electrical signal (e.g., 4-20mA or voltage) and should not be a communication protocol.
- For the "plageMesure" field, extract min, max, and unit separately.
- For the "plageMesure" field: ONLY accept physical measurement ranges with real units (e.g., m3/h, l/h, bar, °C, %). If the excerpt is about a ratio like "10:1" / "100:1" (rangeability / turndown / plage étendue), you MUST return null.
- For "sortiesAlarme", extract all alarm entries as a list.
- Respond with ONLY valid JSON, no other text.
- If you return a non-null value, you MUST also return a "quote" copied verbatim from the excerpts that supports the value.
- The quote MUST be an exact substring of the excerpts (do not paraphrase).
- DO NOT use "..." or "…" in the quote. If you cannot copy a fully verbatim quote, return null.

Response format:
{{"value": <extracted_value>, "confidence": <number between 0.0 and 1.0>, "quote": <string or null>}}

For plageMesure, use:
{{"value": {{"min": <number or null>, "max": <number or null>, "unite": "<unit or null>"}}, "confidence": <0.0-1.0>, "quote": <string or null>}}

For sortiesAlarme, use:
{{"value": [{{"nomAlarme": "<name>", "typeAlarme": "<type>", "seuilAlarme": <number>, "uniteAlarme": "<unit>", "relaisAssocie": "<relay>"}}], "confidence": <0.0-1.0>, "quote": <string or null>}}

Your JSON response:"""
    
    return prompt


def _build_strict_json_prompt(field_name: str, chunks: list[dict], doc_ctx: DocumentContext | None = None) -> str:
    """A stricter prompt variant used when the model drifts away from JSON."""
    base = _build_extraction_prompt(field_name, chunks, doc_ctx=doc_ctx)
    return (
        base
        + "\n\nSTRICT MODE: Output MUST be a single JSON object and MUST start with '{' and end with '}'. "
        + "Do not output any other characters before or after the JSON."
    )


def extract_field(field_name: str, chunks: list[dict], doc_ctx: DocumentContext | None = None) -> dict:
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

    prompt = _build_extraction_prompt(field_name, chunks, doc_ctx=doc_ctx)

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

            # If the model drifted away from JSON, retry once with a stricter instruction.
            if attempt < 3 and (result.get("value") is None and (not raw.lstrip().startswith("{") or not raw.rstrip().endswith("}"))):
                prompt = _build_strict_json_prompt(field_name, chunks, doc_ctx=doc_ctx)
                raise ValueError("LLM did not return JSON; retrying with strict JSON prompt")

            result["_raw"] = raw[:1200]
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
    return {"value": None, "confidence": 0.0, "quote": None, "_raw": None, "_error": str(last_error) if last_error else None}


def _field_applicable(field_name: str, doc_ctx: DocumentContext | None) -> bool:
    """Return False when a field should be forced null for a given document type."""
    if doc_ctx is None:
        return True

    # For HMI manuals/spec sheets, these schema fields (instrument taxonomy) are usually inapplicable.
    if doc_ctx.doc_type == "hmi_manual":
        if field_name in {"categorie", "typeMesure", "technologie", "plageMesure", "sortiesAlarme", "dateCalibration", "classe"}:
            return False

    return True


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling common formatting issues."""
    if not raw:
        return {"value": None, "confidence": 0.0, "quote": None}

    # Strip markdown code block markers if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    def _extract_first_json_object(text: str) -> str | None:
        s = text or ""
        start = s.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]

            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(raw)
        if extracted:
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                print(f"[LLM] JSON parse failed (showing first 240 chars): {raw[:240]!r}")
                return {"value": None, "confidence": 0.0, "quote": None}
        else:
            print(f"[LLM] JSON parse failed (no '{{' found; first 240 chars): {raw[:240]!r}")
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
    if not ACCURACY_FIRST:
        return True

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

    # Quotes containing ellipses are not verbatim and should not be accepted.
    if "..." in quote or "…" in quote:
        return False

    context = "\n---\n".join([(c.get("text") or "") for c in chunks])
    if not context:
        return False

    q = quote.strip()
    if q in context:
        return True

    def _norm(s: str) -> str:
        # Normalize whitespace for robust matching, but keep content strict.
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"\s+", " ", s)
        return s.strip().lower()

    qn = _norm(q)
    cn = _norm(context)

    # If the model copied a multi-line quote, it may not appear as one contiguous substring
    # because of line breaks or repeated headers. We still want verbatim behavior, so we
    # only accept if each non-empty line appears in the context *in order*.
    if "\n" in q:
        parts = [p.strip() for p in q.splitlines() if p.strip()]
        if not parts:
            return False

        pos = 0
        for p in parts:
            pn = _norm(p)
            if not pn:
                continue
            idx = cn.find(pn, pos)
            if idx < 0:
                return False
            pos = idx + len(pn)
        return True

    return bool(qn) and qn in cn


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
        # Reject rangeability ratios and rangeability contexts being misread as a measuring range.
        if re.search(r"\b\d+\s*:\s*\d+\b", ql):
            return False
        if "rangeability" in ql or "turndown" in ql:
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


def extract_all_fields(
    field_chunks: dict[str, list[dict]],
    min_confidence: float = 0.3,
    doc_ctx: DocumentContext | None = None,
) -> EquipmentSchema:
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
        result = extract_field(field_name, chunks, doc_ctx=doc_ctx)
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
    extracted = _apply_cross_field_guards(extracted)
    extracted = _post_extract_reretrieval_verification(extracted, field_chunks=field_chunks)

    schema_data = {}
    for field_name, value in extracted.items():
        if field_name == "plageMesure" and isinstance(value, dict):
            schema_data["plageMesure"] = value
        elif field_name == "sortiesAlarme" and isinstance(value, list):
            schema_data["sortiesAlarme"] = value
        else:
            schema_data[field_name] = value
    
    return EquipmentSchema(**schema_data)


def _normalize_for_query(field_name: str, value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if field_name == "plageMesure" and isinstance(value, dict):
        min_v = value.get("min")
        max_v = value.get("max")
        unite = value.get("unite")
        return f"{min_v} {unite} {max_v}"
    if field_name == "sortiesAlarme" and isinstance(value, list):
        names = []
        for a in value:
            if isinstance(a, dict) and a.get("nomAlarme"):
                names.append(str(a.get("nomAlarme")))
        return " ".join(names)
    return str(value)


def _top_result_supports_value(field_name: str, value, top_chunk: dict | None) -> bool:
    if value is None:
        return True
    if not top_chunk:
        return False

    text = (top_chunk.get("text") or "")
    if not text:
        return False

    # Reuse the existing lexical support check against the chunk text.
    if not _basic_value_supported_by_quote(value, text):
        return False

    # Minimal semantic guard as a second pass.
    return _passes_semantic_guard(field_name, value, text)


def _post_extract_reretrieval_verification(
    extracted: dict[str, object],
    field_chunks: dict[str, list[dict]] | None = None,
) -> dict[str, object]:
    """Re-query Chroma with the extracted value and ensure the top chunk supports it.

    If the top result does not support the value, mark the field as suspicious and drop it.
    """
    try:
        from services.vector_store import query_chunks
    except Exception:
        return extracted

    doc_id = None
    if field_chunks:
        for chunks in field_chunks.values():
            for c in chunks or []:
                md = c.get("metadata") if isinstance(c, dict) else None
                if md and md.get("doc_id"):
                    doc_id = md.get("doc_id")
                    break
            if doc_id:
                break

    where = {"doc_id": doc_id} if doc_id else None

    verified: dict[str, object] = {}
    for field_name, value in extracted.items():
        qv = _normalize_for_query(field_name, value).strip()
        if not qv:
            continue

        query = f"What is {qv}?"
        results = query_chunks(query, n_results=1, where=where)
        top = results[0] if results else None
        if _top_result_supports_value(field_name, value, top):
            verified[field_name] = value
        else:
            print(f"  [VERIFY FAIL] {field_name}='{value}' not supported by top re-retrieval")

    return verified


def extract_all_fields_with_meta(
    field_chunks: dict[str, list[dict]],
    min_confidence: float = 0.3,
    regex_results: dict[str, dict] | None = None,
    doc_ctx: DocumentContext | None = None,
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
        result = extract_field(field_name, chunks, doc_ctx=doc_ctx)
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

                checks = {
                    "non_null": value is not None,
                    "min_confidence": confidence >= min_confidence,
                    "expected_type": _is_expected_type(field_name, value),
                    "allowed_value": _is_value_allowed(field_name, value),
                    "quote_supported": _value_supported_by_quote(value, quote),
                    "semantic_guard": _passes_semantic_guard(field_name, value, quote),
                    "quote_in_context": _quote_in_context(quote, field_chunks.get(field_name, [])),
                }

                ok = (
                    checks["non_null"]
                    and checks["min_confidence"]
                    and checks["expected_type"]
                    and checks["allowed_value"]
                    and checks["quote_supported"]
                    and checks["semantic_guard"]
                    and checks["quote_in_context"]
                )

                rejection_reason = None
                if not ok:
                    for k in [
                        "non_null",
                        "min_confidence",
                        "expected_type",
                        "allowed_value",
                        "quote_supported",
                        "semantic_guard",
                        "quote_in_context",
                    ]:
                        if not checks.get(k, False):
                            rejection_reason = k
                            break

                meta[field_name] = {
                    "accepted": ok,
                    "confidence": confidence,
                    "value": value if ok else None,
                    "quote": quote,
                    "source": "llm",
                    "checks": checks,
                    "rejection_reason": rejection_reason,
                    "raw": result.get("_raw"),
                    "error": result.get("_error"),
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


def _apply_cross_field_guards(extracted: dict[str, object]) -> dict[str, object]:
    """Apply cross-field consistency checks to reduce wrong-but-plausible outputs."""
    out = dict(extracted)

    def _looks_like_reference(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        # Common industrial order numbers: long alphanumeric tokens and vendor prefixes.
        if re.fullmatch(r"[A-Z0-9\-_/\.]{8,}", t.upper()):
            if any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t):
                return True
        if re.match(r"^(6AV|6ES7|7MF)\b", t.strip(), flags=re.IGNORECASE):
            return True
        return False

    # equipmentName must be a commercial name; never accept a technical reference/order number.
    eq = out.get("equipmentName")
    if isinstance(eq, str):
        ref = out.get("reference")
        if _looks_like_reference(eq) or (isinstance(ref, str) and eq.strip() == ref.strip()):
            out.pop("equipmentName", None)

    type_mesure = (out.get("typeMesure") or "")
    if isinstance(type_mesure, str) and type_mesure.lower() == "pression":
        pm = out.get("plageMesure")
        if isinstance(pm, dict):
            unite = pm.get("unite")
            if isinstance(unite, str):
                allowed_pressure_units = {"bar", "pa", "psi", "kpa", "mpa", "mbar"}
                if unite.strip().lower() not in allowed_pressure_units:
                    out.pop("plageMesure", None)

    techno = (out.get("technologie") or "")
    if isinstance(techno, str) and techno.strip().lower() == "ultrasonique":
        ts = out.get("typeSignal")
        if isinstance(ts, str) and ts.strip() in {"Autre", ""}:
            # Don't force a value; just drop an unhelpful placeholder.
            out.pop("typeSignal", None)

    return out
