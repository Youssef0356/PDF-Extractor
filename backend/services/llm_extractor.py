"""
LLM extraction service.
Uses Ollama (Qwen2.5) to extract structured data from document chunks.
"""
import json
import ollama as ollama_client

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

Response format:
{{"value": <extracted_value>, "confidence": <number between 0.0 and 1.0>}}

For plageMesure, use:
{{"value": {{"min": <number or null>, "max": <number or null>, "unite": "<unit or null>"}}, "confidence": <0.0-1.0>}}

For sortiesAlarme, use:
{{"value": [{{"nomAlarme": "<name>", "typeAlarme": "<type>", "seuilAlarme": <number>, "uniteAlarme": "<unit>", "relaisAssocie": "<relay>"}}], "confidence": <0.0-1.0>}}

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
        return {"value": None, "confidence": 0.0}
    
    prompt = _build_extraction_prompt(field_name, chunks)
    
    try:
        response = ollama_client.chat(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
            keep_alive="10m",  # Keep model in VRAM for 10 minutes
        )
        
        raw = response["message"]["content"].strip()
        result = _parse_llm_json(raw)
        return result
        
    except Exception as e:
        print(f"[LLM] Error extracting '{field_name}': {e}")
        return {"value": None, "confidence": 0.0}


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling common formatting issues."""
    # Strip markdown code block markers if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    
    return {"value": None, "confidence": 0.0}


def extract_all_fields(field_chunks: dict[str, list[dict]]) -> EquipmentSchema:
    """
    Extract all form fields from the document in parallel.
    
    Args:
        field_chunks: Dict mapping field_name -> relevant chunks
                      (from semantic_search.search_all_fields).
                      
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

    # Import time for logging
    import time

    # Run extractions in parallel
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_field, name, chunks) 
            for name, chunks in field_chunks.items()
        ]
        
        for future in futures:
            field_name, result = future.result()
            value = result.get("value")
            confidence = result.get("confidence", 0.0)
            
            if value is not None and confidence > 0.3:
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
