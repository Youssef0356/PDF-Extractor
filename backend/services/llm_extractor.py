import json
import logging
import time
import requests
from typing import List, Dict, Any, Optional

from config import OLLAMA_BASE_URL, TEXT_MODEL
from models.schema_v2 import (
    BaseInstrument, 
    ActionneurInstrument, 
    FIELD_DESCRIPTIONS_V2,
    TRANSMETTEUR_ONLY_FIELDS,
    ACTIONNEUR_ONLY_FIELDS,
)
from models.field_options import (
    CODE_MAP, 
    TECHNOLOGY_MAP, 
    SORTIE_TOR_FIELDS,
    MANUAL_ONLY_FIELDS,
    MATERIAU_MEMBRANE_TYPES,
)
from services.field_prompts_v2 import SYSTEM_PROMPT_V2, FIELD_PROMPT_RULES_V2
from services.normalize_v2 import normalize_value_v2, detect_isa_tag_info
from services.feedback import get_corrections_for_field
from services.semantic_search import SemanticSearchService
from services.confidence_scoring import compute_confidence
from services.pre_extraction_classifier import run_pre_extraction, CLASSIFIER_OWNED_FIELDS

logger = logging.getLogger(__name__)

class LLMExtractor:
    def __init__(self):
        self.model = TEXT_MODEL
        self.base_url = OLLAMA_BASE_URL

    def extract_field(
        self, 
        field_name: str, 
        chunks: List[Dict], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extracts a single field from document chunks using Ollama."""
        
        # 0. Check if manual only
        if field_name in MANUAL_ONLY_FIELDS:
            return {"value": None, "confidence": 0.0, "quote": "Manual only field"}

        # 1. Check conditional visibility/applicability
        if not self._is_field_applicable(field_name, context):
            return {"value": None, "confidence": 0.0, "quote": "Field not applicable to current context"}

        # 2. Build prompt
        prompt = self._build_prompt(field_name, chunks, context)

        # 3. Call Ollama
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            raw_json = result.get("response", "{}")
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                # Fallback if LLM failed to produce valid JSON
                data = {"value": None, "confidence": 0.0, "quote": "Invalid JSON response"}
            
            # 4. Normalize + compute blended confidence
            raw_val  = data.get("value")
            norm_val = normalize_value_v2(field_name, raw_val)
            quote    = data.get("quote", "") or ""
            llm_conf = float(data.get("confidence") or 0.5)

            field_meta = FIELD_DESCRIPTIONS_V2.get(field_name, {})
            is_enum    = bool(field_meta.get("allowed_values"))

            blended = compute_confidence(
                field_name=field_name,
                value=norm_val,
                llm_raw_confidence=llm_conf,
                quote=quote,
                chunks=chunks,
                is_enum=is_enum,
            )

            return {
                "value":            norm_val,
                "confidence":       blended,    # blended multi-signal score
                "model_confidence": llm_conf,   # raw LLM value kept for debug
                "quote":            quote,
                "raw_extraction":   raw_val,
            }
        except Exception as e:
            logger.error(f"Error extracting field {field_name}: {e}")
            return {"value": None, "confidence": 0.0, "quote": f"Error: {str(e)}"}

    def _is_field_applicable(self, field_name: str, context: Dict[str, Any]) -> bool:
        """Determines if a field should be extracted based on current context (category, type)."""
        if not context:
            return True
            
        category = context.get("category")
        type_mesure = context.get("typeMesure")
        type_actionneur = context.get("typeActionneur")
        sortie_tor = context.get("sortieTOR", False)
        alimentation = context.get("alimentation")

        # Category constraints
        if category == "Actionneur" and field_name in TRANSMETTEUR_ONLY_FIELDS:
            return False
        if category == "Transmetteur/Capteur" and field_name in ACTIONNEUR_ONLY_FIELDS:
            return False

        # TOR output constraints
        if field_name in SORTIE_TOR_FIELDS and not sortie_tor:
            return False
            
        # Matériau membrane specific to Pression/Niveau
        if field_name == "matériauMembrane":
            if type_mesure and type_mesure not in MATERIAU_MEMBRANE_TYPES:
                return False

        # Communication hidden for loop-powered by default (standard)
        if field_name == "communication" and alimentation == "boucle":
            return False

        return True

    def _build_prompt(self, field_name: str, chunks: List[Dict], context: Dict[str, Any]) -> str:
        """Constructs the LLM prompt for extraction."""
        
        # Gather relevant chunks text
        text_block = "\n---\n".join([c["text"] for c in chunks[:5]])
        
        # Field specific rules
        field_rules = FIELD_PROMPT_RULES_V2.get(field_name, "Extract the value accurately.")
        
        # Allowed values if any
        field_meta = FIELD_DESCRIPTIONS_V2.get(field_name, {})
        allowed = field_meta.get("allowed_values")
        
        # Context summary
        context_str = json.dumps(context, indent=2, ensure_ascii=False) if context else "None"
        
        # Past corrections
        corrections = get_corrections_for_field(field_name)
        corr_str = ""
        if corrections:
            corr_str = "\nPAST ERRORS TO AVOID:\n"
            for c in corrections:
                corr_str += f"- User corrected '{c['ai_extracted_value']}' to '{c['user_corrected_value']}'. Rule: {c.get('rule', 'None')}\n"

        prompt = f"""{SYSTEM_PROMPT_V2}

DOCUMENT EXCERPTS:
{text_block}

FIELD TO EXTRACT: {field_name}
FIELD DESCRIPTION: {field_meta.get('description', '')}
SPECIFIC RULES: {field_rules}
ALLOWED VALUES: {allowed if allowed else 'Open text'}

CURRENT EXTRACTION CONTEXT:
{context_str}
{corr_str}

Return JSON: {{"value": ..., "confidence": ..., "quote": "..."}}
"""
        return prompt

    def extract_all_fields(
        self,
        chunks_per_field: Dict[str, List[Dict]],
        full_text: str = None
    ) -> Dict[str, Any]:
        """Orchestrates extraction of all applicable fields."""
        extracted_data = {}
        confidence_data = {}
        evidence_data = {}

        # ── STEP 1: Deterministic pre-classifier (runs on full text) ─────────
        # Handles typeMesure, code, technologie, marque, signalSortie, hart,
        # alimentation, indiceIP, températureProcess, certificats.
        # These are never sent to the LLM — they are regex-certain.
        if full_text:
            classifier_results = run_pre_extraction(full_text)
            logger.info(f"Classifier results: {json.dumps(classifier_results, indent=2)}")
            for field, result in classifier_results.items():
                extracted_data[field] = result["value"]
                confidence_data[field] = result["confidence"]
                evidence_data[field] = [{
                    "text": f"Classifier: {result['value']}",
                    "confidence": result["confidence"],
                    "source": "classifier"
                }]
            logger.info(f"Pre-classifier locked {len(classifier_results)} fields: "
                        f"{list(classifier_results.keys())}")

        # ── STEP 2: ISA tag (overrides classifier if tag found) ───────────────
        if full_text:
            isa_info = detect_isa_tag_info(full_text)
            if isa_info:
                logger.info(f"ISA Tag detected: {isa_info}")
                for k, v in isa_info.items():
                    if k != "source":
                        extracted_data[k] = v
                        confidence_data[k] = 1.0
                        evidence_data[k] = [{
                            "text": f"ISA Tag: {v}",
                            "source": "isa_rule"
                        }]

        # ── STEP 3: LLM extracts remaining fields only ────────────────────────
        priority_fields = [
            "category", "typeMesure", "typeActionneur", "code",
            "technologie", "alimentation", "sortieTOR"
        ]
        for field in priority_fields:
            if field in extracted_data:
                continue   # already locked by classifier or ISA tag
            res = self.extract_field(field, chunks_per_field.get(field, []), extracted_data)
            if res["value"] is not None:
                extracted_data[field] = res["value"]
                confidence_data[field] = res["confidence"]
                evidence_data[field] = [{"text": res["quote"], "confidence": res["confidence"]}]

        for field in FIELD_DESCRIPTIONS_V2.keys():
            if field in extracted_data:
                continue
            if not FIELD_DESCRIPTIONS_V2[field].get("ai_fills", True):
                continue
            res = self.extract_field(field, chunks_per_field.get(field, []), extracted_data)
            if res["value"] is not None:
                extracted_data[field] = res["value"]
                confidence_data[field] = res["confidence"]
                evidence_data[field] = [{"text": res["quote"], "confidence": res["confidence"]}]

        # ── STEP 4: Final guards ──────────────────────────────────────────────
        self._apply_v2_guards(extracted_data)

        try:
            if extracted_data.get("category") == "Actionneur":
                result_obj = ActionneurInstrument(**extracted_data)
            else:
                result_obj = BaseInstrument(**extracted_data)
            data_dict = result_obj.dict()
        except Exception as e:
            logger.error(f"Error building final instrument model: {e}")
            data_dict = extracted_data

        return {
            "data": data_dict,
            "confidence": confidence_data,
            "evidence": evidence_data,
        }

    def _apply_v2_guards(self, data: Dict[str, Any]):
        """Terminal validation and re-normalization."""
        # 1. HART detection from signalSortie string
        if data.get("signalSortie") and "HART" in str(data.get("signalSortie")).upper():
            data["hart"] = True
            
        # 2. Technology validation against typeMesure
        if data.get("typeMesure") and data.get("technologie"):
            valid_techs = TECHNOLOGY_MAP.get(data["typeMesure"], [])
            if valid_techs and data["technologie"] not in valid_techs and not str(data["technologie"]).startswith("Autre"):
                data["technologie"] = f"Autre: {data['technologie']}"