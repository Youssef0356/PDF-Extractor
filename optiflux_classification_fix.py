#!/usr/bin/env python3
"""
Proposed fix for OPTIFLUX "capteur de mesure" classification issue.

ROOT CAUSE ANALYSIS:
1. The LLM classification prompt in llm_extractor.py has a mapping alias that converts
   "capteur" → "Capteurs" but the prompt rules don't explicitly prioritize "capteur de mesure" 
   over "système de mesure" when both phrases appear in context.

2. The evidence shows "Système de mesure" appears in table contexts with higher frequency,
   leading the LLM to prefer this classification despite "capteur de mesure" being the 
   more specific and accurate description.

3. The confidence threshold for "categorie" field is set to 0.25, which is too low to 
   distinguish between similar categories.

PROPOSED FIXES:
"""

# PROPOSED FIX 1: Enhanced prompt rules for categorie field
ENHANCED_CATEGORIE_PROMPT = '''
Classify the equipment into ONE of: Transmetteur, Débitmètre, Capteurs, Actionneur, Automate, IHM, Autre.

PRIORITY RULES (apply in order):
1. If the document contains "capteur de mesure" or "capteur de débit" or "capteur de niveau" → Capteurs
2. If the document contains "transmetteur" or "transmitter" → Transmetteur  
3. If the document contains "débitmètre" or "flowmeter" → Débitmètre
4. If the document contains "système de mesure" but NOT "capteur de mesure" → analyze context

CONTEXT ANALYSIS:
- Radar level transmitters, level transmitters, distance/level sensors with 4-20mA output → Transmetteur
- Flow meters (débitmètres, variable area, Coriolis, vortex, electromagnetic flow) → Débitmètre
- Pressure/level/temp transmitters with 4-20mA output → Transmetteur
- PLCs/CPUs/controllers → Automate
- Touchscreens/HMI panels → IHM
- Passive sensors WITHOUT signal conditioning → Capteurs
- Valves/positioners/actuators → Actionneur

IMPORTANT: 
- "capteur de mesure" explicitly indicates Capteurs category
- "système de mesure" is generic and should be overridden by more specific terms
- For the quote, use the EXACT verbatim fragment from the source language
- The "value" must be one of the French labels above, but the "quote" must be from the document.

GOOD: 'Capteurs' with quote 'capteur de mesure électromagnétique' | BAD: 'Système de mesure' when 'capteur de mesure' is present
'''

# PROPOSED FIX 2: Enhanced alias mapping with multi-word phrases
ENHANCED_CATEGORIE_ALIAS = {
    # Priority mappings for capteur phrases
    "capteur de mesure": "Capteurs",
    "capteur de débit": "Capteurs", 
    "capteur de niveau": "Capteurs",
    "capteur de pression": "Capteurs",
    "capteur de température": "Capteurs",
    
    # Existing mappings
    "transmitter": "Transmetteur", 
    "transmetteur": "Transmetteur",
    "débitmètre": "Débitmètre", 
    "debitmetre": "Débitmètre", 
    "flowmeter": "Débitmètre",
    "flow meter": "Débitmètre", 
    "compteur": "Débitmètre", 
    "rotameter": "Débitmètre",
    "capteur": "Capteurs", 
    "capteurs": "Capteurs", 
    "sensor": "Capteurs", 
    "sensors": "Capteurs",
    "actionneur": "Actionneur", 
    "actuator": "Actionneur",
    "automate": "Automate", 
    "plc": "Automate", 
    "cpu": "Automate",
    "ihm": "IHM", 
    "hmi": "IHM",
    "autre": "Autre", 
    "other": "Autre",
}

# PROPOSED FIX 3: Confidence threshold adjustment
ENHANCED_FIELD_MIN_CONFIDENCE_OVERRIDE = {
    "typeSignal": 0.25,   # table dots vs ellipsis causes quote mismatches
    "alimentation": 0.25, # same issue with voltage tables  
    "categorie": 0.45,    # INCREASED from 0.25 for better category discrimination
}

# PROPOSED FIX 4: Post-processing validation function
def validate_categorie_classification(extracted_value: str, quote: str, chunks: list) -> str:
    """
    Post-process category classification to fix common mis-classifications.
    
    Args:
        extracted_value: The category value extracted by LLM
        quote: The supporting quote from the document
        chunks: The document chunks for context analysis
    
    Returns:
        Corrected category value if mis-classification detected
    """
    if not isinstance(extracted_value, str) or not quote:
        return extracted_value
    
    # Combine all chunk text for context analysis
    full_context = " ".join(chunk.get("text", "") for chunk in chunks).lower()
    
    # Rule 1: If "capteur de mesure" is present but classified as something else, correct to "Capteurs"
    capteur_phrases = ["capteur de mesure", "capteur de débit", "capteur de niveau", 
                      "capteur de pression", "capteur de température"]
    
    for phrase in capteur_phrases:
        if phrase in full_context and extracted_value != "Capteurs":
            print(f"[CATEGORY FIX] Found '{phrase}' but classified as '{extracted_value}' → correcting to 'Capteurs'")
            return "Capteurs"
    
    # Rule 2: Don't classify as "Système de mesure" if specific sensor type is mentioned
    if extracted_value == "Système de mesure":
        # Check if any specific sensor terms are present
        specific_terms = ["capteur", "sensor", "transmetteur", "transmitter", 
                         "débitmètre", "flowmeter"]
        for term in specific_terms:
            if term in full_context:
                print(f"[CATEGORY FIX] Classified as 'Système de mesure' but '{term}' found → needs manual review")
                return None  # Force null to trigger manual review
    
    return extracted_value

# PROPOSED FIX 5: Unit tests for the fix
import unittest

class TestOptifluxClassificationFix(unittest.TestCase):
    """Unit tests for the OPTIFLUX classification fix."""
    
    def test_capteur_de_mesure_classification(self):
        """Test that 'capteur de mesure' correctly classifies as Capteurs."""
        test_cases = [
            {
                "context": "OPTIFLUX 2000 - Capteur de mesure électromagnétique pour liquides conducteurs",
                "expected": "Capteurs"
            },
            {
                "context": "SITRANS LR150 - Capteur de niveau radar pour liquides et solides", 
                "expected": "Capteurs"
            },
            {
                "context": "H250 M40 - Capteur de débit à section variable avec sortie 4-20mA",
                "expected": "Capteurs"
            },
            {
                "context": "Capteur de pression piezo-résistif - Capteur de mesure pour fluides industriels",
                "expected": "Capteurs"
            },
            {
                "context": "Capteur de température NTC M12 - Capteur de mesure pour applications industrielles",
                "expected": "Capteurs"
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(context=test_case["context"]):
                # Create mock chunks
                chunks = [{"text": test_case["context"]}]
                
                # Test the validation function
                result = validate_categorie_classification(
                    "Système de mesure",  # Simulate LLM mis-classification
                    "some quote",
                    chunks
                )
                
                self.assertEqual(result, test_case["expected"], 
                    f"Expected {test_case['expected']} but got {result} for context: {test_case['context']}")
    
    def test_systeme_de_mesure_with_specific_terms(self):
        """Test that 'Système de mesure' classification triggers review when specific terms present."""
        test_cases = [
            {
                "context": "Le système de mesure OPTIFLUX utilise un capteur électromagnétique",
                "should_trigger_review": True
            },
            {
                "context": "Système de mesure avec transmetteur 4-20mA intégré",
                "should_trigger_review": True
            },
            {
                "context": "Système de mesure générique sans spécifications techniques",
                "should_trigger_review": False
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(context=test_case["context"]):
                chunks = [{"text": test_case["context"]}]
                
                result = validate_categorie_classification(
                    "Système de mesure",
                    "some quote", 
                    chunks
                )
                
                if test_case["should_trigger_review"]:
                    self.assertIsNone(result, "Should trigger manual review (return None)")
                else:
                    self.assertEqual(result, "Système de mesure", "Should keep original classification")

# Implementation guide for applying the fix
def apply_optiflux_fix():
    """
    Step-by-step guide to apply the OPTIFLUX classification fix.
    
    1. Update FIELD_PROMPT_RULES["categorie"] in llm_extractor.py with ENHANCED_CATEGORIE_PROMPT
    2. Update the ALIAS["categorie"] mapping with ENHANCED_CATEGORIE_ALIAS  
    3. Update FIELD_MIN_CONFIDENCE_OVERRIDE with ENHANCED_FIELD_MIN_CONFIDENCE_OVERRIDE
    4. Add validate_categorie_classification() function call in the categorie extraction pipeline
    5. Run the unit tests to verify the fix works correctly
    """
    print("OPTIFLUX Classification Fix Implementation Guide:")
    print("1. Replace FIELD_PROMPT_RULES['categorie'] with ENHANCED_CATEGORIE_PROMPT")
    print("2. Update ALIAS['categorie'] mapping with ENHANCED_CATEGORIE_ALIAS")
    print("3. Update FIELD_MIN_CONFIDENCE_OVERRIDE with new threshold")
    print("4. Integrate validate_categorie_classification() into extraction pipeline")
    print("5. Run unit tests: python -m pytest test_optiflux_fix.py")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)