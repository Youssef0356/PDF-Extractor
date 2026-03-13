#!/usr/bin/env python3
"""
Minimal reproducible test case for OPTIFLUX "capteur de mesure" mis-classification issue.
This test demonstrates the specific case where "optiflux" is described as "capteur de mesure"
but gets classified as "système de mesure" instead of "capteurs".
"""

import json
from backend.services.llm_extractor import extract_field
from backend.services.document_classifier import DocumentContext

def test_optiflux_capteur_classification():
    """Test case reproducing the OPTIFLUX mis-classification issue."""
    
    # Minimal document excerpt containing the key phrase "capteur de mesure"
    # This simulates page 1 content where optiflux is described as "capteur de mesure"
    test_chunks = [
        {
            "text": """OPTIFLUX 2000
Notice technique
Capteur de mesure électromagnétique pour liquides conducteurs
Le OPTIFLUX 2000 est un capteur de mesure de débit électromagnétique destiné aux applications industrielles.""",
            "distance": 0.2,
            "page_number": 1,
            "chunk_id": "test_p1_c1"
        }
    ]
    
    # Document context for instrumentation document
    doc_ctx = DocumentContext(
        doc_type="instrument_datasheet",
        confidence=0.85,
        rationale="instrument_score=8, hmi_score=2"
    )
    
    # Test the categorie field extraction
    result = extract_field("categorie", test_chunks, doc_ctx)
    
    print("=== OPTIFLUX Classification Test Results ===")
    print(f"Extracted value: {result.get('value')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Quote: {result.get('quote')}")
    print(f"Raw LLM response: {result.get('_raw', 'N/A')}")
    
    # The expected behavior: should classify as "Capteurs" due to "capteur de mesure" phrase
    # Current buggy behavior: likely classifies as "Système de mesure"
    expected_value = "Capteurs"
    actual_value = result.get('value')
    
    print(f"\nExpected: {expected_value}")
    print(f"Actual: {actual_value}")
    print(f"Test {'PASSED' if actual_value == expected_value else 'FAILED'}")
    
    return result

def test_multiple_capteur_samples():
    """Test multiple "capteur de mesure" samples to ensure robust classification."""
    
    test_cases = [
        {
            "name": "OPTIFLUX - capteur de mesure",
            "text": "OPTIFLUX 2000 - Capteur de mesure électromagnétique pour applications industrielles",
            "expected": "Capteurs"
        },
        {
            "name": "SITRANS - capteur de niveau",
            "text": "SITRANS LR150 - Capteur de niveau radar pour liquides et solides",
            "expected": "Capteurs"
        },
        {
            "name": "H250 - capteur de débit",
            "text": "H250 M40 - Capteur de débit à section variable avec sortie 4-20mA",
            "expected": "Capteurs"
        },
        {
            "name": "Temperature sensor",
            "text": "Capteur de température NTC M12 - Capteur de mesure pour applications industrielles",
            "expected": "Capteurs"
        },
        {
            "name": "Pressure sensor",
            "text": "Capteur de pression piezo-résistif - Capteur de mesure pour fluides industriels",
            "expected": "Capteurs"
        }
    ]
    
    doc_ctx = DocumentContext(
        doc_type="instrument_datasheet",
        confidence=0.85,
        rationale="instrument_score=8, hmi_score=2"
    )
    
    results = []
    
    print("\n=== Multiple Capteur Samples Test ===")
    for test_case in test_cases:
        chunks = [{
            "text": test_case["text"],
            "distance": 0.2,
            "page_number": 1,
            "chunk_id": f"test_{test_case['name'].lower().replace(' ', '_')}"
        }]
        
        result = extract_field("categorie", chunks, doc_ctx)
        
        passed = result.get('value') == test_case['expected']
        results.append({
            'name': test_case['name'],
            'expected': test_case['expected'],
            'actual': result.get('value'),
            'confidence': result.get('confidence'),
            'passed': passed
        })
        
        print(f"{test_case['name']}: {result.get('value')} (conf: {result.get('confidence')}) - {'PASS' if passed else 'FAIL'}")
    
    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"\nSummary: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    print("Testing OPTIFLUX capteur de mesure classification...")
    
    # Run the main test case
    optiflux_result = test_optiflux_capteur_classification()
    
    # Run multiple sample tests
    sample_results = test_multiple_capteur_samples()
    
    # Save results for analysis
    results = {
        "optiflux_test": optiflux_result,
        "sample_tests": sample_results,
        "summary": {
            "total_samples": len(sample_results),
            "passed_samples": sum(1 for r in sample_results if r['passed'])
        }
    }
    
    with open("capteur_classification_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nTest results saved to capteur_classification_test_results.json")