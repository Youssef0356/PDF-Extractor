
import os
import sys
import json

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from services.pdf_parser import open_pdf, iter_pages
from services.regex_extractor import extract_with_regex
from services.llm_extractor import _field_applicable, _normalize_value, _is_expected_type, _is_value_allowed, _value_supported_by_quote, _compute_confidence, _get_min_confidence
from services.document_classifier import classify_document
from config import MIN_EXTRACT_CONFIDENCE

def test_extraction(file_path):
    print(f"Testing PDF: {file_path}")
    pdf = open_pdf(file_path)
    full_text = "\n\n".join([(page.text or "") for page in iter_pages(pdf)])
    
    print("\n--- PHASE 1: REGEX ---")
    regex_results = extract_with_regex(full_text)
    for field in ['equipmentName', 'categorie', 'typeMesure', 'alimentation']:
        if field in regex_results:
            print(f"REGEX {field}: {regex_results[field]}")
        else:
            print(f"REGEX {field}: MISSING")

    print("\n--- PHASE 2: DOCUMENT CLASSIFICATION ---")
    doc_ctx = classify_document(full_text)
    print(f"Doc Type: {doc_ctx.doc_type}, Conf: {doc_ctx.confidence}")

    print("\n--- PHASE 3: APPLICABILITY CHECK ---")
    for field in ['equipmentName', 'categorie', 'typeMesure', 'alimentation', 'reference', 'plageMesure']:
        app = _field_applicable(field, doc_ctx)
        print(f"Field {field} applicable: {app}")

if __name__ == "__main__":
    pdf_path = r"c:\Users\nejiy\Documents\Development\PDF Extractor\backend\uploads\sitransl_lr150_fi01_en.pdf"
    if os.path.exists(pdf_path):
        test_extraction(pdf_path)
    else:
        print(f"File not found: {pdf_path}")
        # List files in uploads to find the right one
        uploads = os.listdir(r"c:\Users\nejiy\Documents\Development\PDF Extractor\backend\uploads")
        print(f"Available uploads: {uploads}")
