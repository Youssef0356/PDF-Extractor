import traceback
import sys
import os
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

try:
    print("Importing modules...")
    from ImprovedOCR import process_single_image
    from ocr_corrector import OCRCorrector
    
    print("Initializing corrector...")
    corrector = OCRCorrector()
    
    image_path = "Rosemount.jpeg"
    if not os.path.exists(image_path):
        print(f"ERROR: {image_path} not found")
        sys.exit(1)
        
    print(f"Processing {image_path}...")
    res = process_single_image(image_path, corrector)
    
    if res:
        print("SUCCESS!")
        print(f"Config used: {res['config_used']}")
        print("Parsed Data:")
        for k, v in res['parsed_data'].items():
            print(f"  {k}: {v}")
    else:
        print("FAILED (returned None)")
        
except Exception:
    print("EXCEPTION CAUGHT:")
    traceback.print_exc()
