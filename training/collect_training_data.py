"""
Training Data Collector for Tesseract Fine-Tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Collects image-text pairs from processed equipment labels.
These pairs can be used to fine-tune Tesseract's LSTM engine
on your specific domain.

Usage:
    # Automatic: Collected automatically when ImprovedOCR.py runs
    
    # Manual: Add ground truth for specific images
    python collect_training_data.py path/to/image.jpeg "CORRECT TEXT HERE"
    
    # From PDF pages
    python collect_training_data.py path/to/document.pdf
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ImprovedOCR import preprocess_for_ocr, save_debug

TRAINING_DATA_DIR = Path(__file__).parent / "data"
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_next_index() -> int:
    """Get the next available training pair index."""
    existing = [f for f in TRAINING_DATA_DIR.iterdir() if f.suffix == ".gt.txt"]
    return len(existing) + 1


def save_training_pair(img: np.ndarray, ground_truth: str, prefix: str = "equipment"):
    """
    Save an image and its ground truth text as a training pair.
    
    Args:
        img: Preprocessed grayscale image
        ground_truth: Correct text for this image
        prefix: File name prefix
    """
    idx = get_next_index()
    
    img_path = TRAINING_DATA_DIR / f"{prefix}_{idx:04d}.tif"
    gt_path = TRAINING_DATA_DIR / f"{prefix}_{idx:04d}.gt.txt"
    
    # Save as TIFF (required format for Tesseract training)
    cv2.imwrite(str(img_path), img)
    
    # Save ground truth text
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write(ground_truth)
    
    print(f"[TRAINING] Saved pair #{idx}:")
    print(f"  Image: {img_path}")
    print(f"  Truth: {gt_path}")
    print(f"  Text:  {ground_truth[:80]}...")
    
    return idx


def extract_text_lines(img: np.ndarray, ground_truth_lines: list):
    """
    Split an image into individual text lines and save each as a
    separate training pair. This creates better training data because
    Tesseract trains line by line.
    """
    import pytesseract
    from pytesseract import Output
    
    # Get bounding boxes for each line
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    
    # Group by line
    lines = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        line_num = data["line_num"][i]
        if line_num not in lines:
            lines[line_num] = {
                "boxes": [],
                "texts": []
            }
        lines[line_num]["boxes"].append((
            data["left"][i], 
            data["top"][i],
            data["width"][i], 
            data["height"][i]
        ))
        lines[line_num]["texts"].append(text)
    
    # Match OCR lines with ground truth lines
    gt_idx = 0
    for line_num in sorted(lines.keys()):
        if gt_idx >= len(ground_truth_lines):
            break
            
        line = lines[line_num]
        boxes = line["boxes"]
        
        if not boxes:
            continue
        
        # Compute bounding box for entire line
        x_min = min(b[0] for b in boxes)
        y_min = min(b[1] for b in boxes)
        x_max = max(b[0] + b[2] for b in boxes)
        y_max = max(b[1] + b[3] for b in boxes)
        
        # Add padding
        pad = 5
        y1 = max(0, y_min - pad)
        y2 = min(img.shape[0], y_max + pad)
        x1 = max(0, x_min - pad)
        x2 = min(img.shape[1], x_max + pad)
        
        line_img = img[y1:y2, x1:x2]
        
        if line_img.size > 0:
            save_training_pair(line_img, ground_truth_lines[gt_idx], "line")
            gt_idx += 1


def collect_from_image(image_path: str, ground_truth: str = None):
    """
    Process an image and save as training data.
    If ground_truth is None, runs OCR to get initial text (for manual correction).
    """
    import pytesseract
    
    print(f"\n{'='*50}")
    print(f"  Collecting training data from: {image_path}")
    print(f"{'='*50}")
    
    # Preprocess
    label_img = preprocess_for_ocr(image_path)
    gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY) if len(label_img.shape) == 3 else label_img
    
    if ground_truth is None:
        # Run OCR to get initial text
        text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
        print(f"\n  OCR output (edit this for ground truth):")
        print(f"  {'-'*40}")
        print(f"  {text}")
        print(f"  {'-'*40}")
        print(f"\n  Re-run with corrected text:")
        print(f'  python collect_training_data.py "{image_path}" "CORRECTED TEXT"')
        return
    
    # Save the full image pair
    save_training_pair(gray, ground_truth)
    
    # Also try to save line-by-line
    gt_lines = [line for line in ground_truth.split("\n") if line.strip()]
    if len(gt_lines) > 1:
        print(f"\n  Also extracting {len(gt_lines)} individual lines...")
        extract_text_lines(gray, gt_lines)


def collect_from_pdf(pdf_path: str, dpi: int = 300):
    """
    Extract pages from a PDF and save as training data.
    Ground truth must be provided separately.
    """
    print(f"\n{'='*50}")
    print(f"  Extracting pages from PDF: {pdf_path}")
    print(f"{'='*50}")
    
    pages = convert_from_path(pdf_path, dpi=dpi)
    
    for i, page in enumerate(pages):
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        idx = save_training_pair(img, f"[PAGE {i+1} — NEEDS GROUND TRUTH]", "pdf_page")
        print(f"  Page {i+1} saved. Edit the .gt.txt file with correct text.")


def show_stats():
    """Show training data collection statistics."""
    tif_files = list(TRAINING_DATA_DIR.glob("*.tif"))
    gt_files = list(TRAINING_DATA_DIR.glob("*.gt.txt"))
    
    print(f"\n{'='*50}")
    print(f"  TRAINING DATA STATS")
    print(f"{'='*50}")
    print(f"  Directory: {TRAINING_DATA_DIR}")
    print(f"  Image files (.tif): {len(tif_files)}")
    print(f"  Ground truth files: {len(gt_files)}")
    
    # Check which ones need ground truth
    needs_gt = 0
    for gt_file in gt_files:
        with open(gt_file, "r", encoding="utf-8") as f:
            content = f.read()
            if "NEEDS GROUND TRUTH" in content:
                needs_gt += 1
    
    if needs_gt > 0:
        print(f"  ⚠ Needs ground truth: {needs_gt}")
    
    print(f"\n  Ready for training: {len(gt_files) - needs_gt} pairs")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python collect_training_data.py <image_or_pdf>")
        print("  python collect_training_data.py <image> \"CORRECT TEXT\"")
        print("  python collect_training_data.py --stats")
        sys.exit(1)
    
    if sys.argv[1] == "--stats":
        show_stats()
    elif sys.argv[1].lower().endswith(".pdf"):
        collect_from_pdf(sys.argv[1])
    else:
        gt = sys.argv[2] if len(sys.argv) > 2 else None
        collect_from_image(sys.argv[1], gt)
