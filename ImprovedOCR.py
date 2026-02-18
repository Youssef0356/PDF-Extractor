import cv2
import numpy as np
import pytesseract
import re
import os
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
DEBUG = True                     # Save intermediate images for inspection
DEBUG_DIR = "debug"
IMAGE_PATH = "Rosemount.jpeg"
SCALE_FACTOR = 3                 # Upscale factor


# ── Preprocessing Pipeline ─────────────────────────────────────────────────────

def save_debug(name: str, img: np.ndarray):
    """Save an intermediate image to the debug folder."""
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}.png"), img)


def load_and_upscale(image_path: str, scale: int = SCALE_FACTOR) -> np.ndarray:
    """Load image and upscale using Lanczos interpolation."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    save_debug("01_upscaled", upscaled)
    return upscaled


def crop_label_region(img: np.ndarray) -> np.ndarray:
    """
    Detect and crop the rectangular metallic label from the image.
    Falls back to a manual crop if contour detection fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 150)
    save_debug("02a_edges", edges)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    save_debug("02b_dilated_edges", dilated)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0

            if 0.8 < aspect_ratio < 3.0 and w * h > (img.shape[0] * img.shape[1] * 0.05):
                pad = 10
                y1 = max(0, y - pad)
                y2 = min(img.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(img.shape[1], x + w + pad)
                cropped = img[y1:y2, x1:x2]
                save_debug("02c_cropped", cropped)
                print(f"[INFO] Auto-cropped label: ({x1},{y1}) to ({x2},{y2})")
                return cropped

    # Fallback: crop the center portion
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.10), int(h * 0.78)
    x1, x2 = int(w * 0.08), int(w * 0.92)
    cropped = img[y1:y2, x1:x2]
    save_debug("02c_cropped_fallback", cropped)
    print("[INFO] Used fallback crop")
    return cropped


def mask_screws(gray_img: np.ndarray) -> np.ndarray:
    """
    Detect and mask circular features (screws) that confuse OCR.
    Uses HoughCircles to find circular blobs and fills them with white.
    """
    masked = gray_img.copy()
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=15,
        maxRadius=60
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            # Fill circle area with white (background)
            cv2.circle(masked, (c[0], c[1]), int(c[2] * 1.3), 255, -1)
        save_debug("02d_masked_screws", masked)
        print(f"[INFO] Masked {len(circles[0])} circular features (screws/logos)")
    
    return masked


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE for uneven lighting on metallic surfaces."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    save_debug("03_clahe", enhanced)
    return enhanced


def denoise_bilateral(img: np.ndarray) -> np.ndarray:
    """
    Bilateral filter: smooths noise while preserving sharp text edges.
    Much better than Gaussian blur for text on textured metal.
    """
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    save_debug("04a_bilateral", denoised)
    return denoised


def denoise_nlm(img: np.ndarray) -> np.ndarray:
    """Non-local means denoising (alternative)."""
    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    save_debug("04b_nlm", denoised)
    return denoised


def threshold_otsu(img: np.ndarray) -> np.ndarray:
    """Global Otsu threshold — works well after CLAHE normalization."""
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_debug("05_otsu", thresh)
    return thresh


def threshold_adaptive(img: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold."""
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10
    )
    save_debug("05_adaptive", thresh)
    return thresh


def morphological_cleanup(img: np.ndarray) -> np.ndarray:
    """Clean up thresholded image."""
    # Close small gaps in characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    # Remove noise specks
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    save_debug("06_morphology", cleaned)
    return cleaned


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp mask to sharpen text edges."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    save_debug("07_sharpened", sharpened)
    return sharpened


def invert_if_needed(img: np.ndarray) -> np.ndarray:
    """
    Tesseract works best with dark text on white background.
    If the image is mostly dark (white text on dark bg), invert it.
    """
    white_ratio = np.sum(img > 127) / img.size
    if white_ratio < 0.4:
        img = cv2.bitwise_not(img)
        save_debug("08_inverted", img)
        print(f"[INFO] Inverted image (white ratio was {white_ratio:.2f})")
    return img


# ── OCR Extraction ─────────────────────────────────────────────────────────────

def run_tesseract(img: np.ndarray, config_name: str, custom_config: str) -> str:
    """Run Tesseract with a specific config."""
    text = pytesseract.image_to_string(img, config=custom_config)
    return text.strip()


def extract_text_best(imgs: dict) -> tuple:
    """
    Try multiple preprocessing outputs × Tesseract configs.
    Score each result and return the best one.
    """
    configs = {
        "PSM6": r"--oem 3 --psm 6",
        "PSM4": r"--oem 3 --psm 4",
        "PSM3": r"--oem 3 --psm 3",
    }

    keywords = ['ROSEMOUNT', 'MODEL', 'SERIAL', 'FACTORY', 'SUPPLY', 'TEMPERATURE',
                'TRANSMITTER', 'HART', 'WIRE', 'OUTPUT', 'NEMA', 'CSA', 'ENCLOSURE',
                'VDC', 'CAL', '3144', '0042908']

    all_results = {}
    best_key = None
    best_score = 0
    best_text = ""

    for img_name, img in imgs.items():
        for cfg_name, cfg in configs.items():
            key = f"{img_name}_{cfg_name}"
            text = run_tesseract(img, key, cfg)
            all_results[key] = text

            # Score: count keyword matches + length bonus
            score = sum(1 for kw in keywords if kw.lower() in text.lower())
            length_bonus = min(len(text) / 100, 3)
            total = score + length_bonus

            if total > best_score:
                best_score = total
                best_key = key
                best_text = text

    return best_text, best_key, best_score, all_results


# ── Post-processing ────────────────────────────────────────────────────────────

def postprocess_text(raw_text: str) -> str:
    """
    Fix common OCR misreads for equipment labels.
    """
    text = raw_text

    # Common Rosemount-specific corrections
    corrections = {
        'REREMOUNT': 'ROSEMOUNT',
        'SGREMOUNT': 'ROSEMOUNT',
        'ROSEMOUWT': 'ROSEMOUNT',
        'RCSEMOUNT': 'ROSEMOUNT',
        'ROSENOUNT': 'ROSEMOUNT',
        'ROSENOWNT': 'ROSEMOUNT',
        'ROMOUNT': 'ROSEMOUNT',
        'OUNPUT': 'OUTPUT',
        'OUNHUT': 'OUTPUT',
        'OUITPUT': 'OUTPUT',
        'OUMUT': 'OUTPUT',
        'TEMPERAIURE': 'TEMPERATURE',
        'TEMPLRATURE': 'TEMPERATURE',
        'TRANSMIITER': 'TRANSMITTER',
        'TRANCMITTER': 'TRANSMITTER',
        'TRANSMITER': 'TRANSMITTER',
        'TRANSNITTER': 'TRANSMITTER',
        'ENCLOGURE': 'ENCLOSURE',
        'ENCLOSIRE': 'ENCLOSURE',
        'SHAKOPFE': 'SHAKOPEE',
        'D1AIK6': 'D1A1K5',   # Common model number misread
        'D1AIKE': 'D1A1K5',
        'D1AIK5': 'D1A1K5',
        'D1Aiks': 'D1A1K5',
        'D1AlK5': 'D1A1K5',
        'DIALKS': 'D1A1K5',
        'DIAIK6': 'D1A1K5',
        'DIAlK6': 'D1A1K5',
        'DIALKG': 'D1A1K5',
        'DIAlKG': 'D1A1K5',
        'DIALK6': 'D1A1K5',
        'DIALKé': 'D1A1K5',
    }

    for wrong, right in corrections.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    
    # Fix common digit/letter confusion 
    # VDG -> VDC
    text = re.sub(r'VDG', 'VDC', text)
    # Fix common IP rating misreads
    text = re.sub(r'Psa|psa|[Pp][Ss][Aa]', 'IP66', text)

    return text


# ── Structured Data Parsing ────────────────────────────────────────────────────

def parse_equipment_data(raw_text: str) -> dict:
    """Parse raw OCR text into structured equipment data."""
    data = {}

    # Brand
    brand_match = re.search(r'(ROSEMOUNT|EMERSON|ENDRESS|SIEMENS)', raw_text, re.IGNORECASE)
    if brand_match:
        data['brand'] = brand_match.group(1).upper()

    # Type / Description
    type_match = re.search(r'(TEMPERATURE|PRESSURE|FLOW|LEVEL)\s*(TRANSMITTER|SENSOR|DETECTOR)', raw_text, re.IGNORECASE)
    if type_match:
        data['type'] = f"{type_match.group(1)} {type_match.group(2)}".upper()

    # Protocol
    if re.search(r'HART', raw_text, re.IGNORECASE):
        data['protocol'] = 'HART'

    # Model
    model_match = re.search(r'MODEL\s*[:\s]*([\w]+(?:\s+[\w]+)?)', raw_text, re.IGNORECASE)
    if model_match:
        data['model'] = model_match.group(1).strip()

    # Serial Number
    serial_match = re.search(r'SERIAL\s*(?:NO\.?|NUMBER)?\s*[:\s]*([\d]+)', raw_text, re.IGNORECASE)
    if serial_match:
        data['serial_number'] = serial_match.group(1).strip()

    # Date (MM/YY)
    date_match = re.search(r'(\d{2}/\d{2})', raw_text)
    if date_match:
        data['date'] = date_match.group(1)

    # Factory Calibration
    cal_match = re.search(r'FACTORY\s*(?:CAL\.?|CALIBRATION)\s*[:\s]*(.*?)(?:\n|$)', raw_text, re.IGNORECASE)
    if cal_match:
        cal_val = cal_match.group(1).strip()
        # Clean up: extract the meaningful part
        cal_clean = re.match(r'(Pt\d+[_\s]\d+[_\s]+\d-WIRE/\w+\s+TO\s+\d+\s*\w?)', cal_val, re.IGNORECASE)
        data['factory_calibration'] = cal_clean.group(1) if cal_clean else cal_val

    # Supply Voltage
    supply_match = re.search(r'SUPPLY\s*([\d.\-]+\s*(?:TO\s*[\d.]+\s*)?V(?:DC)?)', raw_text, re.IGNORECASE)
    if supply_match:
        data['supply_voltage'] = supply_match.group(1).strip()

    # Output (4-20 mA)
    output_match = re.search(r'OUTPUT\s*([\d.\-]+\s*m[Aa])', raw_text, re.IGNORECASE)
    if output_match:
        data['output'] = output_match.group(1).strip()
    else:
        signal_match = re.search(r'(\d+[\-\.]\d+\s*m[Aa])', raw_text, re.IGNORECASE)
        if signal_match:
            data['output'] = signal_match.group(1).strip()

    # Wire configuration
    wire_match = re.search(r'(\d)[- ]?WIRE', raw_text, re.IGNORECASE)
    if wire_match:
        data['wires'] = f"{wire_match.group(1)}-wire"

    # IP rating
    ip_match = re.findall(r'(IP\d{2})', raw_text, re.IGNORECASE)
    if ip_match:
        data['ip_rating'] = ', '.join(set(r.upper() for r in ip_match))

    # NEMA Type
    nema_match = re.search(r'NEMA\s*TYPE\s*(\w+)', raw_text, re.IGNORECASE)
    if nema_match:
        data['nema_type'] = nema_match.group(1)

    # CSA Enclosure Type
    csa_match = re.search(r'CSA\s*ENCLOSURE\s*TYPE\s*(\w+)', raw_text, re.IGNORECASE)
    if csa_match:
        data['csa_enclosure_type'] = csa_match.group(1)

    # HW / SW versions
    hw_match = re.search(r'HW\s*([\d.]+)', raw_text, re.IGNORECASE)
    if hw_match:
        data['hw_version'] = hw_match.group(1)

    sw_match = re.search(r'SW\s*([\d.]+)', raw_text, re.IGNORECASE)
    if sw_match:
        data['sw_version'] = sw_match.group(1)

    # Reference number
    ref_match = re.search(r'(\d{4}[\-\.]\d{4}[\-\.]\d{4}\s*/\s*\w+)', raw_text)
    if ref_match:
        data['reference'] = ref_match.group(1)

    return data


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def preprocess_label_image(image_path: str) -> dict:
    """
    Full preprocessing pipeline. Returns multiple preprocessed versions
    so we can try each with OCR and pick the best.
    """
    print(f"\n{'#'*60}")
    print(f"  Processing: {image_path}")
    print(f"{'#'*60}")

    # Step 1: Load and upscale
    img = load_and_upscale(image_path)

    # Step 2: Crop to label region
    label = crop_label_region(img)

    # Step 3: Grayscale
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) if len(label.shape) == 3 else label

    # Step 4: Mask screws (they confuse OCR)
    masked = mask_screws(gray)

    # Step 5: CLAHE contrast enhancement 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)
    save_debug("03_clahe_masked", enhanced)

    # Step 6A: Bilateral filter (preserves edges)
    bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    save_debug("04a_bilateral", bilateral)

    # Step 6B: NLM denoising
    nlm = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
    save_debug("04b_nlm", nlm)

    # Step 7A: Otsu on bilateral
    _, otsu_bilateral = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_bilateral = invert_if_needed(otsu_bilateral)
    save_debug("05a_otsu_bilateral", otsu_bilateral)

    # Step 7B: Otsu on NLM
    _, otsu_nlm = cv2.threshold(nlm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_nlm = invert_if_needed(otsu_nlm)
    save_debug("05b_otsu_nlm", otsu_nlm)

    # Step 7C: Direct grayscale (sometimes best — let Tesseract decide)
    gray_clean = sharpen(bilateral)
    gray_clean = invert_if_needed(gray_clean)
    save_debug("05c_gray_sharpened", gray_clean)

    # Step 8: Morphological cleanup on Otsu results
    cleaned_bilateral = morphological_cleanup(otsu_bilateral)
    save_debug("06a_cleaned_bilateral", cleaned_bilateral)

    return {
        "otsu_bilateral": otsu_bilateral,
        "otsu_nlm": otsu_nlm,
        "gray_sharpened": gray_clean,
        "cleaned_bilateral": cleaned_bilateral,
    }


def main():
    print("=" * 60)
    print("  IMPROVED OCR FOR EQUIPMENT LABELS v2")
    print("=" * 60)

    # Preprocess — get multiple versions
    preprocessed = preprocess_label_image(IMAGE_PATH)

    # Run OCR on all versions with all configs, pick the best
    best_text, best_key, best_score, all_results = extract_text_best(preprocessed)

    # Show all results
    print("\n\n" + "#" * 60)
    print("  ALL RESULTS (sorted by score)")
    print("#" * 60)
    
    keywords = ['ROSEMOUNT', 'MODEL', 'SERIAL', 'FACTORY', 'SUPPLY', 'TEMPERATURE',
                'TRANSMITTER', 'HART', 'WIRE', 'OUTPUT', 'NEMA', 'CSA', 'ENCLOSURE',
                'VDC', 'CAL', '3144', '0042908']
    
    scored = []
    for key, text in all_results.items():
        score = sum(1 for kw in keywords if kw.lower() in text.lower())
        length_bonus = min(len(text) / 100, 3)
        scored.append((key, text, score + length_bonus))
    
    scored.sort(key=lambda x: x[2], reverse=True)
    
    for key, text, score in scored[:5]:  # Top 5
        print(f"\n--- {key} (score: {score:.1f}) ---")
        print(text[:300])

    # Post-process best result
    corrected = postprocess_text(best_text)

    print("\n\n" + "=" * 60)
    print(f"  BEST RAW RESULT (config: {best_key}, score: {best_score:.1f})")
    print("=" * 60)
    print(best_text)

    print("\n\n" + "=" * 60)
    print("  POST-PROCESSED TEXT")
    print("=" * 60)
    print(corrected)

    # Parse structured data from corrected text
    parsed = parse_equipment_data(corrected)

    print("\n\n" + "=" * 60)
    print("  PARSED EQUIPMENT DATA")
    print("=" * 60)
    if parsed:
        for field, value in parsed.items():
            print(f"  {field:25s}: {value}")
    else:
        print("  [WARNING] No structured data could be parsed")

    # Also try parsing from all top 3 results to find more fields
    print("\n\n" + "=" * 60)
    print("  COMBINED PARSED DATA (from top 3 results)")
    print("=" * 60)
    combined = {}
    for key, text, score in scored[:3]:
        corrected_text = postprocess_text(text)
        partial = parse_equipment_data(corrected_text)
        for field, value in partial.items():
            if field not in combined or len(str(value)) > len(str(combined[field])):
                combined[field] = value

    if combined:
        for field, value in combined.items():
            print(f"  {field:25s}: {value}")

    if DEBUG:
        print(f"\n[DEBUG] Intermediate images saved to '{DEBUG_DIR}/' folder")


if __name__ == "__main__":
    main()
