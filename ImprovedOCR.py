"""
Improved OCR Pipeline for Equipment Labels v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enhanced preprocessing + fully dynamic correction (zero hardcoded corrections).

Changes from v2:
  ✓ Deskewing (auto-detects and corrects rotation)
  ✓ Multi-scale OCR (tries 2x, 3x, 4x and picks best)
  ✓ Adaptive CLAHE (auto-tunes clipLimit based on histogram)
  ✓ Connected component filtering (removes noise blobs)
  ✓ Tesseract confidence data extraction
  ✓ Dynamic correction via ocr_corrector (SymSpell + self-learning)
  ✗ No static/hardcoded corrections — everything is dynamic
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import os
from pathlib import Path

from ocr_corrector import OCRCorrector, extract_word_confidences


# ── Configuration ──────────────────────────────────────────────────────────────
DEBUG = True                     # Save intermediate images for inspection
DEBUG_DIR = "debug"
IMAGE_PATH = "Rosemount.jpeg"
CUSTOM_TRAINEDDATA = "equipment"  # Use custom model if available


# ── Debug Utility ──────────────────────────────────────────────────────────────

def save_debug(name: str, img: np.ndarray):
    """Save an intermediate image to the debug folder."""
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}.png"), img)


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_image(image_path: str) -> np.ndarray:
    """Load image from path."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    """Upscale image using Lanczos interpolation."""
    upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return upscaled


def crop_label_region(img: np.ndarray) -> np.ndarray:
    """
    Detect and crop the rectangular metallic label from the image.
    Falls back to a manual crop if contour detection fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 150)
    save_debug("02a_edges", edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    save_debug("02b_dilated_edges", dilated)

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


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Auto-detect and correct text rotation using minAreaRect.
    Works well for labels that are slightly tilted.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only deskew if rotation is small (< 15°)
    if abs(angle) > 15 or abs(angle) < 0.5:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    save_debug("02d_deskewed", rotated)
    print(f"[INFO] Deskewed by {angle:.2f}°")
    return rotated


def mask_screws(gray_img: np.ndarray) -> np.ndarray:
    """
    Detect and mask circular features (screws) that confuse OCR.
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
            cv2.circle(masked, (c[0], c[1]), int(c[2] * 1.3), 255, -1)
        save_debug("03_masked_screws", masked)
        print(f"[INFO] Masked {len(circles[0])} circular features (screws/logos)")

    return masked


def adaptive_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE with auto-tuned clipLimit based on image histogram.
    Darker/low-contrast images get higher clipLimit.
    """
    # Analyze histogram to determine optimal clipLimit
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()

    # Compute contrast score (std of histogram)
    mean_val = np.mean(img)
    std_val = np.std(img)

    # Lower contrast → higher clipLimit
    if std_val < 30:
        clip_limit = 5.0
    elif std_val < 50:
        clip_limit = 3.5
    else:
        clip_limit = 2.5

    # Darker images need more enhancement
    if mean_val < 80:
        clip_limit += 1.0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    save_debug("04_adaptive_clahe", enhanced)
    print(f"[INFO] CLAHE clipLimit={clip_limit:.1f} (mean={mean_val:.0f}, std={std_val:.0f})")
    return enhanced


def denoise_bilateral(img: np.ndarray) -> np.ndarray:
    """Bilateral filter: smooths noise while preserving sharp text edges."""
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    save_debug("05a_bilateral", denoised)
    return denoised


def denoise_nlm(img: np.ndarray) -> np.ndarray:
    """Non-local means denoising."""
    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    save_debug("05b_nlm", denoised)
    return denoised


def threshold_otsu(img: np.ndarray) -> np.ndarray:
    """Global Otsu threshold."""
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def threshold_adaptive(img: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold — better for uneven lighting."""
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10
    )
    return thresh


def filter_connected_components(binary_img: np.ndarray) -> np.ndarray:
    """
    Remove noise blobs that are too small or too large to be characters.
    Keeps only connected components within a reasonable size range.
    """
    h, w = binary_img.shape
    total_area = h * w

    # Invert for connected component analysis (text should be white)
    inverted = cv2.bitwise_not(binary_img)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.ones_like(binary_img) * 255  # white background

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter: too small (noise) or too large (borders/artifacts)
        min_area = max(15, total_area * 0.00005)
        max_area = total_area * 0.15
        min_dim = 3
        max_aspect = 15.0

        if area < min_area or area > max_area:
            continue
        if comp_w < min_dim or comp_h < min_dim:
            continue
        aspect = max(comp_w, comp_h) / max(min(comp_w, comp_h), 1)
        if aspect > max_aspect:
            continue

        # Keep this component
        cleaned[labels == i] = 0  # black text

    save_debug("07_cc_filtered", cleaned)
    return cleaned


def morphological_cleanup(img: np.ndarray) -> np.ndarray:
    """Clean up thresholded image with morphological operations."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    save_debug("08_morphology", cleaned)
    return cleaned


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp mask to sharpen text edges."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened


def invert_if_needed(img: np.ndarray) -> np.ndarray:
    """
    Tesseract works best with dark text on white background.
    If mostly dark (white text on dark bg), invert it.
    """
    white_ratio = np.sum(img > 127) / img.size
    if white_ratio < 0.4:
        img = cv2.bitwise_not(img)
        save_debug("09_inverted", img)
        print(f"[INFO] Inverted image (white ratio was {white_ratio:.2f})")
    return img


# ══════════════════════════════════════════════════════════════════════════════
# OCR EXTRACTION WITH CONFIDENCE DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_tesseract_config() -> str:
    """Build Tesseract config, using custom traineddata if available."""
    # Check if custom trained model exists
    try:
        tesseract_dir = pytesseract.get_tesseract_version()
        # Try to find tessdata directory
        tessdata_paths = [
            Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
            Path(os.environ.get("TESSDATA_PREFIX", "")) / "tessdata" if os.environ.get("TESSDATA_PREFIX") else None,
        ]
        for td in tessdata_paths:
            if td and (td / f"{CUSTOM_TRAINEDDATA}.traineddata").exists():
                print(f"[INFO] Using custom trained model: {CUSTOM_TRAINEDDATA}")
                return f"-l {CUSTOM_TRAINEDDATA}"
    except Exception:
        pass

    return "-l eng"  # fallback to English


def run_ocr_with_confidence(img: np.ndarray, config: str) -> tuple:
    """
    Run Tesseract and return both text and word-level confidence data.

    Returns:
        (text, word_confidences) where word_confidences is {word: confidence}
    """
    # Get detailed data with confidence
    data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)

    # Extract text and confidence
    text_parts = []
    word_confidences = {}
    current_line = -1

    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue

        line_num = data["line_num"][i]
        if line_num != current_line and text_parts:
            text_parts.append("\n")
            current_line = line_num
        elif current_line == -1:
            current_line = line_num

        text_parts.append(word)
        text_parts.append(" ")

        # Store confidence (Tesseract gives 0-100)
        conf = data["conf"][i]
        try:
            conf_val = float(conf) / 100.0
        except (ValueError, TypeError):
            conf_val = 0.0

        word_confidences[word] = max(conf_val, 0.0)

    text = "".join(text_parts).strip()
    return text, word_confidences


def extract_text_multiscale(img_original: np.ndarray) -> tuple:
    """
    Multi-scale OCR: try multiple upscale factors and preprocessing
    combinations. Return the best result based on keyword scoring.

    This is the enhanced replacement for the old extract_text_best.
    """
    scales = [2, 3, 4]
    configs = {
        "PSM6": r"--oem 3 --psm 6",
        "PSM4": r"--oem 3 --psm 4",
        "PSM3": r"--oem 3 --psm 3",
    }

    # Dynamic keyword list — loaded from vocabulary, not hardcoded
    from ocr_corrector import OCRCorrector
    temp_corrector = OCRCorrector()
    # Use the top vocabulary terms as scoring keywords
    keywords = []
    
    # Extract brands and types specifically for scoring
    vocab_words = temp_corrector.sym_spell.words
    for word, count in vocab_words.items():
        if count >= 60:  # high-frequency terms
            keywords.append(word)
    del temp_corrector

    all_results = {}
    best_key = None
    best_score = 0
    best_text = ""
    best_confidences = {}

    for scale in scales:
        print(f"\n[INFO] Trying scale {scale}x...")
        scaled = upscale(img_original, scale)

        # Preprocess at this scale
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if len(scaled.shape) == 3 else scaled
        masked = mask_screws(gray)
        enhanced = adaptive_clahe(masked)

        # Create multiple preprocessed variants
        bilateral = denoise_bilateral(enhanced)
        nlm = denoise_nlm(enhanced)

        otsu_bil = threshold_otsu(bilateral)
        otsu_bil = invert_if_needed(otsu_bil)

        otsu_nlm = threshold_otsu(nlm)
        otsu_nlm = invert_if_needed(otsu_nlm)

        adaptive_thresh = threshold_adaptive(bilateral)
        adaptive_thresh = invert_if_needed(adaptive_thresh)

        gray_sharp = sharpen(bilateral)
        gray_sharp = invert_if_needed(gray_sharp)

        # Connected component filtering on best binary
        cc_filtered = filter_connected_components(otsu_bil)

        variants = {
            f"s{scale}_otsu_bil": otsu_bil,
            f"s{scale}_otsu_nlm": otsu_nlm,
            f"s{scale}_adaptive": adaptive_thresh,
            f"s{scale}_gray_sharp": gray_sharp,
            f"s{scale}_cc_filtered": cc_filtered,
        }

        for var_name, var_img in variants.items():
            cleaned = morphological_cleanup(var_img)

            for cfg_name, cfg in configs.items():
                key = f"{var_name}_{cfg_name}"
                try:
                    text, word_confs = run_ocr_with_confidence(cleaned, cfg)
                except Exception as e:
                    print(f"[WARNING] OCR failed for {key}: {e}")
                    continue

                all_results[key] = (text, word_confs)

                # Score: keyword matches + length + confidence
                score = sum(1 for kw in keywords if kw.lower() in text.lower())
                length_bonus = min(len(text) / 100, 3)
                avg_conf = (sum(word_confs.values()) / max(len(word_confs), 1)) if word_confs else 0
                conf_bonus = avg_conf * 2
                total = score + length_bonus + conf_bonus

                if total > best_score:
                    best_score = total
                    best_key = key
                    best_text = text
                    best_confidences = word_confs

    return best_text, best_key, best_score, best_confidences, all_results


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED DATA PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_equipment_data(raw_text: str, corrector=None) -> dict:
    """Parse raw OCR text into structured equipment data."""
    data = {}
    
    # Initialize corrector if not provided (fallback)
    if corrector is None:
        from ocr_corrector import OCRCorrector
        corrector = OCRCorrector()
    
    # Extract Brands from vocabulary
    brands = []
    with open(corrector.vocabulary_path, "r", encoding="utf-8") as f:
        found_brand_section = False
        for line in f:
            if "Brands" in line: found_brand_section = True
            elif line.startswith("# ──") and found_brand_section: break
            elif found_brand_section and not line.startswith("#") and line.strip():
                brands.append(line.split()[0].upper())
    
    brand_pattern = r'(' + '|'.join(brands) + r')'
    brand_match = re.search(brand_pattern, raw_text, re.IGNORECASE)
    if brand_match:
        data['brand'] = brand_match.group(1).upper()
    else:
        # Fallback to a broader list if vocab search fails
        fallback_brands = r'(ROSEMOUNT|EMERSON|ENDRESS|SIEMENS|YOKOGAWA|HONEYWELL|ABB|VEGA|KROHNE)'
        brand_match = re.search(fallback_brands, raw_text, re.IGNORECASE)
        if brand_match:
            data['brand'] = brand_match.group(1).upper()

    # Extract Measurement Types (Temperature, Pressure, etc.)
    measurements = ["TEMPERATURE", "PRESSURE", "FLOW", "LEVEL", "HUMIDITY", "DENSITY"]
    meas_pattern = r'(' + '|'.join(measurements) + r')'
    meas_match = re.search(meas_pattern, raw_text, re.IGNORECASE)
    
    # Extract Equipment Types (Transmitter, Sensor, etc.)
    types = ["TRANSMITTER", "SENSOR", "DETECTOR", "CONTROLLER", "INDICATOR"]
    type_pattern = r'(' + '|'.join(types) + r')'
    type_match = re.search(type_pattern, raw_text, re.IGNORECASE)
    
    if meas_match and type_match:
        data['type'] = f"{meas_match.group(1)} {type_match.group(2)}".upper()
    elif type_match:
        data['type'] = type_match.group(1).upper()

    # Protocol
    if re.search(r'HART', raw_text, re.IGNORECASE):
        data['protocol'] = 'HART'
    elif re.search(r'MODBUS', raw_text, re.IGNORECASE):
        data['protocol'] = 'MODBUS'

    # Model - Improved regex
    model_match = re.search(r'MODEL\s*[:\s]*([0-9A-Z]{3,}(?:[\s-][0-9A-Z]{2,})*)', raw_text, re.IGNORECASE)
    if model_match:
        data['model'] = model_match.group(1).strip()

    # Serial Number - Handle common OCR noise in SERIAL NO.
    serial_match = re.search(r'SER(?:IAL)?\s*(?:NO\.?|NUMBER)?\s*[:\s]*([0-9A-Z]{5,})', raw_text, re.IGNORECASE)
    if serial_match:
        data['serial_number'] = serial_match.group(1).strip()

    # Date (MM/YY or MM/YYYY)
    date_match = re.search(r'(\d{2}/\d{2}(?:\d{2})?)', raw_text)
    if date_match:
        data['date'] = date_match.group(1)

    # Factory Calibration
    cal_match = re.search(r'FACT(?:ORY)?\s*(?:CAL\.?|CALIBRATION)\s*[:\s]*(.*?)(?:\n|$)', raw_text, re.IGNORECASE)
    if cal_match:
        cal_val = cal_match.group(1).strip()
        data['factory_calibration'] = cal_val

    # Supply Voltage - Handle confusion like VDC vs ¥DC (though corrector should fix it now)
    supply_match = re.search(r'SUPPLY\s*[:\s]*([\d.\-]+\s*(?:TO\s*[\d.]+\s*)?V(?:DC)?)', raw_text, re.IGNORECASE)
    if supply_match:
        data['supply_voltage'] = supply_match.group(1).strip()

    # Output (4-20 mA)
    output_match = re.search(r'OUT(?:PUT)?\s*[:\s]*([\d.\-]+\s*m[Aa])', raw_text, re.IGNORECASE)
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

    # NEMA / CSA
    nema_match = re.search(r'NEMA\s*TYPE\s*(\w+)', raw_text, re.IGNORECASE)
    if nema_match: data['nema_type'] = nema_match.group(1)
    
    csa_match = re.search(r'CSA\s*(?:ENCL(?:\.|OSURE)?\s*TYPE)?\s*(\w+)', raw_text, re.IGNORECASE)
    if csa_match: data['csa_enclosure_type'] = csa_match.group(1)

    # HW / SW versions
    hw_match = re.search(r'HW\s*[:\s]*([\d.]+)', raw_text, re.IGNORECASE)
    if hw_match: data['hw_version'] = hw_match.group(1)
    
    sw_match = re.search(r'SW\s*[:\s]*([\d.]+)', raw_text, re.IGNORECASE)
    if sw_match: data['sw_version'] = sw_match.group(1)

    return data


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA COLLECTION (for future Tesseract fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

def collect_training_pair(img: np.ndarray, corrected_text: str, output_dir: str = "training/data"):
    """
    Save an image-text pair for future Tesseract fine-tuning.
    Accumulates data over time that can be used to train a custom model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Count existing pairs
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".gt.txt")])
    idx = existing + 1

    # Save image as TIFF (Tesseract training format)
    img_path = os.path.join(output_dir, f"equipment_{idx:04d}.tif")
    gt_path = os.path.join(output_dir, f"equipment_{idx:04d}.gt.txt")

    cv2.imwrite(img_path, img)
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    print(f"[TRAINING] Saved training pair #{idx}: {img_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline. Returns the cropped and deskewed
    label image ready for multi-scale OCR.
    """
    print(f"\n{'#'*60}")
    print(f"  Processing: {image_path}")
    print(f"{'#'*60}")

    # Load original
    img = load_image(image_path)
    save_debug("01_original", img)

    # Crop to label region
    label = crop_label_region(img)

    # Deskew
    label = deskew(label)

    return label


def process_single_image(image_path: str, corrector: OCRCorrector):
    """Run the full pipeline on a single image and return results."""
    try:
        # Step 1: Preprocess
        label_img = preprocess_for_ocr(image_path)

        # Step 2: Multi-scale OCR
        best_text, best_key, best_score, best_confidences, all_results = \
            extract_text_multiscale(label_img)

        # Step 3: Dynamic correction
        corrected = corrector.correct(best_text, best_confidences)

        # Step 4: Parse structured data (Combined approach)
        scored = []
        for key, (text, confs) in all_results.items():
            avg_conf = (sum(confs.values()) / max(len(confs), 1)) if confs else 0
            score = len(text) / 50 + avg_conf * 3
            scored.append((key, text, confs, score))
        scored.sort(key=lambda x: x[3], reverse=True)

        combined = {}
        for key, text, confs, score in scored[:3]:
            # Apply correction to each variant before parsing
            corrected_variant = corrector.correct(text, confs)
            partial = parse_equipment_data(corrected_variant, corrector=corrector)
            for field, value in partial.items():
                if field not in combined or len(str(value)) > len(str(combined[field])):
                    combined[field] = value

        # Step 5: Self-learning
        if best_text != corrected:
            corrector.learn_from_comparison(best_text, corrected)

        # Step 6: Collect training data
        final_text = corrected if corrected else best_text
        gray_label = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY) if len(label_img.shape) == 3 else label_img
        collect_training_pair(gray_label, final_text)

        return {
            "raw_text": best_text,
            "corrected_text": corrected,
            "parsed_data": combined,
            "config_used": best_key
        }
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Improved OCR Pipeline for Equipment Labels")
    parser.add_argument("input", nargs="?", default=IMAGE_PATH, help="Path to image, directory, or PDF")
    parser.add_argument("--batch", action="store_true", help="Process all images in a directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  IMPROVED OCR FOR EQUIPMENT LABELS v3")
    print("  (Dynamic Correction + Robust Parsing)")
    print("=" * 60)

    # Initialize dynamic corrector
    corrector = OCRCorrector()
    print(f"\n[CORRECTOR] Stats: {corrector.get_stats()}")

    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Batch mode
        files = []
        if input_path.is_dir():
            files = [f for f in input_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]]
        
        if not files:
            print(f"[ERROR] No images found in {input_path}")
            return

        print(f"\n[BATCH] Processing {len(files)} files...")
        results = {}
        for f in files:
            res = process_single_image(str(f), corrector)
            if res:
                results[f.name] = res
        
        # Save results to JSON
        import json
        output_file = "ocr_results_batch.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\n[BATCH] Completed. Results saved to {output_file}")

    elif input_path.suffix.lower() == ".pdf":
        # PDF mode
        from pdf2image import convert_from_path
        print(f"\n[PDF] Converting {input_path} to images...")
        pages = convert_from_path(str(input_path), dpi=300)
        
        pdf_results = []
        for i, page in enumerate(pages):
            page_img = np.array(page)
            page_img = cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR)
            
            # Temporary save for processing
            temp_img_path = f"temp_page_{i+1}.png"
            cv2.imwrite(temp_img_path, page_img)
            
            print(f"\n[PDF] Processing page {i+1}...")
            res = process_single_image(temp_img_path, corrector)
            if res:
                pdf_results.append(res)
            
            os.remove(temp_img_path)

        # Print summary for PDF
        for i, res in enumerate(pdf_results):
            print(f"\n--- PAGE {i+1} PARSED DATA ---")
            for k, v in res["parsed_data"].items():
                print(f"  {k:20s}: {v}")

    else:
        # Single image mode
        res = process_single_image(str(input_path), corrector)
        if res:
            print("\n" + "=" * 60)
            print("  CORRECTED TEXT")
            print("=" * 60)
            print(res["corrected_text"])

            print("\n" + "=" * 60)
            print("  PARSED EQUIPMENT DATA")
            print("=" * 60)
            for k, v in res["parsed_data"].items():
                print(f"  {k:25s}: {v}")
            
            print(f"\n[CORRECTOR] Updated stats: {corrector.get_stats()}")
        else:
            print("[ERROR] Processing failed.")

    if DEBUG:
        print(f"\n[DEBUG] Intermediate images saved to '{DEBUG_DIR}/' folder")


if __name__ == "__main__":
    main()
