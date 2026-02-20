"""
Tesseract LSTM Fine-Tuning Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fine-tunes Tesseract's LSTM engine on your collected equipment label data.
Creates a custom 'equipment.traineddata' model for better domain accuracy.

Prerequisites:
    - Tesseract 5.x installed with training tools
    - Collected training data (run collect_training_data.py first)
    - Base eng.traineddata from tessdata_best

Usage:
    python fine_tune_tesseract.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────
TRAINING_DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
MODEL_NAME = "equipment"
BASE_LANG = "eng"
MAX_ITERATIONS = 400

# Common Tesseract installation paths on Windows
TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR"),
    Path(os.environ.get("TESSERACT_PATH", ".")),
]

TESSDATA_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
    Path(os.environ.get("TESSDATA_PREFIX", ".")) / "tessdata" if os.environ.get("TESSDATA_PREFIX") else None,
]


def find_tesseract_dir() -> Path:
    """Find the Tesseract installation directory."""
    for p in TESSERACT_PATHS:
        if p and p.exists() and (p / "tesseract.exe").exists():
            return p
    
    # Try from PATH
    result = shutil.which("tesseract")
    if result:
        return Path(result).parent
    
    raise FileNotFoundError(
        "Could not find Tesseract installation. "
        "Set TESSERACT_PATH environment variable or install Tesseract."
    )


def find_tessdata_dir() -> Path:
    """Find the tessdata directory."""
    for p in TESSDATA_PATHS:
        if p and p.exists():
            return p
    
    tesseract_dir = find_tesseract_dir()
    tessdata = tesseract_dir / "tessdata"
    if tessdata.exists():
        return tessdata
    
    raise FileNotFoundError("Could not find tessdata directory.")


def check_prerequisites():
    """Verify all requirements are met before training."""
    print("=" * 50)
    print("  CHECKING PREREQUISITES")
    print("=" * 50)
    
    # 1. Check Tesseract
    tess_dir = find_tesseract_dir()
    print(f"  ✓ Tesseract found: {tess_dir}")
    
    # 2. Check tessdata
    tessdata = find_tessdata_dir()
    print(f"  ✓ tessdata found: {tessdata}")
    
    # 3. Check base model
    base_model = tessdata / f"{BASE_LANG}.traineddata"
    if not base_model.exists():
        print(f"  ✗ Base model not found: {base_model}")
        print(f"    Download from: https://github.com/tesseract-ocr/tessdata_best")
        return False
    print(f"  ✓ Base model: {base_model}")
    
    # 4. Check training tools
    tools = ["combine_tessdata", "lstmtraining", "text2image"]
    missing_tools = []
    for tool in tools:
        tool_path = tess_dir / f"{tool}.exe"
        if not tool_path.exists():
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"  ✗ Missing training tools: {missing_tools}")
        print(f"    Install Tesseract with training tools enabled")
        return False
    print(f"  ✓ Training tools available")
    
    # 5. Check training data
    tif_files = list(TRAINING_DATA_DIR.glob("*.tif"))
    gt_files = list(TRAINING_DATA_DIR.glob("*.gt.txt"))
    
    valid_pairs = 0
    for gt_file in gt_files:
        tif_file = gt_file.with_suffix(".tif")
        if tif_file.exists():
            with open(gt_file, "r", encoding="utf-8") as f:
                content = f.read()
                if "NEEDS GROUND TRUTH" not in content:
                    valid_pairs += 1
    
    if valid_pairs < 5:
        print(f"  ⚠ Only {valid_pairs} valid training pairs (recommend >10)")
        print(f"    Run: python collect_training_data.py <image> \"CORRECT TEXT\"")
        if valid_pairs == 0:
            return False
    else:
        print(f"  ✓ Training data: {valid_pairs} pairs")
    
    return True


def generate_lstmf_files():
    """Generate .lstmf files from .tif + .gt.txt pairs."""
    print("\n" + "=" * 50)
    print("  GENERATING LSTMF FILES")
    print("=" * 50)
    
    tess_dir = find_tesseract_dir()
    tessdata = find_tessdata_dir()
    lstmf_dir = OUTPUT_DIR / "lstmf"
    lstmf_dir.mkdir(parents=True, exist_ok=True)
    
    gt_files = sorted(TRAINING_DATA_DIR.glob("*.gt.txt"))
    generated = 0
    
    for gt_file in gt_files:
        tif_file = gt_file.with_suffix(".tif")
        if not tif_file.exists():
            continue
        
        with open(gt_file, "r", encoding="utf-8") as f:
            content = f.read()
            if "NEEDS GROUND TRUTH" in content:
                continue
        
        lstmf_file = lstmf_dir / gt_file.with_suffix(".lstmf").name
        
        cmd = [
            str(tess_dir / "tesseract.exe"),
            str(tif_file),
            str(lstmf_file.with_suffix("")),  # tesseract adds the extension
            "--psm", "6",
            "lstm.train"
        ]
        
        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = str(tessdata.parent)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                generated += 1
                print(f"  ✓ {gt_file.stem}")
            else:
                print(f"  ✗ {gt_file.stem}: {result.stderr.strip()}")
        except Exception as e:
            print(f"  ✗ {gt_file.stem}: {e}")
    
    print(f"\n  Generated {generated} LSTMF files")
    return generated


def extract_lstm_from_base():
    """Extract LSTM component from base traineddata."""
    print("\n" + "=" * 50)
    print("  EXTRACTING BASE LSTM MODEL")
    print("=" * 50)
    
    tess_dir = find_tesseract_dir()
    tessdata = find_tessdata_dir()
    
    base_model = tessdata / f"{BASE_LANG}.traineddata"
    lstm_dir = OUTPUT_DIR / "lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    
    lstm_model = lstm_dir / f"{BASE_LANG}.lstm"
    
    cmd = [
        str(tess_dir / "combine_tessdata.exe"),
        "-e", str(base_model),
        str(lstm_model)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Extracted LSTM to: {lstm_model}")
        return lstm_model
    else:
        print(f"  ✗ Extraction failed: {result.stderr}")
        return None


def run_training():
    """Run LSTM fine-tuning."""
    print("\n" + "=" * 50)
    print(f"  FINE-TUNING TESSERACT ({MAX_ITERATIONS} iterations)")
    print("=" * 50)
    
    tess_dir = find_tesseract_dir()
    tessdata = find_tessdata_dir()
    
    lstmf_dir = OUTPUT_DIR / "lstmf"
    lstm_dir = OUTPUT_DIR / "lstm"
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training file list
    train_list = OUTPUT_DIR / "training_files.txt"
    lstmf_files = sorted(lstmf_dir.glob("*.lstmf"))
    
    with open(train_list, "w") as f:
        for lstmf in lstmf_files:
            f.write(str(lstmf) + "\n")
    
    base_lstm = lstm_dir / f"{BASE_LANG}.lstm"
    
    cmd = [
        str(tess_dir / "lstmtraining.exe"),
        "--model_output", str(checkpoint_dir / MODEL_NAME),
        "--continue_from", str(base_lstm),
        "--traineddata", str(tessdata / f"{BASE_LANG}.traineddata"),
        "--train_listfile", str(train_list),
        "--max_iterations", str(MAX_ITERATIONS),
        "--target_error_rate", "0.01",
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Training...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Training complete!")
        print(f"  Checkpoints saved to: {checkpoint_dir}")
    else:
        print(f"  ✗ Training failed:")
        print(f"    {result.stderr}")
    
    return result.returncode == 0


def create_traineddata():
    """Combine the trained model into a .traineddata file."""
    print("\n" + "=" * 50)
    print("  CREATING TRAINEDDATA")
    print("=" * 50)
    
    tess_dir = find_tesseract_dir()
    tessdata = find_tessdata_dir()
    
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    
    # Find the best checkpoint
    checkpoints = sorted(checkpoint_dir.glob(f"{MODEL_NAME}_checkpoint"))
    if not checkpoints:
        checkpoints = sorted(checkpoint_dir.glob(f"{MODEL_NAME}*.checkpoint"))
    
    if not checkpoints:
        print("  ✗ No checkpoints found")
        return False
    
    best_checkpoint = checkpoints[-1]
    
    cmd = [
        str(tess_dir / "lstmtraining.exe"),
        "--stop_training",
        "--continue_from", str(best_checkpoint),
        "--traineddata", str(tessdata / f"{BASE_LANG}.traineddata"),
        "--model_output", str(OUTPUT_DIR / f"{MODEL_NAME}.traineddata"),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        output_model = OUTPUT_DIR / f"{MODEL_NAME}.traineddata"
        print(f"  ✓ Created: {output_model}")
        
        # Copy to tessdata for immediate use
        dest = tessdata / f"{MODEL_NAME}.traineddata"
        try:
            shutil.copy2(output_model, dest)
            print(f"  ✓ Installed to: {dest}")
            print(f"  You can now use: pytesseract.image_to_string(img, lang='{MODEL_NAME}')")
        except PermissionError:
            print(f"  ⚠ Could not copy to {dest} (run as admin)")
            print(f"  Manually copy {output_model} to {dest}")
        
        return True
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return False


def main():
    """Full fine-tuning pipeline."""
    print("=" * 60)
    print("  TESSERACT LSTM FINE-TUNING FOR EQUIPMENT LABELS")
    print("=" * 60)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n  ✗ Prerequisites not met. Fix the issues above and try again.")
        sys.exit(1)
    
    # Step 2: Generate LSTMF training files
    num_lstmf = generate_lstmf_files()
    if num_lstmf == 0:
        print("\n  ✗ No training files generated. Collect more training data first.")
        sys.exit(1)
    
    # Step 3: Extract LSTM from base model
    lstm_model = extract_lstm_from_base()
    if not lstm_model:
        sys.exit(1)
    
    # Step 4: Run fine-tuning
    if not run_training():
        sys.exit(1)
    
    # Step 5: Create final traineddata
    if not create_traineddata():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  ✓ FINE-TUNING COMPLETE!")
    print("=" * 60)
    print(f"  Custom model '{MODEL_NAME}' is ready to use.")
    print(f"  ImprovedOCR.py will automatically detect and use it.")


if __name__ == "__main__":
    main()
