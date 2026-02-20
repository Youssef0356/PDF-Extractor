"""
Dynamic OCR Corrector — Self-learning, zero hardcoded corrections.

Three correction layers:
  1. Domain Dictionary  → SymSpell fuzzy matching against equipment vocabulary
  2. Confidence-guided  → Only correct words whose OCR confidence is low
  3. Learned Corrections → Auto-applies past verified corrections

All corrections come from data files, nothing is hardcoded.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from symspellpy import SymSpell, Verbosity
from rapidfuzz import fuzz

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ocr_data"
VOCABULARY_FILE = DATA_DIR / "domain_vocabulary.txt"
LEARNED_FILE = DATA_DIR / "learned_corrections.json"


class OCRCorrector:
    """
    Dynamic, self-learning OCR text corrector.

    Usage:
        corrector = OCRCorrector()
        corrected = corrector.correct(raw_text)
        corrector.learn("REREMOUNT", "ROSEMOUNT")  # teach it a new correction
    """

    def __init__(
        self,
        vocabulary_path: str = str(VOCABULARY_FILE),
        learned_path: str = str(LEARNED_FILE),
        max_edit_distance: int = 2,
        min_confidence: float = 0.60,
        auto_learn_threshold: float = 0.85,
    ):
        self.vocabulary_path = Path(vocabulary_path)
        self.learned_path = Path(learned_path)
        self.max_edit_distance = max_edit_distance
        self.min_confidence = min_confidence  # below this → correction is applied
        self.auto_learn_threshold = auto_learn_threshold  # above this → auto-learns

        # ── Layer 1: SymSpell Domain Dictionary ────────────────────────────
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=7,
        )
        self._load_domain_vocabulary()

        # ── Layer 3: Learned Corrections Database ──────────────────────────
        self.learned_data = self._load_learned_corrections()

    # ══════════════════════════════════════════════════════════════════════════
    # Loading
    # ══════════════════════════════════════════════════════════════════════════

    def _load_domain_vocabulary(self):
        """Load domain vocabulary into SymSpell from the frequency file."""
        if not self.vocabulary_path.exists():
            print(f"[WARNING] Vocabulary file not found: {self.vocabulary_path}")
            return

        # Parse custom format: WORD FREQUENCY (skip comments and blanks)
        temp_path = self.vocabulary_path.parent / "_symspell_temp.txt"
        with open(self.vocabulary_path, "r", encoding="utf-8") as fin, \
             open(temp_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    freq = parts[1]
                    fout.write(f"{word} {freq}\n")

        self.sym_spell.load_dictionary(
            str(temp_path),
            term_index=0,
            count_index=1,
        )
        temp_path.unlink(missing_ok=True)

        # Also load any vocabulary additions from learned data
        if hasattr(self, "learned_data"):
            for word in self.learned_data.get("vocabulary_additions", []):
                self.sym_spell.create_dictionary_entry(word, 50)

        count = self.sym_spell.word_count
        print(f"[OCR Corrector] Loaded {count} domain vocabulary terms")

    def _load_learned_corrections(self) -> dict:
        """Load previously learned corrections from JSON database."""
        if not self.learned_path.exists():
            return {"corrections": {}, "vocabulary_additions": [], "stats": {
                "total_documents_processed": 0,
                "total_corrections_made": 0,
                "last_updated": None,
            }}
        try:
            with open(self.learned_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            count = len(data.get("corrections", {}))
            if count > 0:
                print(f"[OCR Corrector] Loaded {count} learned corrections")
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not load learned corrections: {e}")
            return {"corrections": {}, "vocabulary_additions": [], "stats": {}}

    def _save_learned_corrections(self):
        """Persist learned corrections to JSON."""
        self.learned_data["stats"]["last_updated"] = datetime.now().isoformat()
        os.makedirs(self.learned_path.parent, exist_ok=True)
        with open(self.learned_path, "w", encoding="utf-8") as f:
            json.dump(self.learned_data, f, indent=2, ensure_ascii=False)

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 1: SymSpell Domain Matching
    # ══════════════════════════════════════════════════════════════════════════

    def _symspell_correct_word(self, word: str) -> Optional[str]:
        """
        Look up a word in the domain dictionary using fuzzy matching.
        Returns the corrected word or None if no good match.
        """
        if len(word) < 3:
            return None  # Don't correct tiny words/codes

        suggestions = self.sym_spell.lookup(
            word.upper(),
            Verbosity.CLOSEST,
            max_edit_distance=self.max_edit_distance,
        )

        if not suggestions:
            return None

        best = suggestions[0]

        # Validate: the suggestion must be reasonably close
        similarity = fuzz.ratio(word.upper(), best.term) / 100.0
        if similarity < 0.65:
            return None  # Too different, probably a different word

        # Don't "correct" if already a valid dictionary word
        if word.upper() == best.term:
            return None

        return best.term

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 0: Symbol Normalization
    # ══════════════════════════════════════════════════════════════════════════

    def _normalize_symbols(self, text: str) -> str:
        """Fix common character-level OCR confusions before word correction."""
        # Character-level replacements
        replacements = {
            '¥': 'V',
            '£': 'E',
            '|': 'I',
            '©': 'C',
            '®': 'R',
            '§': 'S',
            '°': 'o',
            '»': '>',
            '«': '<',
            '—': '-',
            '–': '-',
        }
        for char, sub in replacements.items():
            text = text.replace(char, sub)

        # Common digit/letter confusions in specific contexts
        # (e.g., 0UTPUT -> OUTPUT, but keep 3144 as digits)
        text = re.sub(r'\b0([A-Z]{3,})\b', r'O\1', text)  # 0UTPUT -> OUTPUT
        text = re.sub(r'\b([A-Z]{3,})0\b', r'\1O', text)  # CALIBRATI0N -> CALIBRATION
        
        return text

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 1: SymSpell Domain Matching
    # ══════════════════════════════════════════════════════════════════════════

    def correct_with_confidence(
        self, text: str, word_confidences: Optional[dict] = None
    ) -> str:
        """
        Correct text using confidence data from Tesseract.

        Args:
            text: Raw OCR text
            word_confidences: Dict mapping word → confidence (0.0–1.0).
                              If None, all words are treated as candidates.

        Returns:
            Corrected text
        """
        words = text.split()
        corrected_words = []

        for word in words:
            # Skip if it looks like a technical code (contains digits and special chars)
            # Pattern: 3144P, Pt100, 4-WIRE, 12.0-42.4
            if re.search(r'\d', word) and re.search(r'[^A-Za-z0-9]', word):
                corrected_words.append(word)
                continue

            clean_word = re.sub(r'[^A-Za-z0-9]', '', word)
            if not clean_word or len(clean_word) < 3:
                corrected_words.append(word)
                continue

            # Check confidence — skip high-confidence words
            if word_confidences:
                conf = word_confidences.get(clean_word, word_confidences.get(word, 1.0))
                if conf >= self.min_confidence:
                    corrected_words.append(word)
                    continue

            # Try correction
            replacement = self._correct_single_word(clean_word)
            if replacement and replacement != clean_word.upper():
                # Preserve surrounding punctuation
                prefix = word[:len(word) - len(word.lstrip(r'([{'))]
                suffix = word[len(word.rstrip(r')]}.,:;!?')):]
                corrected_words.append(f"{prefix}{replacement}{suffix}")
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 3: Learned Corrections
    # ══════════════════════════════════════════════════════════════════════════

    def _check_learned(self, word: str) -> Optional[str]:
        """Check if we've seen and corrected this word before."""
        corrections = self.learned_data.get("corrections", {})
        key = word.upper()
        if key in corrections:
            entry = corrections[key]
            # Only auto-apply if we've seen this enough times
            if entry.get("count", 0) >= 2 or entry.get("confidence", 0) >= self.auto_learn_threshold:
                return entry["correct"]
        return None

    def learn(self, wrong: str, correct: str, confidence: float = 0.90):
        """
        Teach the corrector a new correction.
        Call this when a user verifies/corrects OCR output.
        """
        corrections = self.learned_data.setdefault("corrections", {})
        key = wrong.upper()

        if key in corrections:
            corrections[key]["count"] = corrections[key].get("count", 0) + 1
            corrections[key]["confidence"] = max(
                corrections[key].get("confidence", 0), confidence
            )
        else:
            corrections[key] = {
                "correct": correct.upper(),
                "count": 1,
                "confidence": confidence,
            }

        # Also add the correct word to vocabulary if not already there
        vocab_additions = self.learned_data.setdefault("vocabulary_additions", [])
        if correct.upper() not in vocab_additions:
            vocab_additions.append(correct.upper())
            self.sym_spell.create_dictionary_entry(correct.upper(), 50)

        # Update stats
        stats = self.learned_data.setdefault("stats", {})
        stats["total_corrections_made"] = stats.get("total_corrections_made", 0) + 1

        self._save_learned_corrections()
        print(f"[OCR Corrector] Learned: '{wrong}' → '{correct}' (count: {corrections[key]['count']})")

    def learn_from_comparison(self, raw_text: str, corrected_text: str):
        """
        Automatically learn corrections by comparing raw OCR text with
        user-corrected text. Finds word-level differences and saves them.
        """
        raw_words = raw_text.upper().split()
        corrected_words = corrected_text.upper().split()

        if len(raw_words) != len(corrected_words):
            return  # Alignment mismatch, skip

        for raw_w, cor_w in zip(raw_words, corrected_words):
            clean_raw = re.sub(r'[^A-Z0-9]', '', raw_w)
            clean_cor = re.sub(r'[^A-Z0-9]', '', cor_w)
            if clean_raw and clean_cor and clean_raw != clean_cor:
                similarity = fuzz.ratio(clean_raw, clean_cor) / 100.0
                if similarity > 0.50:  # Only learn if somewhat similar
                    self.learn(clean_raw, clean_cor, confidence=0.95)

    # ══════════════════════════════════════════════════════════════════════════
    # Main Correction Pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _correct_single_word(self, word: str) -> Optional[str]:
        """
        Try all correction layers for a single word.
        Order: Learned → SymSpell Domain → None
        """
        # Layer 3 first (fastest, most reliable — user-verified)
        learned = self._check_learned(word)
        if learned:
            return learned

        # Layer 1: SymSpell domain dictionary
        symspell_result = self._symspell_correct_word(word)
        if symspell_result:
            return symspell_result

        return None

    def correct(self, text: str, word_confidences: Optional[dict] = None) -> str:
        """
        Main entry point — correct OCR text using all dynamic layers.

        Args:
            text: Raw OCR text to correct
            word_confidences: Optional dict of {word: confidence} from Tesseract

        Returns:
            Corrected text
        """
        # Apply Symbol Normalization first
        text = self._normalize_symbols(text)

        # Process line by line to preserve structure
        lines = text.split("\n")
        corrected_lines = []

        for line in lines:
            if not line.strip():
                corrected_lines.append(line)
                continue

            corrected_line = self.correct_with_confidence(line, word_confidences)
            corrected_lines.append(corrected_line)

        result = "\n".join(corrected_lines)

        # Update processing stats
        stats = self.learned_data.setdefault("stats", {})
        stats["total_documents_processed"] = stats.get("total_documents_processed", 0) + 1
        self._save_learned_corrections()

        return result

    def add_vocabulary(self, word: str, frequency: int = 50):
        """Add a new word to the domain vocabulary (persists to file)."""
        self.sym_spell.create_dictionary_entry(word.upper(), frequency)

        # Also append to vocabulary file
        with open(self.vocabulary_path, "a", encoding="utf-8") as f:
            f.write(f"\n{word.upper()} {frequency}")

        print(f"[OCR Corrector] Added to vocabulary: {word.upper()}")

    def get_stats(self) -> dict:
        """Get correction statistics."""
        return {
            "vocabulary_size": self.sym_spell.word_count,
            "learned_corrections": len(self.learned_data.get("corrections", {})),
            "total_documents": self.learned_data.get("stats", {}).get("total_documents_processed", 0),
            "total_corrections": self.learned_data.get("stats", {}).get("total_corrections_made", 0),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Utility: Extract word-level confidence from Tesseract data
# ══════════════════════════════════════════════════════════════════════════════

def extract_word_confidences(tesseract_data: dict) -> dict:
    """
    Parse Tesseract `image_to_data()` output into a word→confidence mapping.

    Args:
        tesseract_data: Output from pytesseract.image_to_data(output_type=Output.DICT)

    Returns:
        Dict mapping each word to its OCR confidence (0.0–1.0)
    """
    confidences = {}
    texts = tesseract_data.get("text", [])
    confs = tesseract_data.get("conf", [])

    for text, conf in zip(texts, confs):
        text = text.strip()
        if not text:
            continue
        try:
            conf_val = float(conf) if isinstance(conf, str) else float(conf)
        except (ValueError, TypeError):
            continue
        if conf_val < 0:
            continue
        confidences[text] = conf_val / 100.0  # Tesseract gives 0–100

    return confidences


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    corrector = OCRCorrector()

    test_inputs = [
        "REREMOUNT TEMPERAIURE TRANSMIITER",
        "SGREMOUNT OUNPUT ENCLOGURE",
        "ROSEMOUWT TEMPLRATURE TRANCMITTER",
        "FACTORY CALIBRATION HART PROTOCOL",  # already correct → should stay
    ]

    print("\n" + "=" * 60)
    print("  DYNAMIC OCR CORRECTOR TEST")
    print("=" * 60)

    for text in test_inputs:
        corrected = corrector.correct(text)
        print(f"\n  Input:  {text}")
        print(f"  Output: {corrected}")

    print(f"\n  Stats: {corrector.get_stats()}")
