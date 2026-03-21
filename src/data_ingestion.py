"""
=============================================================
src/data_ingestion.py
=============================================================
Chest Cancer Classification - Data Ingestion Module

Purpose:
  - Validate raw dataset folder structure
  - Count images per class per split (train/test/valid)
  - Display basic dataset statistics
  - Save class counts to JSON for downstream stages
=============================================================
"""

import os
import json
import logging
from pathlib import Path
from collections import defaultdict

# ── Logging setup ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
RAW_DIR        = Path("data/raw")
PROCESSED_DIR  = Path("data/processed")
OUTPUT_FILE    = PROCESSED_DIR / "class_counts.json"
VALID_SPLITS   = ["train", "test", "valid"]
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def validate_structure(raw_dir: Path) -> bool:
    """
    Check that the expected folder structure exists.
    Expected: data/raw/{train,test,valid}/{class_name}/image.jpg
    """
    logger.info(f"Validating dataset structure at: {raw_dir}")

    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.info("Please download dataset from Kaggle and place it at data/raw/")
        logger.info("  kaggle datasets download -d mohamedhanyyy/chest-ctscan-images")
        return False

    missing = []
    for split in VALID_SPLITS:
        split_path = raw_dir / split
        if not split_path.exists():
            missing.append(str(split_path))

    if missing:
        logger.warning(f"Missing split folders: {missing}")
        logger.info("Creating placeholder structure for demonstration...")
        for split in VALID_SPLITS:
            (raw_dir / split).mkdir(parents=True, exist_ok=True)

    logger.info("Dataset structure validated successfully ✓")
    return True


def count_images(raw_dir: Path) -> dict:
    """
    Walk through all split/class folders and count images.

    Returns:
        {
          "train": {"adenocarcinoma": 100, "normal": 80, ...},
          "test":  {...},
          "valid": {...},
          "total": {"adenocarcinoma": 200, ...}
        }
    """
    counts = {}
    total  = defaultdict(int)

    for split in VALID_SPLITS:
        split_path = raw_dir / split
        if not split_path.exists():
            logger.warning(f"Split folder not found: {split_path}")
            counts[split] = {}
            continue

        split_counts = {}
        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])

        if not class_dirs:
            logger.warning(f"No class sub-folders found in: {split_path}")
            counts[split] = {}
            continue

        for class_dir in class_dirs:
            class_name = class_dir.name
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            ]
            n = len(images)
            split_counts[class_name] = n
            total[class_name] += n

        counts[split] = split_counts

    counts["total"] = dict(total)
    return counts


def print_stats(counts: dict) -> None:
    """Pretty-print dataset statistics."""
    print("\n" + "=" * 55)
    print("  CHEST CT-SCAN DATASET STATISTICS")
    print("=" * 55)

    for split in VALID_SPLITS:
        split_data = counts.get(split, {})
        if not split_data:
            continue
        print(f"\n  [{split.upper()}]")
        total_split = 0
        for cls, n in split_data.items():
            print(f"    {cls:<35} {n:>5} images")
            total_split += n
        print(f"    {'─' * 43}")
        print(f"    {'TOTAL':<35} {total_split:>5} images")

    grand_total = counts.get("total", {})
    if grand_total:
        print(f"\n  [GRAND TOTAL ACROSS ALL SPLITS]")
        overall = 0
        for cls, n in grand_total.items():
            print(f"    {cls:<35} {n:>5} images")
            overall += n
        print(f"    {'─' * 43}")
        print(f"    {'ALL CLASSES':<35} {overall:>5} images")

    print("=" * 55 + "\n")


def save_counts(counts: dict, output_path: Path) -> None:
    """Save class counts to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(counts, f, indent=2)
    logger.info(f"Class counts saved to: {output_path}")


def main():
    logger.info("Starting Data Ingestion Stage...")

    # Step 1: Validate structure
    is_valid = validate_structure(RAW_DIR)

    # Step 2: Count images
    logger.info("Counting images per class per split...")
    counts = count_images(RAW_DIR)

    # Step 3: Print statistics
    print_stats(counts)

    # Step 4: Save to JSON
    save_counts(counts, OUTPUT_FILE)

    logger.info("Data Ingestion Stage completed successfully ✓")
    return counts


if __name__ == "__main__":
    main()
