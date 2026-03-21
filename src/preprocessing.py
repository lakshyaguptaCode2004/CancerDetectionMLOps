"""
=============================================================
src/preprocessing.py
=============================================================
Chest Cancer Classification - Preprocessing Module

Purpose:
  - Configure Keras ImageDataGenerators for train/val/test
  - Apply image augmentation to training set
  - Normalize pixel values to [0, 1]
  - Return ready-to-use data generators
  - Save a preprocessing_done.flag for DVC tracking
=============================================================
"""

import os
import json
import logging
from pathlib import Path

import yaml
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Logging setup ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PARAMS_FILE   = Path("params.yaml")
FLAG_FILE     = PROCESSED_DIR / "preprocessing_done.flag"


def load_params() -> dict:
    """Load hyperparameters from params.yaml."""
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def build_train_generator(
    train_dir: Path,
    image_size: tuple,
    batch_size: int,
    aug_params: dict
) -> ImageDataGenerator:
    """
    Build augmented ImageDataGenerator for training.
    Augmentation helps prevent overfitting on small medical datasets.
    """
    logger.info("Building training data generator with augmentation...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,                          # Normalize [0, 255] → [0, 1]
        rotation_range=aug_params["rotation_range"],
        zoom_range=aug_params["zoom_range"],
        horizontal_flip=aug_params["horizontal_flip"],
        width_shift_range=aug_params["width_shift_range"],
        height_shift_range=aug_params["height_shift_range"],
        fill_mode=aug_params["fill_mode"],
        shear_range=0.1,
        brightness_range=[0.8, 1.2],               # Random brightness
    )

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",                   # One-hot labels for 4 classes
        shuffle=True,
        seed=42
    )

    logger.info(f"  Classes found: {train_generator.class_indices}")
    logger.info(f"  Training samples: {train_generator.samples}")
    return train_generator


def build_val_generator(
    val_dir: Path,
    image_size: tuple,
    batch_size: int
) -> ImageDataGenerator:
    """
    Build non-augmented ImageDataGenerator for validation.
    Only rescale — NO augmentation during evaluation.
    """
    logger.info("Building validation data generator...")

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_generator = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    logger.info(f"  Validation samples: {val_generator.samples}")
    return val_generator


def build_test_generator(
    test_dir: Path,
    image_size: tuple,
    batch_size: int
) -> ImageDataGenerator:
    """
    Build non-augmented ImageDataGenerator for test set.
    shuffle=False so predictions align with true labels.
    """
    logger.info("Building test data generator...")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    logger.info(f"  Test samples: {test_generator.samples}")
    return test_generator


def get_class_weights(train_generator) -> dict:
    """
    Compute class weights to handle class imbalance.
    Gives higher weight to under-represented classes.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes    = np.array(list(train_generator.class_indices.values()))
    y_labels   = train_generator.classes

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_labels),
        y=y_labels
    )

    class_weight_dict = dict(zip(np.unique(y_labels), weights))
    logger.info(f"  Class weights: {class_weight_dict}")
    return class_weight_dict


def save_class_mapping(train_generator, output_dir: Path) -> None:
    """Save class name → index mapping for inference."""
    mapping = train_generator.class_indices
    # Invert: index → class name (for prediction output)
    inv_mapping = {v: k for k, v in mapping.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(inv_mapping, f, indent=2)
    logger.info(f"  Class mapping saved to: {output_dir / 'class_mapping.json'}")


def main():
    logger.info("Starting Preprocessing Stage...")

    # Load params
    params     = load_params()
    image_size = tuple(params["data"]["image_size"])
    batch_size = params["data"]["batch_size"]
    aug_params = params["augmentation"]

    train_dir = RAW_DIR / "train"
    val_dir   = RAW_DIR / "valid"
    test_dir  = RAW_DIR / "test"

    # Build generators
    if train_dir.exists() and any(train_dir.iterdir()):
        train_gen = build_train_generator(train_dir, image_size, batch_size, aug_params)
        val_gen   = build_val_generator(val_dir, image_size, batch_size)
        test_gen  = build_test_generator(test_dir, image_size, batch_size)

        # Class weights for imbalanced data
        class_weights = get_class_weights(train_gen)

        # Save class mapping
        save_class_mapping(train_gen, PROCESSED_DIR)
    else:
        logger.warning("Training data not found — skipping generator build.")
        logger.info("Place your dataset at: data/raw/train/, data/raw/valid/, data/raw/test/")

    # Write flag file for DVC to track stage completion
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FLAG_FILE.write_text("Preprocessing completed successfully.")
    logger.info(f"Flag written: {FLAG_FILE}")
    logger.info("Preprocessing Stage completed ✓")


if __name__ == "__main__":
    main()
