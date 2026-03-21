"""
=============================================================
src/evaluate.py
=============================================================
Chest Cancer Classification - Evaluation Module

Purpose:
  - Load best saved model
  - Evaluate on test set
  - Generate confusion matrix
  - Print classification report
  - Save evaluation metrics for DVC
  - Plot accuracy/loss curves
=============================================================
"""

import json
import logging
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
PARAMS_FILE   = Path("params.yaml")

CLASS_NAMES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_model(model_path: Path):
    """Load the saved Keras model."""
    import tensorflow as tf
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded successfully ✓")
    return model


def get_test_generator(params: dict):
    """Build test data generator."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    image_size = tuple(params["data"]["image_size"])
    batch_size = params["data"]["batch_size"]

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        str(RAW_DIR / "test"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    return test_gen


def evaluate_model(model, test_gen) -> dict:
    """Compute test loss, accuracy, and predictions."""
    logger.info("Evaluating model on test set...")

    # Overall metrics
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    logger.info(f"  Test Loss:     {loss:.4f}")
    logger.info(f"  Test Accuracy: {accuracy:.4f}")

    # Predictions
    logger.info("Generating predictions...")
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_gen.classes

    return {
        "loss":       loss,
        "accuracy":   accuracy,
        "y_true":     y_true,
        "y_pred":     y_pred,
        "y_pred_prob": y_pred_prob,
    }


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_dir: Path) -> None:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("Confusion Matrix - Chest Cancer CNN", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = save_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Confusion matrix saved: {save_path}")

    # Also save as CSV for DVC plots
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(save_dir / "confusion_matrix.csv")


def plot_training_history(history_path: Path, save_dir: Path) -> None:
    """
    Plot training/validation accuracy and loss curves.
    Loads from saved history JSON if available.
    """
    history_file = PROCESSED_DIR / "training_history.json"
    if not history_file.exists():
        logger.warning("Training history not found — skipping curve plots.")
        return

    with open(history_file) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──────────────────────────────────────────
    axes[0].plot(history.get("accuracy", []),     label="Train Accuracy", color="steelblue", linewidth=2)
    axes[0].plot(history.get("val_accuracy", []), label="Val Accuracy",   color="coral",     linewidth=2, linestyle="--")
    axes[0].set_title("Model Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────
    axes[1].plot(history.get("loss", []),     label="Train Loss", color="steelblue", linewidth=2)
    axes[1].plot(history.get("val_loss", []), label="Val Loss",   color="coral",     linewidth=2, linestyle="--")
    axes[1].set_title("Model Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History - Chest Cancer CNN", fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Training curves saved: {save_path}")


def print_classification_report(y_true, y_pred, class_names: list) -> str:
    """Print and return full sklearn classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60 + "\n")
    return report


def save_metrics(eval_results: dict, class_names: list, save_dir: Path) -> None:
    """Save evaluation metrics as JSON for DVC."""
    y_true = eval_results["y_true"]
    y_pred = eval_results["y_pred"]

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    metrics = {
        "test_accuracy":   round(eval_results["accuracy"], 4),
        "test_loss":       round(eval_results["loss"], 4),
        "f1_macro":        round(f1_macro, 4),
        "f1_weighted":     round(f1_weighted, 4),
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  Metrics saved: {save_dir / 'test_metrics.json'}")
    logger.info(f"\n  Final Test Results:")
    for k, v in metrics.items():
        logger.info(f"    {k:<25} {v}")


def main():
    logger.info("Starting Evaluation Stage...")
    params = load_params()

    # Paths
    model_path = MODELS_DIR / "best_model.h5"
    save_dir   = PROCESSED_DIR / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if not model_path.exists():
        logger.warning(f"No trained model found at {model_path}")
        logger.info("Please run training first: python src/train_cnn.py")
        # Save dummy metrics for DVC compatibility
        dummy = {"test_accuracy": 0.0, "test_loss": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}
        with open(PROCESSED_DIR / "test_metrics.json", "w") as f:
            json.dump(dummy, f, indent=2)
        return

    model = load_model(model_path)

    # Check data
    if not (RAW_DIR / "test").exists():
        logger.warning("Test data not found. Skipping evaluation.")
        return

    test_gen = get_test_generator(params)
    class_names = list(test_gen.class_indices.keys())

    # Evaluate
    eval_results = evaluate_model(model, test_gen)

    # Confusion matrix
    plot_confusion_matrix(
        eval_results["y_true"],
        eval_results["y_pred"],
        class_names,
        save_dir
    )

    # Classification report
    report = print_classification_report(
        eval_results["y_true"],
        eval_results["y_pred"],
        class_names
    )

    # Save report as text
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Training curves
    plot_training_history(PROCESSED_DIR / "training_history.json", save_dir)

    # Save metrics JSON
    save_metrics(eval_results, class_names, PROCESSED_DIR)

    logger.info("Evaluation Stage completed ✓")


if __name__ == "__main__":
    main()
