"""
Comprehensive Model Evaluation Script
======================================
Evaluates the pre-trained hair analysis models on the existing datasets.
Produces per-class accuracy, overall accuracy, and a confusion-style report.
"""

import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR      = os.path.join(BASE_DIR, 'model')
DATASET_TYPE   = os.path.join(BASE_DIR, 'datasets_type')
DATASET_COND   = os.path.join(BASE_DIR, 'datasets_condition')
DATASET_MAIN   = os.path.join(BASE_DIR, 'datasets')

# ── Class lists (must match training order) ────────────────────────────────
TYPE_CLASSES      = ['Straight', 'Wavy', 'curly', 'dreadlocks', 'kinky']
CONDITION_CLASSES = ['bald', 'dry', 'hairfall', 'healthy']
MAIN_CLASSES      = ['Straight', 'Wavy', 'bald', 'curly', 'dreadlocks',
                     'dry', 'frizzy', 'hairfall', 'healthy', 'kinky', 'notbald']

IMG_SIZE_MAIN = 299   # best_model uses 299 (InceptionV3 / EfficientNetB3)
IMG_SIZE_SUB  = 224   # type & condition models use 224 (EfficientNetB0)


def load_images(dataset_dir, class_names, img_size, max_per_class=None):
    """Load images from class sub-folders. Returns (images, labels, label_names)."""
    images, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠  Folder not found: {cls_dir}")
            continue
        files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        if max_per_class:
            files = files[:max_per_class]
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert('RGB').resize((img_size, img_size))
                images.append(np.array(img, dtype=np.float32))
                labels.append(idx)
            except Exception:
                pass  # skip corrupted
    return np.array(images), np.array(labels), class_names


def evaluate(model, images, labels, class_names, model_name):
    """Run predictions and print per-class + overall accuracy."""
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")

    if len(images) == 0:
        print("  No images loaded — skipping.\n")
        return

    print(f"  Total images: {len(images)}")
    start = time.time()
    preds = model.predict(images, batch_size=32, verbose=0)
    elapsed = time.time() - start
    pred_labels = np.argmax(preds, axis=1)

    correct_total = 0
    total_total   = 0

    print(f"\n  {'Class':<16} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*44}")

    for idx, cls in enumerate(class_names):
        mask  = (labels == idx)
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((pred_labels[mask] == idx).sum())
        acc     = correct / total * 100
        correct_total += correct
        total_total   += total
        print(f"  {cls:<16} {correct:>8} {total:>8} {acc:>9.1f}%")

    overall = correct_total / total_total * 100 if total_total else 0
    print(f"  {'-'*44}")
    print(f"  {'OVERALL':<16} {correct_total:>8} {total_total:>8} {overall:>9.1f}%")
    print(f"  Inference time: {elapsed:.1f}s  ({len(images)/elapsed:.0f} img/s)")
    print()
    return overall


def main():
    print("=" * 60)
    print("  HAIR ANALYSIS — MODEL ACCURACY REPORT")
    print("=" * 60)

    results = {}

    # ── 1. Best / Main model ──────────────────────────────────────────
    main_model_path = os.path.join(MODEL_DIR, 'best_model.h5')
    if os.path.exists(main_model_path):
        print(f"\nLoading {main_model_path} ...")
        model = tf.keras.models.load_model(main_model_path)
        imgs, lbls, cls = load_images(DATASET_MAIN, MAIN_CLASSES, IMG_SIZE_MAIN)
        acc = evaluate(model, imgs, lbls, cls, "best_model.h5  (main 11-class)")
        results['best_model (11-class)'] = acc
        del model, imgs, lbls  # free memory
    else:
        print(f"\n⚠  {main_model_path} not found — skipping.")

    # ── 2. Type model ─────────────────────────────────────────────────
    type_model_path = os.path.join(MODEL_DIR, 'type_model.h5')
    if os.path.exists(type_model_path):
        print(f"\nLoading {type_model_path} ...")
        model = tf.keras.models.load_model(type_model_path)
        imgs, lbls, cls = load_images(DATASET_TYPE, TYPE_CLASSES, IMG_SIZE_SUB)
        acc = evaluate(model, imgs, lbls, cls, "type_model.h5  (hair type)")
        results['type_model (5-class)'] = acc
        del model, imgs, lbls
    else:
        print(f"\n⚠  {type_model_path} not found — skipping.")

    # ── 3. Condition model ────────────────────────────────────────────
    cond_model_path = os.path.join(MODEL_DIR, 'condition_model.h5')
    if os.path.exists(cond_model_path):
        print(f"\nLoading {cond_model_path} ...")
        model = tf.keras.models.load_model(cond_model_path)
        imgs, lbls, cls = load_images(DATASET_COND, CONDITION_CLASSES, IMG_SIZE_SUB)
        acc = evaluate(model, imgs, lbls, cls, "condition_model.h5  (hair condition)")
        results['condition_model (4-class)'] = acc
        del model, imgs, lbls
    else:
        print(f"\n⚠  {cond_model_path} not found — skipping.")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, acc in results.items():
        if acc is not None:
            print(f"  {name:<35} → {acc:.1f}%")
    print("=" * 60)
    print("  ✅ Evaluation complete!")


if __name__ == '__main__':
    main()
