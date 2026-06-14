# deep_clean.py — Removes corrupted/unreadable images from all dataset folders

import os
from PIL import Image

DATASET_DIR = "../datasets"

CLASSES = [
    "straight", "wavy", "bald", "curly", "dreadlocks",
    "dry", "frizzy", "hairfall", "healthy", "kinky", "notbald"
]

total_removed = 0
total_kept    = 0

for cls in CLASSES:
    cls_path = os.path.join(DATASET_DIR, cls)
    if not os.path.exists(cls_path):
        print(f"⚠  Skipping {cls} — folder not found")
        continue

    removed = 0
    kept    = 0

    for fname in os.listdir(cls_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue

        fpath = os.path.join(cls_path, fname)
        try:
            img = Image.open(fpath)
            img.verify()  # Check for corruption

            # Re-open to check size (verify() closes the file)
            img2 = Image.open(fpath)
            w, h = img2.size
            if w < 50 or h < 50:
                raise ValueError(f"Too small: {w}x{h}")

            kept += 1

        except Exception as e:
            os.remove(fpath)
            removed += 1
            print(f"  ❌ Removed {cls}/{fname} — {e}")

    print(f"✅ {cls:12s} → kept: {kept:4d}  removed: {removed}")
    total_removed += removed
    total_kept    += kept

print(f"\n{'='*40}")
print(f"Total kept:    {total_kept}")
print(f"Total removed: {total_removed}")
print(f"{'='*40}")
print("✅ Deep clean complete! Ready to train.")