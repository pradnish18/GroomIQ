import os
import shutil

DATASET_PATH = "../datasets"

# ===== HAIR TYPE CLASSES =====
type_classes = ['Straight', 'Wavy', 'curly', 'kinky', 'dreadlocks']

# ===== HAIR CONDITION CLASSES =====
condition_classes = ['dry', 'hairfall', 'bald', 'healthy']

TYPE_PATH      = "../datasets_type"
CONDITION_PATH = "../datasets_condition"

# Create folders
for cls in type_classes:
    os.makedirs(os.path.join(TYPE_PATH, cls), exist_ok=True)

for cls in condition_classes:
    os.makedirs(os.path.join(CONDITION_PATH, cls), exist_ok=True)

print("===== SPLITTING DATASET =====")

# Copy type classes
print("\n--- Hair TYPE dataset ---")
for cls in type_classes:
    src = os.path.join(DATASET_PATH, cls)
    dst = os.path.join(TYPE_PATH, cls)
    if not os.path.exists(src):
        print(f"  ⚠️  {cls}: folder not found")
        continue
    files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print(f"  ✅ {cls}: {len(files)} images")

# Copy condition classes
print("\n--- Hair CONDITION dataset ---")
for cls in condition_classes:
    src = os.path.join(DATASET_PATH, cls)
    dst = os.path.join(CONDITION_PATH, cls)
    if not os.path.exists(src):
        print(f"  ⚠️  {cls}: folder not found")
        continue
    files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print(f"  ✅ {cls}: {len(files)} images")

print("\n✅ Dataset split complete!")
print(f"   Type dataset:      {TYPE_PATH}")
print(f"   Condition dataset: {CONDITION_PATH}")