import tensorflow as tf
import numpy as np
from PIL import Image
import sys

model = tf.keras.models.load_model("../model/best_model.h5")
# ✅ Updated classes
classes = ['Straight', 'Wavy', 'bald', 'curly', 'dreadlocks',
           'dry', 'frizzy', 'hairfall', 'healthy', 'kinky', 'notbald']

if len(sys.argv) < 2:
    print("Usage: python3 predict.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]

try:
    img = Image.open(img_path).convert("RGB").resize((224, 224))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)


img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

# ✅ No softmax needed — already applied in model
confidence = float(np.max(prediction)) * 100
result = classes[np.argmax(prediction)]

print(f"Hair Type : {result}")
print(f"Confidence: {round(confidence, 2)}%")