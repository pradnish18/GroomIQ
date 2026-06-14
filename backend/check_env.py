import sys
print("Python:", sys.version, flush=True)

print("Importing tensorflow...", flush=True)
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__, flush=True)
except Exception as e:
    print("TF FAILED:", type(e).__name__, e, flush=True)
    sys.exit(1)

print("Importing numpy...", flush=True)
try:
    import numpy as np
    print("NumPy:", np.__version__, flush=True)
except Exception as e:
    print("NumPy FAILED:", e, flush=True)

print("Importing PIL...", flush=True)
try:
    from PIL import Image
    print("Pillow: OK", flush=True)
except Exception as e:
    print("Pillow FAILED:", e, flush=True)

print("ALL DONE", flush=True)
