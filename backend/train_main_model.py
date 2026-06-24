import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

dataset_path = "../datasets"
all_classes  = ['Straight', 'Wavy', 'bald', 'curly', 'dreadlocks',
                'dry', 'frizzy', 'hairfall', 'healthy', 'kinky', 'notbald']

print("===== COMBINED HAIR MODEL (11 classes) =====")
print("Classes:", all_classes)

IMG_SIZE   = 224
BATCH_SIZE = 32

full_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path, seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)

class_names = full_ds.class_names
num_classes = len(class_names)
print(f"Classes detected: {class_names}")

total = sum(1 for _ in full_ds.unbatch())
train_size = int(0.7 * total)
val_size   = int(0.2 * total)

train_ds = full_ds.take(train_size // BATCH_SIZE)
val_ds   = full_ds.skip(train_size // BATCH_SIZE).take(val_size // BATCH_SIZE)
test_ds  = full_ds.skip((train_size + val_size) // BATCH_SIZE)

print(f"Train: {train_size}, Val: {val_size}, Test: {total - train_size - val_size}")

def normalize(image, label):
    return preprocess_input(image), label

train_ds = train_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("../model/best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1)
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()

print("\n===== PHASE 1: Frozen base =====")
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

print("\n===== PHASE 2: Fine tuning =====")
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks_ft = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("../model/best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8, verbose=1)
]

model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks_ft)
model.save("../model/best_model.h5")

print("\n===== EVALUATION ON TEST SET =====")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test loss: {test_loss:.4f}")
print("\nCombined model saved to ../model/best_model.h5")
