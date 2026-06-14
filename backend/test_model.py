import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')

TASKS = {
    'type': ['Straight', 'Wavy', 'Bald', 'Curly', 'Dreadlocks', 'Kinky'],
    'condition': ['Healthy', 'Dry', 'Oily', 'Damaged', 'Hairfall'],
    'disease': ['None', 'Dandruff', 'Alopecia', 'SeborrheicDermatitis', 'Psoriasis'],
    'solution': ['Moisturizing', 'AntiDandruff', 'HairFallTreatment', 'ScalpMassage', 'NoAction']
}


def build_backbone(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    return inputs, x


def build_single_task_model(task_name, num_classes):
    inputs, x = build_backbone()
    outputs = layers.Dense(num_classes, activation='softmax', name=task_name)(x)
    return Model(inputs=inputs, outputs=outputs, name=f'{task_name}_model')


def load_or_build_model(task_name, num_classes):
    model_path = os.path.join(MODEL_DIR, f'{task_name}_model.h5')
    if os.path.exists(model_path):
        return load_model(model_path)
    return build_single_task_model(task_name, num_classes)


def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def prepare_datasets(task_name, labels):
    dataset_root = os.path.join(DATASET_DIR, task_name)
    if not os.path.isdir(dataset_root):
        return None, None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root,
        labels='inferred',
        label_mode='categorical',
        class_names=labels,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root,
        labels='inferred',
        label_mode='categorical',
        class_names=labels,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


def evaluate_model(model, dataset_root, classes):
    total = 0
    correct = 0

    for class_name in classes:
        folder = os.path.join(dataset_root, class_name)
        if not os.path.isdir(folder):
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:50]

        if not files:
            continue

        class_correct = 0
        for fname in files:
            img_path = os.path.join(folder, fname)
            x = preprocess_image(img_path)
            pred = model.predict(x, verbose=0)
            pred_label = classes[np.argmax(pred)]
            if pred_label == class_name:
                class_correct += 1
            total += 1

        accuracy = class_correct / len(files) * 100
        correct += class_correct
        print(f"{class_name:16} → {class_correct}/{len(files)} = {accuracy:.1f}%")

    if total:
        print(f"OVERALL → {correct}/{total} = {correct/total*100:.1f}%")
    else:
        print("No samples found for this task.")


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    for task_name, labels in TASKS.items():
        print(f"\n=== {task_name.upper()} MODEL ===")
        model_path = os.path.join(MODEL_DIR, f'{task_name}_model.h5')

        model = load_or_build_model(task_name, len(labels))
        model = compile_model(model)

        train_ds, val_ds = prepare_datasets(task_name, labels)
        if train_ds is not None:
            print(f"Training {task_name} model...")
            model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
            model.save(model_path)
            print(f"Saved {task_name} model to {model_path}")
        else:
            print(f"Missing dataset folder for {task_name}: {os.path.join(DATASET_DIR, task_name)}")

        print(f"Evaluating {task_name} model...")
        dataset_root = os.path.join(DATASET_DIR, task_name)
        evaluate_model(model, dataset_root, labels)