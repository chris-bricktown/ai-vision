import random
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
# tf_keras (Keras 2) instead of tensorflow.keras (Keras 3): Keras 3's model
# serialization format (nested Functional models, node data) isn't fully
# compatible with tensorflowjs_converter's layers-model deserializer.
import tf_keras as keras
from tf_keras import layers, models
from PIL import Image

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
VAL_SPLIT = 0.2
SEED = 42

DATASET_DIR = Path("dataset/images")
OUT_DIR = Path("trained_model")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Persian", "Siamese", "Bengal", "beagle", "pug", "yorkshire_terrier"]

random.seed(SEED)
tf.random.set_seed(SEED)


def collect_files():
    by_class = {}
    for cls in CLASSES:
        files = sorted(DATASET_DIR.glob(f"{cls}_*.jpg"))
        by_class[cls] = files
        print(f"{cls}: {len(files)} files")
    return by_class


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32)
        # MobileNetV2 preprocessing (rescale to [-1, 1]) applied here in Python
        # rather than as a layer inside the model: Keras 3 traces
        # preprocess_input as TFOpLambda-style ops (TrueDivide/Subtract) that
        # TF.js's layers-model deserializer doesn't recognize. The exported
        # model expects already-normalized input; app.js normalizes to match.
        return (arr / 127.5) - 1.0
    except Exception as e:
        print(f"skip {path}: {e}")
        return None


def build_dataset(by_class):
    train_x, train_y, val_x, val_y = [], [], [], []
    for idx, cls in enumerate(CLASSES):
        files = list(by_class[cls])
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_SPLIT))
        val_files = files[:n_val]
        train_files = files[n_val:]
        for f in train_files:
            arr = load_image(f)
            if arr is not None:
                train_x.append(arr)
                train_y.append(idx)
        for f in val_files:
            arr = load_image(f)
            if arr is not None:
                val_x.append(arr)
                val_y.append(idx)
    return (
        np.stack(train_x), np.array(train_y),
        np.stack(val_x), np.array(val_y),
    )


def build_model(num_classes):
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    by_class = collect_files()
    train_x, train_y, val_x, val_y = build_dataset(by_class)
    print(f"train: {train_x.shape}, val: {val_x.shape}")

    model = build_model(len(CLASSES))
    model.summary()

    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    val_loss, val_acc = model.evaluate(val_x, val_y)
    print(f"Final val accuracy: {val_acc:.4f}")

    model.save(OUT_DIR / "saved_model.h5")
    with open(OUT_DIR / "labels.json", "w") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    with open(OUT_DIR / "eval.json", "w") as f:
        json.dump({"val_loss": float(val_loss), "val_accuracy": float(val_acc)}, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
