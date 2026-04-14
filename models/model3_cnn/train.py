#!/usr/bin/env python3
"""
Model 3: CNN — Training Script
================================
Train a convolutional neural network for image classification.
Transfer learning is recommended (ResNet50, EfficientNet, DenseNet).

Framework: TensorFlow / Keras

IMPORTANT: Resize images before training! Raw images may be very high resolution
and will cause memory errors if loaded full-size.
"""
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RAW_IMAGES = Path("../../data/raw/retinal_scan_images")
SAVED_MODEL_DIR = Path("../../models/model3_cnn/saved_model/")
CSV_PATH = Path("../../data/raw/retinal_labels.csv")


def load_images(image_dir, csv_path, target_size=(224, 224), batch_size=32):
    """Load and preprocess images from directory.

    Example using Keras ImageDataGenerator:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            horizontal_flip=True,
            zoom_range=0.2,
        )
        train_gen = datagen.flow_from_directory(
            image_dir,
            target_size=target_size,
            batch_size=32,
            class_mode='binary',
            subset='training',
        )

    IMPORTANT: Handle class imbalance with class_weight or augmentation.
    """
    # Read in image directory path and create dateframe for retinal_labels.csv
    image_dir = Path(image_dir)

    df = pd.read_csv(csv_path)

    # Add new columns to the dataframe

    # filename
    df["filename"] = df["id_code"].astype(str)
    df["filename"] = df["filename"].apply(
        lambda x: x if x.lower().endswith(".png") else f"{x}.png"
    )

    # filepath
    df["filepath"] = df["filename"].apply(lambda x: str(image_dir / x))

    # binary label, convert to string
    df["binary_label"] = df["diagnosis"].apply(lambda x: 0 if int(x) == 0 else 1)
    df["binary_label"] = df["binary_label"].astype(str)

    # Split data
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["binary_label"],
        random_state=42
    )

    # Generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="binary_label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="binary_label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen, train_df, val_df


def build_model():
    """Build or fine-tune a CNN.

    Transfer learning example:
        import tensorflow as tf

        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False  # Freeze base layers initially

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_data, val_data):
    """Train the CNN with callbacks.

    Use EarlyStopping and optionally ReduceLROnPlateau.
    Pass class_weight to model.fit() to handle imbalance.
    """

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    return history


def evaluate_model(model, val_data):
    """Evaluate CNN performance.

    Must include:
    - Accuracy and weighted F1
    - Confusion matrix
    - Sample predictions with images

    Bonus: Grad-CAM visualizations showing what the model "sees"
    """

    # Reset generator to ensure predictions align with labels
    val_data.reset()

    # Predict probabilities
    y_pred_probs = model.predict(val_data)

    # Convert probabilities to binary class predictions
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # True labels
    y_true = val_data.classes

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Show sample predictions with images
    x_batch, y_batch = next(val_data)
    sample_probs = model.predict(x_batch)
    sample_preds = (sample_probs > 0.5).astype(int).flatten()

    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(x_batch))):
        plt.figure()
        plt.imshow(x_batch[i])
        plt.title(
            f"Actual: {int(y_batch[i])} | Pred: {sample_preds[i]} | Prob: {sample_probs[i][0]:.3f}"
        )
        plt.axis("off")
        plt.show()

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm
    }


def save_model(model):
    """Save the trained model.

    Example:
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(SAVED_MODEL_DIR / "model.keras")
    """
    if not os.path.exists(SAVED_MODEL_DIR):
        os.makedirs(SAVED_MODEL_DIR)

    model.save(os.path.join(SAVED_MODEL_DIR, 'cnn_model.keras'))


def main():
    # 1. Load and preprocess images
    # train_data, val_data = load_images(RAW_IMAGES / "images")

    # 2. Build model
    # model = build_model()

    # 3. Train
    # train_model(model, train_data, val_data)

    # 4. Evaluate
    # evaluate_model(model, val_data)

    # 5. Save
    # save_model(model)


    print("\n=== STEP 1: Load Data ===")
    train_data, val_data, train_df, val_df = load_images(
        image_dir=RAW_IMAGES,
        csv_path=CSV_PATH,
        target_size=(224, 224),
        batch_size=32
    )

    print("Train samples:", train_data.samples)
    print("Validation samples:", val_data.samples)
    print("Class indices:", train_data.class_indices)

    # Inspect one batch
    x_batch, y_batch = next(train_data)
    print("Batch X shape:", x_batch.shape)
    print("Batch y shape:", y_batch.shape)
    print("Sample labels:", y_batch[:10])

    print("\n=== STEP 2: Build Model ===")
    model = build_model()
    model.summary()

    print("\n=== STEP 3: Train Model ===")
    history = train_model(model, train_data, val_data)

    print("\n=== STEP 4: Training Results ===")
    print("History keys:", history.history.keys())

    if "loss" in history.history:
        print("Final training loss:", history.history["loss"][-1])

    if "accuracy" in history.history:
        print("Final training accuracy:", history.history["accuracy"][-1])

    if "val_loss" in history.history:
        print("Final validation loss:", history.history["val_loss"][-1])

    if "val_accuracy" in history.history:
        print("Final validation accuracy:", history.history["val_accuracy"][-1])

    print("\n=== STEP 5: Evaluate Model ===")
    results = evaluate_model(model, val_data)
    print(results)

    print("\n=== STEP 6: Save Model ===")
    save_model(model)
    print("\n=== Model saved successfully ===")    

    print("\n=== Training test complete ===")


if __name__ == "__main__":
    main()
