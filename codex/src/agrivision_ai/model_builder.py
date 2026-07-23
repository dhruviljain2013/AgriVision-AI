"""Model creation for AgriVision AI."""

from __future__ import annotations

import tensorflow as tf


def build_data_augmentation() -> tf.keras.Sequential:
    """Create small random image changes for training.

    Augmentation helps the model learn patterns instead of memorizing exact
    photos. For plant leaves, gentle flips, rotations, zooms, and contrast
    changes are useful because real photos will not always be perfectly aligned.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def build_transfer_learning_model(config) -> tf.keras.Model:
    """Build the configured transfer learning model for the leaf classes.

    The selected ImageNet model is used only as a frozen feature extractor.
    AgriVision AI trains a small classifier head on top using the user's plant
    leaf dataset.
    """

    inputs = tf.keras.Input(shape=(config.image_size[0], config.image_size[1], 3), name="leaf_image")
    augmented = build_data_augmentation()(inputs)
    base_model, preprocessed = build_feature_extractor(config, augmented)
    base_model.trainable = False

    features = base_model(preprocessed, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    dropped = tf.keras.layers.Dropout(config.dropout_rate, name="dropout")(pooled)
    dense = tf.keras.layers.Dense(config.dense_units, activation="relu", name="classifier_dense")(dropped)
    outputs = tf.keras.layers.Dense(
        len(config.classes),
        activation="softmax",
        name="class_probabilities",
    )(dense)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"AgriVision_{config.model_architecture}",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_feature_extractor(config, inputs):
    """Create the frozen ImageNet feature extractor selected in config."""

    input_shape = (config.image_size[0], config.image_size[1], 3)

    if config.model_architecture == "MobileNetV2":
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        return base_model, preprocessed

    if config.model_architecture == "EfficientNetB0":
        preprocessed = tf.keras.applications.efficientnet.preprocess_input(inputs)
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        return base_model, preprocessed

    if config.model_architecture == "ResNet50":
        preprocessed = tf.keras.applications.resnet50.preprocess_input(inputs)
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        return base_model, preprocessed

    raise ValueError(f"Unsupported model architecture: {config.model_architecture}")


def build_mobilenetv2_model(config) -> tf.keras.Model:
    """Backward-compatible helper for earlier project phases."""

    return build_transfer_learning_model(config)
