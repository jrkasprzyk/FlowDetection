# NOTE! I've found that it might make more sense to have this be part of the project
# script and not hidden in the /src/ folder, so this might be deprecated at some point

import tensorflow as tf


def train_model(config, train_ds, val_ds, num_classes):
    '''Build, compile, and train a CNN image classifier.

    Architecture: three Conv2D+MaxPool blocks for feature extraction, followed by
    a Dense hidden layer and a final logits output layer.

    Args:
        config: Dict with keys: edge_size, activation_function, epochs, verbose.
        train_ds: Batched training tf.data.Dataset.
        val_ds: Batched validation tf.data.Dataset.
        num_classes: Number of output classes (determines the size of the final layer).

    Returns:
        (model, history): Trained Keras model and the History object from model.fit().
    '''
    # The model processes square RGB images of size edge_size x edge_size x 3.
    # Each Conv2D+MaxPooling2D block learns increasingly abstract spatial features
    # while halving the spatial dimensions (MaxPool stride defaults to the pool size).
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(config["edge_size"], config["edge_size"], 3)),
        tf.keras.layers.Rescaling(1. / 255),          # normalize pixel values from [0, 255] to [0, 1]
        tf.keras.layers.Conv2D(16, 3, padding='same', activation=config["activation_function"]),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation=config["activation_function"]),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation=config["activation_function"]),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),                    # flatten spatial feature maps into a 1-D vector
        tf.keras.layers.Dense(128, activation=config["activation_function"]),
        tf.keras.layers.Dense(num_classes)            # raw logits; softmax is applied by the loss function
    ])

    # SparseCategoricalCrossentropy expects integer class indices (not one-hot vectors).
    # from_logits=True means the final layer outputs raw scores rather than probabilities,
    # which is numerically more stable than applying softmax then computing cross-entropy.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if config["verbose"]:
        model.summary()

    # model training
    history = model.fit(
        train_ds,
        epochs=config["epochs"],  # Set epochs count
        validation_data=val_ds
    )

    return model, history
