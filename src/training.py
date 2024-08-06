from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def train_model(config, train_ds, val_ds):

    num_classes = len(val_ds.class_names)

    model = Sequential([
        layers.Input(shape=(config["edge_size"], config["edge_size"], 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation=config["activation_function"]),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation=config["activation_function"]),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=config["activation_function"]),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation=config["activation_function"]),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
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
