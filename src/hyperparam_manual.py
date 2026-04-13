# Manual hyperparameter search over image edge size (resolution).
# For each candidate value in edge_sizes, a fresh model is built and trained,
# and validation loss/accuracy are printed for comparison.
# This is a simpler alternative to the scikeras grid-search approach in
# hyperparam_scikeras.py and does not require scikit-learn compatibility.

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from src.data import get_train_val_data
from src.config import create_config
from src.evaluation import evaluate_model


def create_model(edge_size):
    '''Build and compile a CNN for the given square image edge size.

    num_classes is hard-coded to 3 here (matching the labeled dataset).
    The architecture mirrors train_model() in FlowDetection/training.py.
    '''
    num_classes = 3

    model = Sequential([
        layers.Input(shape=(edge_size, edge_size, 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def main():
    config = create_config("hyperparam_manual.yaml")

    # List of image resolutions to compare; each run trains a completely fresh model.
    edge_sizes = [128, 256]

    print(f"Constant params")
    print(f"val_split: {config["validation_split"]}")
    print(f"batch_size: {config["batch_size"]}")
    print(f"epochs: {config["epochs"]}")

    for edge_size in edge_sizes:

        print(f"Sampled Edge Size: {edge_size}")

        # Data must be reloaded for each edge_size because the images are resized
        # at load time by image_dataset_from_directory.
        train_batched_ds, val_batched_ds = get_train_val_data(
            supervisor_path=config["supervisor_path"],
            validation_split=config["validation_split"],
            seed=config["seed"],
            image_size=edge_size,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            verbose=False
        )

        model = create_model(edge_size)

        # model training
        history = model.fit(
            train_batched_ds,
            epochs=config["epochs"],  # Set epochs count
            validation_data=val_batched_ds,
            verbose=0
        )

        val_loss, val_acc = evaluate_model(val_batched_ds, model)
        print(f"val_loss={val_loss}, val_acc={val_acc}")


if __name__ == "__main__":
    main()
