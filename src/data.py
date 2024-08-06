from tensorflow.keras.utils import image_dataset_from_directory


def get_train_val_data(supervisor_path, validation_split, seed, image_size, batch_size):
    # based on the latest keras training: https://www.tensorflow.org/guide/data
    # we can manipulate Dataset objects and perform transforms on them if we need them
    # so for example we can create a basic dataset and then augment it, batch it, etc.
    # later on
    train_ds, val_ds = image_dataset_from_directory(
        supervisor_path,
        validation_split=validation_split,  # typically 0.2, but made smaller to make this example quick
        subset="both",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )
    return train_ds, val_ds
