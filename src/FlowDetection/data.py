import tensorflow as tf


def get_train_val_data(supervisor_path, validation_split, seed, image_size, batch_size, shuffle=True, verbose=True):
    
    '''returns the training and validation datasets as tf.data.Dataset objects. 
    The datasets are created from the images in the supervisor folder. 
    The images are split into training and validation sets based on the validation_split parameter. 
    The images are resized to the specified image_size and batched according to the batch_size parameter. 
    The shuffle parameter determines whether the datasets are shuffled or not. 
    The verbose parameter determines whether to print out information about the datasets or not.'''
    
    # based on the latest keras training: https://www.tensorflow.org/guide/data
    # we can manipulate Dataset objects and perform transforms on them if we need them
    # so for example we can create a basic dataset and then augment it, batch it, etc.
    # later on
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        supervisor_path,
        validation_split=validation_split,  # typically 0.2, but made smaller to make this example quick
        subset="both",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=verbose
    )
    return train_ds, val_ds
