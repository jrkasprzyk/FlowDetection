import numpy as np
from tensorflow.math import confusion_matrix
from tensorflow.data.experimental import ignore_errors

import matplotlib.pyplot as plt
import os


def predict_one_image(x, y, model):

    # this routine assumes that the true index is known

    # to handle the fact the model was trained with batches and thus expects the first dimension
    # to be 'None', see here:
    # https://stackoverflow.com/questions/60486437/add-none-dimension-in-tensorflow-2-0

    # also note that predict_class is deprecated in recent tensorflow,
    # so we use this np.argmax construction when predicting

    true_index = y.numpy()
    predicted_index = np.argmax(model.predict(x[None, :, :, :], verbose=0), axis=-1)
    return true_index, predicted_index


def predict_image_list(ds, model):

    # this routine assumes that the true index is known

    predictions = np.array([])
    labels = np.array([])

    for x, y in ds:
        true_index, predicted_index = predict_one_image(x, y, model)
        predictions = np.concatenate([predictions, predicted_index], axis=None)
        labels = np.concatenate([labels, true_index], axis=None)

    return labels, predictions


def predict_unlabeled_image_list(ds, model, filename='test.txt', start_ix=0):

    #predictions = np.empty([len(ds)])

    with open(filename, 'w') as f:
        # Write header
        f.write('index,year,month,day,time,camera,tag,prediction\n')

        #ds = ds.apply(ignore_errors()) # once you apply a filter, len() doesn't work anymore
        index = 0
        #TODO: incrementally save output file as predictions are being made
        for item in ds:
            print(f"i={index}")
            temp_filename = ds.file_paths[index]
            # Extract base filename without extension
            full_name, extension = os.path.splitext(temp_filename)  # base_name = "image_20231124_1035", extension = ".jpg"

            big_parts = full_name.split("\\")
            parts = big_parts[6].split("_")

            # Extract date and time information (if present)
            year = parts[0]
            month = parts[1]
            day = parts[2]
            time = parts[3]
            # the word "camera" is parts[4]
            camera_index = parts[5]
            image_tag = parts[6]

            temp_prediction = np.argmax(
                model.predict(
                    item[None, :, :, :],
                    verbose=0
                ),
                axis=-1
            )
            f.write(f"{index},"
                    f"{year},"
                    f"{month},"
                    f"{day},"
                    f"{time},"
                    f"{camera_index},"
                    f"{image_tag},"
                    f"{temp_prediction[0]}\n")
            index = index+1

    return


def create_confusion_matrix(labels, predictions):

    return confusion_matrix(labels=labels, predictions=predictions)


def evaluate_model(ds, model):

    #print("Evaluating model behavior...")
    loss, acc = model.evaluate(ds)
    #print("loss=", val_loss, " acc=", val_acc)

    return loss, acc


def plot_history(config, history, plot_filename=None):

    # TODO check that these hard-coded numbers are OK
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(config["epochs"])

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    if plot_filename is not None:
        plt.savefig(plot_filename)
