import numpy as np
from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt


def predict_one_image(x, y, model):

    # to handle the fact the model was trained with batches and thus expects the first dimension
    # to be 'None', see here:
    # https://stackoverflow.com/questions/60486437/add-none-dimension-in-tensorflow-2-0

    # also note that predict_class is deprecated in recent tensorflow,
    # so we use this np.argmax construction when predicting

    true_index = y.numpy()
    predicted_index = np.argmax(model.predict(x[None, :, :, :], verbose=0), axis=-1)
    return true_index, predicted_index


def predict_image_list(ds, model):

    predictions = np.array([])
    labels = np.array([])

    for x, y in ds:
        true_index, predicted_index = predict_one_image(x, y, model)
        predictions = np.concatenate([predictions, predicted_index], axis=None)
        labels = np.concatenate([labels, true_index], axis=None)

    return labels, predictions


def create_confusion_matrix(labels, predictions):

    return confusion_matrix(labels=labels, predictions=predictions)


def evaluate_model(ds, model):

    #print("Evaluating model behavior...")
    loss, acc = model.evaluate(ds)
    #print("loss=", val_loss, " acc=", val_acc)

    return loss, acc


def plot_history(config, history):

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
