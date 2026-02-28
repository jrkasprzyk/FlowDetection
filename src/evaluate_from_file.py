from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

from src.config import set_output_path, set_supervisor_path
from src.data import get_train_val_data
from src.evaluation import predict_image_list, create_confusion_matrix, evaluate_model


def main():

    supervisor_path = set_supervisor_path("2020laptop")
    output_path = set_output_path("2020laptop")

    model_filename = output_path / "model001.keras"

    # load model from file
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = load_model(model_filename)

    print(model.summary())

    train_unbatched_ds, val_unbatched_ds = get_train_val_data(
        supervisor_path,
        0.20,
        234,
        128,
        None)

    #labels, predictions = predict_image_list(val_unbatched_ds, model)
    #confusion_matrix = create_confusion_matrix(labels, predictions)

    val_batched_ds = val_unbatched_ds.batch(24, drop_remainder=True)

    # In my testing, it seems like the evaluate function only works with batched
    # data, although maybe that has to do with how the model is originally trained?
    val_loss, val_acc = evaluate_model(val_batched_ds, model)

    print(f"{model_filename}: validation loss={val_loss}, validation accuracy={val_acc}")

    # TODO: can evaluate model manually:
    # https://stackoverflow.com/questions/66688040/calling-keras-model-evaluate-on-every-batch-element-separately


if __name__ == "__main__":
    main()
