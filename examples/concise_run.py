# prototyping fucntions for use within hyperparameter tuning and other experiments

# https://stackoverflow.com/questions/58663198/does-tf-data-dataset-take-return-random-sample
# in the above, ds.take(n) takes n samples; if shuffle=True, these samples are random

from tensorflow.keras.models import load_model
import pandas as pd

from src.config import set_output_path, set_supervisor_path
from src.data import get_train_val_data
from src.evaluation import predict_image_list, create_confusion_matrix, evaluate_model


def main():

    supervisor_path = set_supervisor_path("2020laptop")
    output_path = set_output_path("2020laptop")

    model_filename = output_path / "edge128batch128.keras"

    # load model from file
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = load_model(model_filename)

    print(model.summary())

    train_unbatched_ds, val_unbatched_ds = get_train_val_data(
        supervisor_path,
        validation_split=0.20,
        seed=123,
        image_size=128,
        batch_size=None,
        shuffle=True)

    print("Using ds.take() to predict a set of 100 values from the validation set")

    val_unbatched_ds_hundred = val_unbatched_ds.take(100)

    labels, predictions = predict_image_list(val_unbatched_ds_hundred, model)
    confusion_matrix = create_confusion_matrix(labels, predictions)

    # save a dataframe with relevant info
    # note: the 'take' function does not preserve the filenames so this is commented out
    # until we figure out a workaround
    image_list_results = pd.DataFrame({
                                    #'filename': val_unbatched_ds_hundred.file_paths,
                                       'true index': labels,
                                       'predicted index': predictions})
    image_list_results.to_excel('image_results_hundred.xlsx')
    print("finished writing image list results to excel")


if __name__ == "__main__":
    main()
