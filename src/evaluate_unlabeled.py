from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data.experimental import ignore_errors
import pandas as pd

from src.config import set_output_path, set_custom_path
from src.data import get_train_val_data
from src.evaluation import predict_unlabeled_image_list, create_confusion_matrix, evaluate_model


def main():

    #supervisor_path = set_supervisor_path("2020laptop")
    image_path = set_custom_path("C:/Users/josep/OneDrive - UCB-O365/Datasets/flow_detection_unlabeled_camera_b")
    ds = image_dataset_from_directory(
        image_path,
        labels=None,
        batch_size=None,
        image_size=(128,128),
        shuffle=False,
        verbose=True
    )
    print(f"At first, dataset contains {len(ds)} entries")

    #ds = ds.apply(ignore_errors())

    #print(f"After the ignore_errors subroutine, ds has {len(ds)} entries")

    output_path = set_output_path("2020laptop")
    model_filename = output_path / "model001.keras"

    # load model from file
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = load_model(model_filename)

    print(model.summary())

    predictions = predict_unlabeled_image_list(ds, model, filename=output_path / "model001_unlabeled_camera_b")

    #confusion_matrix = create_confusion_matrix(labels, predictions)

    # save a dataframe with val_unbatched_ds.file_paths, labels (true index) and predictions (predicted index)
    #image_list_results = pd.DataFrame({'filename': ds.file_paths[0:len(predictions)],
    #                                   'predicted index': predictions})
    #image_list_results.to_excel('unlabeled_image_results_2023.xlsx')


if __name__ == "__main__":
    main()
