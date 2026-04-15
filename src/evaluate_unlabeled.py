# Script to run a trained model against a folder of unlabeled images and write
# predictions to a CSV file.  Unlike the labeled evaluation scripts, there is no
# ground-truth class to compare against, so no confusion matrix is computed.

from FlowDetection.runtime import configure_tensorflow_environment

configure_tensorflow_environment()

import tensorflow as tf
import pandas as pd

from FlowDetection.config import set_output_path, set_custom_path
from FlowDetection.data import get_train_val_data
from FlowDetection.evaluation import predict_unlabeled_image_list, create_confusion_matrix, evaluate_model


def main():

    #supervisor_path = set_supervisor_path("2020laptop")
    image_path = set_custom_path("C:/Users/joka0958/OneDrive - UCB-O365/Datasets/flow_detection_unlabeled_camera_b")
    # labels=None because the images have no ground-truth class subdirectories.
    # shuffle=False preserves filename order so that ds.file_paths[index] aligns
    # with the iteration order inside predict_unlabeled_image_list.
    ds = tf.keras.utils.image_dataset_from_directory(
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

    model_filename = set_custom_path("C:/GitHub/FlowDetection/models/model001.keras")

    # load model from file
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = tf.keras.models.load_model(model_filename)

    print(model.summary())

    # predict the class index for each image in the dataset and write to output.txt
    # the output file will have one line per image, with the predicted class index (0, 1, or 2) for that image.
    # the filename of the output file is set by set_custom_path, and will be created in the same directory as this script.
    # the str() around set_custom_path is needed because set_custom_path returns a pathlib.Path object, 
    # but predict_unlabeled_image_list expects a string filename.
    predictions = predict_unlabeled_image_list(ds, 
                                               model, 
                                               filename=str(set_custom_path("C:/GitHub/FlowDetection/src/output.txt")))

    #confusion_matrix = create_confusion_matrix(labels, predictions)

    # save a dataframe with val_unbatched_ds.file_paths, labels (true index) and predictions (predicted index)
    #image_list_results = pd.DataFrame({'filename': ds.file_paths[0:len(predictions)],
    #                                   'predicted index': predictions})
    #image_list_results.to_excel('unlabeled_image_results_2023.xlsx')


if __name__ == "__main__":
    main()
