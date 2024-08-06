from tensorflow.keras.models import load_model

from flow_detection.src.config import set_output_path, set_supervisor_path
from flow_detection.src.data import get_train_val_data
from flow_detection.src.evaluation import predict_image_list, create_confusion_matrix, evaluate_model


def main():

    supervisor_path = set_supervisor_path("2020laptop")
    output_path = set_output_path("2020laptop")

    model_filename = output_path / "edge256batch128.keras"

    # load model from file
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = load_model(model_filename)

    print(model.summary())

    train_unbatched_ds, val_unbatched_ds = get_train_val_data(
        supervisor_path,
        0.20,
        123,
        256,
        None)

    labels, predictions = predict_image_list(val_unbatched_ds, model)
    confusion_matrix = create_confusion_matrix(labels, predictions)

    val_batched_ds = val_unbatched_ds.batch(32, drop_remainder=True)

    val_loss, val_acc = evaluate_model(val_batched_ds, model)


if __name__ == "__main__":
    main()
