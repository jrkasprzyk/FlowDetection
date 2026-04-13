# Main entry point for training a flow-detection CNN and saving the result.
# Configuration (paths, hyperparameters, flags) is read from model001.yaml
# located in the same directory as this script.

from FlowDetection.data import get_train_val_data
from FlowDetection.config import load_config
from FlowDetection.training import train_model
from FlowDetection.evaluation import plot_history


def main():

    config = load_config("model001.yaml")

    train_batched_ds, val_batched_ds = get_train_val_data(
        config["supervisor_path"],
        config["validation_split"],
        config["seed"],
        config["edge_size"],
        config["batch_size"]
    )

    # class_names are the subdirectory names under supervisor_path (one per flow class).
    # num_classes drives the size of the final Dense layer in the CNN.
    num_classes = len(train_batched_ds.class_names)

    model, history = train_model(config, train_batched_ds, val_batched_ds, num_classes)

    # Both save_model and plot_history are boolean flags in the YAML config,
    # allowing the same script to be used for quick test runs (flags=False) and
    # production training runs (flags=True) without editing code.
    if config["save_model"]:
        model.save(config["output_path"] / (config["trial_label"] + ".keras"))

    if config["plot_history"]:
        plot_history(config, history)


if __name__ == "__main__":
    main()
