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

    num_classes = len(train_batched_ds.class_names)

    model, history = train_model(config, train_batched_ds, val_batched_ds, num_classes)

    if config["save_model"]:
        model.save(config["output_path"] / (config["trial_label"] + ".keras"))

    if config["plot_history"]:
        plot_history(config, history)


if __name__ == "__main__":
    main()
