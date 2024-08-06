from flow_detection.src.config import create_config, get_train_val_data
from flow_detection.src.training import train_model
from flow_detection.src.evaluation import plot_history


def main():

    config = create_config("train_and_save_config.yaml")

    train_batched_ds, val_batched_ds = get_train_val_data(
        config["supervisor_path"],
        config["validation_split"],
        config["seed"],
        config["edge_size"],
        config["batch_size"]
    )

    model, history = train_model(config, train_batched_ds, val_batched_ds)

    if config["save_model"]:
        model.save(config["output_path"] / (config["trial_label"] + ".keras"))

    if config["plot_history"]:
        plot_history(config, history)


if __name__ == "__main__":
    main()
