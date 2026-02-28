import numpy
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from src.data import get_train_val_data
from src.config import create_config
#from src.training import train_model
from src.evaluation import plot_history


def create_model(edge_size=128):

    num_classes = 3

    model = Sequential([
        layers.Input(shape=(edge_size, edge_size, 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def main():

    # trying to implement:
    # https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594

    config = create_config("hyperparam_scikeras.yaml")

    train_batched_ds, val_batched_ds = get_train_val_data(
        config["supervisor_path"],
        config["validation_split"],
        config["seed"],
        config["edge_size"],
        config["batch_size"],
        config["shuffle"]
    )

    model_CV = KerasClassifier(
        model=create_model,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=1
    )

    edge_size = [64, 128, 256]
    param_grid = dict(edge_size=edge_size)
    grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3)

    #TODO: apparently only scikit-Learn compatible inputs are supported, so this won't work
    grid_result = grid.fit(train_batched_ds)

    #TODO: the values inside of grid_result may not work anymore
    print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')

    # since 'take' doesn't preserve class names, need to save these separately
    #class_names = train_batched_ds.class_names
    #num_classes = len(class_names)

    print("Training on 100 points")
    #model, history = train_model(config, train_batched_ds.take(100), val_batched_ds.take(100), num_classes)

    #if config["save_model"]:
    #    model.save(config["output_path"] / (config["trial_label"] + ".keras"))

    #if config["plot_history"]:
    #    plot_history(config, history)


if __name__ == "__main__":
    main()
