import pathlib
from typing import Tuple

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras


def prepare_dataset(filename: str = "auto-mpg.data") -> pd.DataFrame:
    dataset_path = keras.utils.get_file(
        filename,
        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
    )
    column_names = [
        'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin',
    ]
    raw_dataset = pd.read_csv(
        dataset_path,
        names=column_names,
        na_values="?",
        comment='\t',
        sep=" ",
        skipinitialspace=True,
    )

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()

    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.
    dataset['Europe'] = (origin == 2) * 1.
    dataset['Japan'] = (origin == 3) * 1.
    return dataset


def split_dataset(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset


def get_labels(dataset: pd.DataFrame) -> pd.Series:
    return dataset.pop('MPG')


def norm(dataset: pd.DataFrame, train_stats: pd.DataFrame) -> pd.DataFrame:
    return (dataset - train_stats['mean']) / train_stats['std']


def get_train_stats(train_dataset: pd.DataFrame) -> pd.DataFrame:
    return train_dataset.describe().transpose()


def build_model(train_dataset: pd.DataFrame, layer_sizes: tuple = (64, 64)) -> keras.Model:
    layers = []
    for layer_size in layer_sizes:
        if layers:
            layers.append(keras.layers.Dense(64, activation='relu')),
        else:
            layers.append(keras.layers.Dense(
                layer_size, activation='relu', input_shape=[len(train_dataset.keys())],
            ))
    layers.append(keras.layers.Dense(1))

    model = keras.Sequential(layers)
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'],
    )
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 50 == 0:
            print('')
        print('.', end='')


def train_model(
        model: keras.Model,
        normed_train_data: pd.DataFrame,
        train_labels: pd.Series,
        epochs: int = 1000,
        validation: float = 0.2,
        early_stopping: int = 10,
):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)
    model = model.fit(
        normed_train_data,
        train_labels,
        epochs=epochs,
        validation_split=validation,
        verbose=0,
        callbacks=[PrintDot(), early_stop],
    )
    print('')
    return model


def plot_predict(model, normed_test_data, test_labels):
    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


@click.command()
@click.option('-e', '--epochs', default=1000, type=int)
@click.option('-v', '--validation', default=0.2, type=float)
@click.option('-s', '--early-stopping', default=10, type=int)
@click.option('--plot/--no-plot', default=False)
def main(epochs, validation, early_stopping, plot):
    dataset = prepare_dataset()
    train, test = split_dataset(dataset)
    train_labels, test_labels = get_labels(train), get_labels(test)
    train_stats = get_train_stats(train)
    train_normed, test_normed = norm(train, train_stats), norm(test, train_stats)
    model = build_model(train)
    history = train_model(
        model, train_normed, train_labels, epochs=epochs, validation=validation, early_stopping=early_stopping,
    )

    loss, mae, mse = model.evaluate(test_normed, test_labels, verbose=2)
    click.echo("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    if plot:
        plot_history(history)
        plot_predict(model, test_normed, test_labels)


if __name__ == '__main__':
    main()
