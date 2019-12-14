from typing import Tuple

import tensorflow as tf

import click
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from . import cli


def get_split_dataset() -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def build_model() -> keras.Model:
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


class PrintSlash(keras.callbacks.Callback):
    def on_epoch_start(self, epoch, logs):
        print('/', end='')

    def on_epoch_end(self, epoch, logs):
        print('\\', end='')

    def on_train_end(self, logs=None):
        print('')


def train_model(
        model: keras.Model,
        train_images: np.array,
        train_labels: np.array,
        epochs: int = 10,
        validation: float = 0.1,
        batch_size: int = 64,
):
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        validation_split=validation,
        verbose=0,
        callbacks=[PrintSlash()],
        batch_size=batch_size,
    )

    return history


# @click.option("-e", "--epochs", default=5, type=int)
# @click.option("-v", "--validation", default=0.05, type=float)
# @click.option("-s", "--early-stopping", default=2, type=int)
# @click.option("-em", "--embed-size", default=50, type=int)
# @click.option("-se", "--sent-size", default=100, type=int)
# @click.option("-q", "--query-size", default=100, type=int)
# @click.option("-b", "--batch-size", default=32, type=int)


@cli.command()
@click.option('-e', '--epochs', default=1000, type=int)
@click.option('-v', '--validation', default=0.2, type=float)
@click.option('-l', '--layer', 'layers', multiple=True, default=[64, 64])
@click.option('--plot/--no-plot', default=False)
def lab_3(epochs, validation, layers, plot):
    (train_images, train_labels), (test_images, test_labels) = get_split_dataset()
    click.echo("Build model...")
    model = build_model()
    click.echo("Training")
    history = train_model(
        model, train_images, train_labels, epochs=epochs, validation=validation,
    )

    click.echo("Evaluation")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    click.echo("Test loss / test accuracy = {:.4f} / {:.4f}".format(test_loss, test_acc))

    if plot:
        plot_history(history)
        plot_predict(model, test_images, test_labels)
