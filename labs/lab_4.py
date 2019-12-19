import tensorflow as tf

import click
from skimage import transform
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from . import cli


def get_dataset():
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images[train_images >= .5] = 0.999
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 0.999
    test_images[test_images < .5] = 0.
    train_images = train_images[:5000]

    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    # return resize_batch(train_images), resize_batch(test_images)
    return train_images, test_images


def build_model(input_shape) -> keras.Model:
    model = tf.keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),

        keras.layers.Conv2D(28, kernel_size=5, strides=2, padding='same'),
        keras.layers.Conv2D(14, kernel_size=5, strides=2, padding='same'),
        keras.layers.Conv2D(7, kernel_size=5, strides=2, padding='same'),

        keras.layers.Conv2DTranspose(14, kernel_size=5, strides=1, padding='same', output_padding=0),
        keras.layers.Conv2DTranspose(28, kernel_size=5, strides=1, padding='same', output_padding=0),
        keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    print(model.summary())
    return model


def resize_batch(imgs):
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


def train_model(
        model: keras.Model,
        train_images: np.array,
        epochs: int = 5,
        batch_size: int = 500,
):
    history = model.fit(
        train_images,
        train_images,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    return history


@cli.command()
@click.option('-e', '--epochs', default=5, type=int)
@click.option('-b', '--batch-size', default=500, type=int)
@click.option('--plot/--no-plot', default=False)
def lab_4(epochs, batch_size, plot):
    click.echo("Get dataset...")
    train_images, test_images = get_dataset()

    click.echo("Build model...")
    model = build_model(input_shape=(28, 28, 1))

    click.echo("Training...")
    history = train_model(model, train_images, epochs=epochs, batch_size=batch_size)

    click.echo("Testing...")
    batch_img = test_images[:25]
    recon_img = model.predict(batch_img)
    loss, acc = model.evaluate(test_images, test_images, verbose=0)
    click.echo("Testing set loss/acc: {:5.2f} / {:5.2f}".format(loss, acc))

    if plot:
        plt.figure(1)
        plt.title('Reconstructed Images')
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(recon_img[i, ..., 0], cmap='gray')
        plt.figure(2)
        plt.title('Input Images')
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(batch_img[i, ..., 0], cmap='gray')
        plt.show()
