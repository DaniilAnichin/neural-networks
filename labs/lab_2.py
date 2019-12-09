import re
import tarfile
from enum import Enum
from functools import reduce
from typing import Tuple, List

import click
import numpy as np

from tensorflow import keras

from . import cli


def tokenize(sent):
    """Return the tokens of a sentence including punctuation."""
    return [x.strip() for x in re.split(r"(\W+)", sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode("utf-8").strip()
        nid, line = line.split(" ", 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if "\t" in line:
            q, a, supporting = line.split("\t")
            q = tokenize(q)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append("")
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(file, only_supporting: bool = False, max_length: int = None):
    """Given a file, read it, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(file.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [
        (flatten(story), q, answer)
        for story, q, answer in data
        if not max_length or len(flatten(story)) < max_length
    ]
    return data


class Challenge(Enum):
    QA1 = "tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt"
    QA1B = "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt"
    QA2 = "tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt"
    QA2B = "tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt"


def get_split_dataset(
        filename: str = "babi-tasks-v1-2.tar.gz", challenge: Challenge = Challenge.QA2,
) -> Tuple[List[tuple], List[tuple]]:
    path = keras.utils.get_file(
        filename,
        origin="https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
    )
    with tarfile.open(path) as tar:
        train = get_stories(tar.extractfile(challenge.value.format("train")))
        test = get_stories(tar.extractfile(challenge.value.format("test")))
    return train, test


def get_vocab(dataset: List[tuple]):
    vocab = set()
    for story, q, answer in dataset:
        vocab |= set(story + q + [answer])
    return sorted(vocab)


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let"s not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (
        keras.preprocessing.sequence.pad_sequences(xs, maxlen=story_maxlen),
        keras.preprocessing.sequence.pad_sequences(xqs, maxlen=query_maxlen),
        np.array(ys),
    )


def build_model(rnn, story_maxlen, vocab_size, embed_size, sent_size, query_maxlen, query_size) -> keras.Model:
    sentence = keras.layers.Input(shape=(story_maxlen,), dtype="int32")
    encoded_sentence = keras.layers.Embedding(vocab_size, embed_size)(sentence)
    encoded_sentence = rnn(sent_size)(encoded_sentence)

    question = keras.layers.Input(shape=(query_maxlen,), dtype="int32")
    encoded_question = keras.layers.Embedding(vocab_size, embed_size)(question)
    encoded_question = rnn(query_size)(encoded_question)

    merged = keras.layers.concatenate([encoded_sentence, encoded_question])
    preds = keras.layers.Dense(vocab_size, activation="softmax")(merged)

    model = keras.Model([sentence, question], preds)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@cli.command()
@click.option("-e", "--epochs", default=5, type=int)
@click.option("-v", "--validation", default=0.05, type=float)
@click.option("-s", "--early-stopping", default=2, type=int)
@click.option("-em", "--embed-size", default=50, type=int)
@click.option("-se", "--sent-size", default=100, type=int)
@click.option("-q", "--query-size", default=100, type=int)
@click.option("-b", "--batch-size", default=32, type=int)
def lab_2(epochs, validation, early_stopping, embed_size, sent_size, query_size, batch_size):
    rnn = keras.layers.LSTM
    click.echo("RNN / Embed / Sent / Query = {}, {}, {}, {}".format(
        rnn,
        embed_size,
        sent_size,
        query_size,
    ))

    train, test = get_split_dataset()
    vocab = get_vocab(train + test)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    click.echo("vocab = {}".format(vocab))
    click.echo("x.shape = {}".format(x.shape))
    click.echo("xq.shape = {}".format(xq.shape))
    click.echo("y.shape = {}".format(y.shape))
    click.echo("story_maxlen, query_maxlen = {}, {}".format(story_maxlen, query_maxlen))

    click.echo("Build model...")
    model = build_model(rnn, story_maxlen, vocab_size, embed_size, sent_size, query_maxlen, query_size)

    click.echo("Training")
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)
    model.fit(
        [x, xq], y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation,
        callbacks=[early_stop, ],
    )

    click.echo("Evaluation")
    loss, acc = model.evaluate(
        [tx, txq], ty, batch_size=batch_size, verbose=0,
    )
    click.echo("Test loss / test accuracy = {:.4f} / {:.4f}".format(loss, acc))
