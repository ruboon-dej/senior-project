import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import re
import string

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from transformer_components import TransformerEncoder, TransformerDecoder, PositionalEmbedding

df = pd.read_csv("datasets/deepcut_dedup.csv")
with open("datasets/th_vocab-deepcut-dedup.txt", "r", encoding="utf8") as f:
    word_bag = f.read().split("\n")

text_pairs = df.values.tolist()
random.Random(10).shuffle(text_pairs)

num_val_samples = int(0.1 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "").replace("]", "")


def custom_standardization(input_string):
    return tf.strings.regex_replace(input_string, "[%s]" % re.escape(strip_chars), "")


vocab_size = len(word_bag)
sequence_length = 20
batch_size = 64
embed_dim = 256
dense_dim = 2048
num_heads = 8
epochs = 30

thai_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
thsl_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

train_thai_texts = [pair[0] for pair in train_pairs]
train_thsl_texts = [pair[1] for pair in train_pairs]
thai_vectorization.adapt(train_thai_texts)
thsl_vectorization.adapt(train_thsl_texts)


def format_dataset(thai, thsl):
    thai = thai_vectorization(thai)
    thsl = thsl_vectorization(thsl)
    return (
        {
            "encoder_inputs": thai,
            "decoder_inputs": thsl[:, :-1],
        },
        thsl[:, 1:],
    )


def make_dataset(pairs):
    thai_texts, thsl_texts = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(thai_texts), list(thsl_texts)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
transformer.summary()

checkpoint = keras.callbacks.ModelCheckpoint(
    "models/thai2thsl-deepcut-dedup.hdf5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[checkpoint])
