import tensorflow as tf
from tensorflow import keras


class SequenceModel(keras.Model):
    def __init__(
        self, vocab_size: int, embedding_dim: int, recurrent_size: int, hidden_size: int
    ):
        super(SequenceModel, self).__init__()

        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.gru = keras.layers.GRU(units=recurrent_size, return_sequences=True)
        self.hidden = keras.layers.Dense(units=hidden_size, activation="relu")
        self.output_layer = keras.layers.Dense(units=vocab_size, activation="softmax")

    def call(self, x):
        x = self.embedding(x)
        x = self.gru(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        return x
