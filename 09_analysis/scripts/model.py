import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_encoder(input_shape, latent_dim):
    encoder_inputs = keras.Input(shape=input_shape, name="encoder_input")
    x = encoder_inputs
    for ch in [2, 4, 8, 16, 32, 64, 128]:
        x = layers.Conv2D(
            ch,
            3,
            activation="relu",
            padding="same",
            strides=2,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def get_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(1 * 1 * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((1, 1, 128))(x)

    for ch in [128, 64, 32, 16, 8, 4, 2]:
        x = layers.Conv2DTranspose(ch, 3, activation="relu", padding="same", strides=2)(
            x
        )

    decoder_outputs = layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same", name="decoder_output"
    )(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)
