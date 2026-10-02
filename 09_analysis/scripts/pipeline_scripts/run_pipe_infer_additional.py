import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_DIR = sys.argv[1]

LATENT_DIM = 10

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        eps = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_encoder(input_shape, latent_dim):
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    for filters in [2, 4, 8, 16, 32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', strides=2, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inp, [z_mean, z_log_var, z])

def fix_paths(paths):
    fixed = []
    for p in paths:
        p_str = str(p)
        if p_str.startswith('/Volumes/hikaru/'): fixed.append(p_str.replace('/Volumes/hikaru/', '/Users/hikaru/', 1))
        elif p_str.startswith('/Users/hikarui./'): fixed.append(p_str.replace('/Users/hikarui./', '/Users/hikaru/', 1))
        else: fixed.append(p_str)
    return np.array(fixed)

def get_sn16_path(p_str):
    if '/normal/' not in p_str: return None
    new_p = p_str.replace('/normal/', '/noise/sn1.6/')
    new_p = new_p.replace('.tif', '_sn1.6.tif')
    return new_p

def infer_and_save(prefix):
    print(f"Running inference for {prefix} sn1.6...")
    kills_path = os.path.join(MODEL_DIR, f"kills_{prefix}.npy")
    if not os.path.exists(kills_path): return
    
    kills = fix_paths(np.load(kills_path, allow_pickle=True))
    
    valid_indices = []
    valid_paths = []
    valid_images = []
    
    for i, p in enumerate(kills):
        sn16_p = get_sn16_path(p)
        if sn16_p and os.path.exists(sn16_p):
            try:
                with Image.open(sn16_p) as img:
                    arr = np.array(img).astype('float32') / 255.0
                    if len(arr.shape) == 2: arr = np.expand_dims(arr, axis=-1)
                    valid_images.append(arr)
                    valid_paths.append(sn16_p)
                    valid_indices.append(i)
            except Exception as e:
                pass
                
    if len(valid_images) == 0:
        print(f"No sn1.6 images found for {prefix}.")
        return
        
    valid_images = np.array(valid_images)
    encoder = build_encoder((128, 128, 1), LATENT_DIM)
    # Load weights into the encoder directly from the vae weights
    # Note: VAE weights are usually saved for the whole model. We can build VAE and load weights.
    
    decoder_inp = tf.keras.Input(shape=(LATENT_DIM,))
    decoder_out = tf.keras.layers.Dense(1)(decoder_inp) # dummy
    decoder = tf.keras.Model(decoder_inp, decoder_out)
    
    class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder; self.decoder = decoder
        def call(self, inputs):
            _, _, z = self.encoder(inputs)
            return self.decoder(z)
            
    # Dummy decoder for loading weights
    def build_decoder(latent_dim, output_channels=1):
        inp = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(1 * 1 * 128, activation='relu')(inp)
        x = tf.keras.layers.Reshape((1, 1, 128))(x)
        for filters in [128, 64, 32, 16, 8, 4, 2]:
            x = tf.keras.layers.Conv2DTranspose(filters, 3, activation='relu', padding='same', strides=2)(x)
        out = tf.keras.layers.Conv2DTranspose(output_channels, 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(inp, out)

    vae = VAE(encoder, build_decoder(LATENT_DIM))
    _ = vae(tf.zeros((1, 128, 128, 1)))
    vae.load_weights(os.path.join(MODEL_DIR, 'vae_best_weights.weights.h5'))
    
    print(f"Inferring {len(valid_images)} images...")
    z_mean, _, _ = vae.encoder.predict(valid_images, batch_size=256)
    
    np.save(os.path.join(MODEL_DIR, f"z_mean_{prefix}_sn1.6.npy"), z_mean)
    np.save(os.path.join(MODEL_DIR, f"kills_{prefix}_sn1.6.npy"), valid_paths)
    np.save(os.path.join(MODEL_DIR, f"indices_{prefix}_sn1.6.npy"), valid_indices)
    print(f"Saved {prefix} sn1.6 inference results.")

if __name__ == "__main__":
    # Check if already inferred
    if os.path.exists(os.path.join(MODEL_DIR, "z_mean_test_sn1.6.npy")):
        print("sn1.6 inference already done.")
    else:
        infer_and_save("train")
        infer_and_save("test")
