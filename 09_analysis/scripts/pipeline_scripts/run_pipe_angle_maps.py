import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

MODEL_DIR = sys.argv[1]
OUTPUT_DIR = os.path.join(sys.argv[2], "angle_maps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM = 10
TARGET_PDBS = ['1kqm', '1qvi', '2ec6']
TARGET_SN = 'sn1.6'

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

def build_decoder(latent_dim, output_channels=1):
    inp = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(1 * 1 * 128, activation='relu')(inp)
    x = tf.keras.layers.Reshape((1, 1, 128))(x)
    for filters in [128, 64, 32, 16, 8, 4, 2]:
        x = tf.keras.layers.Conv2DTranspose(filters, 3, activation='relu', padding='same', strides=2)(x)
    out = tf.keras.layers.Conv2DTranspose(output_channels, 3, activation='sigmoid', padding='same')(x)
    return tf.keras.Model(inp, out)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

def fix_paths(paths):
    fixed = []
    for p in paths:
        p_str = str(p)
        if p_str.startswith('/Volumes/hikaru/'): fixed.append(p_str.replace('/Volumes/hikaru/', '/Users/hikaru/', 1))
        elif p_str.startswith('/Users/hikarui./'): fixed.append(p_str.replace('/Users/hikarui./', '/Users/hikaru/', 1))
        else: fixed.append(p_str)
    return np.array(fixed)

def main():
    print("Running Angle Maps Generator...")
    encoder = build_encoder((128, 128, 1), LATENT_DIM)
    decoder = build_decoder(LATENT_DIM)
    vae = VAE(encoder, decoder)
    _ = vae(tf.zeros((1, 128, 128, 1)))
    vae.load_weights(os.path.join(MODEL_DIR, 'vae_best_weights.weights.h5'))

    kills_train = fix_paths(np.load(os.path.join(MODEL_DIR, 'kills_train.npy'), allow_pickle=True))
    kills_test = fix_paths(np.load(os.path.join(MODEL_DIR, 'kills_test.npy'), allow_pickle=True))
    all_paths = np.concatenate([kills_train, kills_test])

    angles = np.arange(0, 360, 10)
    img_size = 128

    for target_pdb in TARGET_PDBS:
        filtered_paths = [p for p in all_paths if target_pdb in p and ('sn1.6' in p or 'normal' in p)]
        if len(filtered_paths) < 100:
            pattern = f"/Users/hikaru/Code/2026_Project/05_Data/*/*_{target_pdb}_*/*normal*/*.tif"
            filtered_paths.extend(glob.glob(pattern))
            pattern2 = f"/Users/hikaru/Code/2026_Project/05_Data/*_{target_pdb}_*/*.tif"
            filtered_paths.extend(glob.glob(pattern2))
            filtered_paths = list(set([p for p in filtered_paths if 'sn1.6' in p or 'normal' in p]))

        path_dict = {}
        for p in filtered_paths:
            fname = os.path.basename(p)
            try:
                parts = fname.split('.')[0].split('_')
                if parts[0] in ('0', '1', '2'): x_angle, z_angle = int(parts[3]), int(parts[4])
                else: x_angle, z_angle = int(parts[2]), int(parts[3])
                path_dict[(x_angle, z_angle)] = p
            except Exception: pass
                
        canvas_original = np.zeros((36 * img_size, 36 * img_size))
        canvas_recon = np.zeros((36 * img_size, 36 * img_size))

        batch_images = []; batch_coords = []
        for r, z in enumerate(angles):
            for c, y in enumerate(angles):
                p = path_dict.get((y, z), None)
                if p is not None and os.path.exists(p):
                    with Image.open(p) as img:
                        arr = np.array(img).astype('float32') / 255.0
                        if len(arr.shape) == 2: arr = np.expand_dims(arr, axis=-1)
                        batch_images.append(arr); batch_coords.append((r, c))

        if len(batch_images) > 0:
            batch_images = np.array(batch_images)
            z_mean, _, _ = vae.encoder.predict(batch_images, batch_size=256, verbose=0)
            reconstructed = vae.decoder.predict(z_mean, batch_size=256, verbose=0)
            
            for i, (r, c) in enumerate(batch_coords):
                y_start = r * img_size; y_end = y_start + img_size
                x_start = c * img_size; x_end = x_start + img_size
                canvas_original[y_start:y_end, x_start:x_end] = batch_images[i].squeeze()
                canvas_recon[y_start:y_end, x_start:x_end] = reconstructed[i].squeeze()

        ticks_pos = np.arange(0, 36) * img_size + img_size // 2
        ticks_labels = [str(a) for a in angles]

        for canvas, title, filename in [
            (canvas_original, "ORIGINAL", f'angle_map_original_{target_pdb}.png'),
            (canvas_recon, "RECONSTRUCTED", f'angle_map_reconstruction_{target_pdb}.png')
        ]:
            plt.figure(figsize=(24, 24))
            plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
            plt.title(f'{target_pdb} ({TARGET_SN}) - {title}\nRows: Angle Z, Cols: Angle X', fontsize=24)
            plt.xticks(ticks_pos, ticks_labels, rotation=90, fontsize=12)
            plt.yticks(ticks_pos, ticks_labels, fontsize=12)
            plt.xlabel('Angle X', fontsize=18)
            plt.ylabel('Angle Z', fontsize=18)
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()
