import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

MODEL_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running Reconstruction Grid...")

PDB_IDS = ["1kk7", "1kk8", "1kqm", "1kwo", "1l2o", "1qvi", "1s5g", "1sr6", "2ec6"]
TARGET_SN = 'sn1.6'
TARGET_ANGLES = '_000_000_000_'
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
    inp = tf.keras.Input(shape=input_shape, name="encoder_input")
    x = inp
    for filters in [2, 4, 8, 16, 32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', strides=2, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inp, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim, output_channels=1):
    inp = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
    x = tf.keras.layers.Dense(1 * 1 * 128, activation='relu')(inp)
    x = tf.keras.layers.Reshape((1, 1, 128))(x)
    for filters in [128, 64, 32, 16, 8, 4, 2]:
        x = tf.keras.layers.Conv2DTranspose(filters, 3, activation='relu', padding='same', strides=2)(x)
    out = tf.keras.layers.Conv2DTranspose(output_channels, 3, activation='sigmoid', padding='same', name='decoder_output')(x)
    return tf.keras.Model(inp, out, name="decoder")

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

encoder = build_encoder((128, 128, 1), LATENT_DIM)
decoder = build_decoder(LATENT_DIM)
vae = VAE(encoder, decoder)
_ = vae(tf.zeros((1, 128, 128, 1)))
vae.load_weights(os.path.join(MODEL_DIR, 'vae_best_weights.weights.h5'))

def fix_paths(paths):
    fixed = []
    for p in paths:
        p_str = str(p)
        if p_str.startswith('/Volumes/hikaru/'):
            fixed.append(p_str.replace('/Volumes/hikaru/', '/Users/hikaru/', 1))
        elif p_str.startswith('/Users/hikarui./'):
            fixed.append(p_str.replace('/Users/hikarui./', '/Users/hikaru/', 1))
        else:
            fixed.append(p_str)
    return np.array(fixed)

kills_train = fix_paths(np.load(os.path.join(MODEL_DIR, "kills_train.npy"), allow_pickle=True))
kills_test = fix_paths(np.load(os.path.join(MODEL_DIR, "kills_test.npy"), allow_pickle=True))
all_paths = np.concatenate([kills_train, kills_test])

selected_images = []
found_pdbs = []

for pdb in PDB_IDS:
    candidates = [p for p in all_paths if f"_{pdb}_" in p and ('sn1.6' in p or 'normal' in p) and TARGET_ANGLES in p]
    if not candidates:
        candidates = [p for p in all_paths if f"_{pdb}_" in p and ('sn1.6' in p or 'normal' in p)]
    if candidates:
        selected_images.append(candidates[0])
        found_pdbs.append(pdb)
        print(f"Found exactly matched image for {pdb}: {candidates[0]}")
    else:
        print(f"Warning: Exact match not found for {pdb}. Skipping.")

if not selected_images:
    print("No images found. Exiting.")
    sys.exit(0)

orig_imgs = []
for p in selected_images:
    with Image.open(p) as img:
        arr = np.array(img).astype('float32') / 255.0
        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=-1)
        orig_imgs.append(arr)

orig_imgs = np.array(orig_imgs)
z_mean, _, _ = vae.encoder.predict(orig_imgs, batch_size=32)
recon_imgs = vae.decoder.predict(z_mean, batch_size=32)

fig, axes = plt.subplots(3, len(found_pdbs), figsize=(3 * len(found_pdbs), 9))

for i, pdb in enumerate(found_pdbs):
    orig = orig_imgs[i].squeeze()
    recon = recon_imgs[i].squeeze()
    diff = np.abs(orig - recon)
    
    # Original
    ax = axes[0, i]
    im = ax.imshow(orig, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"{pdb} Orig", fontsize=14)
    ax.axis('off')
    
    # Reconstruction
    ax = axes[1, i]
    im2 = ax.imshow(recon, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Recon", fontsize=14)
    ax.axis('off')
    
    # Diff
    ax = axes[2, i]
    im3 = ax.imshow(diff, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f"Diff", fontsize=14)
    ax.axis('off')

# Colorbars
cbar_ax_orig = fig.add_axes([0.92, 0.66, 0.015, 0.22])
fig.colorbar(im, cax=cbar_ax_orig)
cbar_ax_recon = fig.add_axes([0.92, 0.39, 0.015, 0.22])
fig.colorbar(im2, cax=cbar_ax_recon)
cbar_ax_diff = fig.add_axes([0.92, 0.12, 0.015, 0.22])
fig.colorbar(im3, cax=cbar_ax_diff)

plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)
out_path = os.path.join(OUTPUT_DIR, 'reconstruction_diff_grid.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {out_path}")
