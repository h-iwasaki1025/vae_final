import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================================
# 設定
# ==========================================================
MODEL_DIR      = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep"
OUTPUT_DIR     = "/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep"
LATENT_DIM     = 10
PIXEL_SIZE     = 128
PDB_IDS        = ["1kk7", "1kk8", "1kqm", "1kwo", "1l2o", "1qvi", "1s5g", "1sr6", "2ec6"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# VAE モデル定義
# ==========================================================
@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_encoder(input_shape, latent_dim):
    inp = tf.keras.Input(shape=input_shape, name="encoder_input")
    x = inp
    for filters in [2, 4, 8, 16, 32, 64, 128]:
        x = tf.keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same',
            strides=2, kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x       = tf.keras.layers.Flatten()(x)
    x       = tf.keras.layers.Dense(16, activation='relu')(x)
    z_mean    = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inp, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim, output_channels=1):
    inp = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
    x   = tf.keras.layers.Dense(1 * 1 * 128, activation='relu')(inp)
    x   = tf.keras.layers.Reshape((1, 1, 128))(x)
    
    for filters in [128, 64, 32, 16, 8, 4, 2]:
        x = tf.keras.layers.Conv2DTranspose(
            filters, 3, activation='relu', padding='same', strides=2
        )(x)
    out = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, activation='sigmoid',
        padding='same', name='decoder_output'
    )(x)
    return tf.keras.Model(inp, out, name="decoder")

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

def load_image(path):
    # パス変換 (/Volumes/hikaru/ -> /Users/hikaru/)
    path = str(path).replace('/Volumes/hikaru/', '/Users/hikaru/')
    path = str(path).replace('/Users/hikarui./', '/Users/hikaru/')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        arr = np.asarray(img.convert('L')).astype('float32') / 255.0
    return arr[..., np.newaxis]

def get_angle0_image_path(pdb_id, kills_train, kills_test):
    # Trainから探す
    for p in kills_train:
        if f"_{pdb_id}_" in str(p) and "_000_000_000_" in str(p) and "sn1.6" in str(p):
            return str(p)
    # Testから探す
    for p in kills_test:
        if f"_{pdb_id}_" in str(p) and "_000_000_000_" in str(p) and "sn1.6" in str(p):
            return str(p)
            
    # kills_train/test内にsn1.6で000_000_000のものがない場合は、元の画像ディレクトリから直接探す
    pattern = f"/Users/hikaru/Code/2026_Project/05_Data/*/*_{pdb_id}_*/noise/sn1.6/*_{pdb_id}_000_000_000_sn1.6.tif"
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
        
    return None

def main():
    print("Loading VAE model...")
    input_shape = (128, 128, 1)
    
    # 構造定義 (ここでは簡略化。重みファイルのロードに失敗したら構造を合わせる)
    encoder = build_encoder(input_shape, LATENT_DIM)
    decoder = build_decoder(LATENT_DIM)
    vae     = VAE(encoder, decoder)
    
    dummy = tf.zeros((1, 128, 128, 1))
    _ = vae(dummy)
    
    weight_path = os.path.join(MODEL_DIR, "vae_best_weights.weights.h5")
    vae.load_weights(weight_path)
    print("Model loaded successfully.")

    # Load paths
    kills_train = np.load(os.path.join(MODEL_DIR, "kills_train.npy"), allow_pickle=True)
    kills_test  = np.load(os.path.join(MODEL_DIR, "kills_test.npy"), allow_pickle=True)

    imgs = []
    found_pdbs = []
    
    for pdb in PDB_IDS:
        path = get_angle0_image_path(pdb, kills_train, kills_test)
        if path is not None:
            imgs.append(load_image(path))
            found_pdbs.append(pdb)
        else:
            print(f"Warning: No Angle0 image found for {pdb}")
    
    imgs = np.array(imgs) # (N, 64, 64, 1)
    
    # Predict
    z_mean, _, _ = vae.encoder.predict(imgs, verbose=0)
    recons = vae.decoder.predict(z_mean, verbose=0)
    
    # Plot
    n = len(found_pdbs)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    for i in range(n):
        orig = imgs[i, :, :, 0]
        recon = recons[i, :, :, 0]
        diff = orig - recon
        
        # Original
        ax = axes[0, i]
        ax.imshow(orig, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f"{found_pdbs[i]}\nOriginal", fontsize=14)
        
        # Reconstructed
        ax = axes[1, i]
        ax.imshow(recon, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title("Reconstructed", fontsize=14)
        
        # Difference
        ax = axes[2, i]
        # vmin, vmax を適度に設定するか、自動スケールに任せる
        ax.imshow(diff, cmap='jet')
        ax.axis('off')
        ax.set_title("Difference", fontsize=14)

    save_path = os.path.join(OUTPUT_DIR, "reconstruction_diff_grid.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved grid to {save_path}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
