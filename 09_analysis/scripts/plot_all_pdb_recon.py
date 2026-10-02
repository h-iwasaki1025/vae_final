import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image

import sys
sys.path.append('/Users/hikaru/Code/2026_Project/02_HandsOn/vae_pdb_sprit_angle')
from model import VAE, build_encoder, build_decoder
import config as vae_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('PlotAllPdbRecon')

# Config loaded
with open('/Users/hikaru/Code/2026_Project/06_Analysis/config.json', 'r') as f:
    config = json.load(f)

ENCODER_MODEL_PATH = config['ENCODER_MODEL_PATH']
model_dir = os.path.dirname(ENCODER_MODEL_PATH)

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

logger.info(f'Loading paths from {model_dir}')
kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
kills_test = fix_paths(np.load(os.path.join(model_dir, 'kills_test.npy'), allow_pickle=True))
pdbids_train = np.load(os.path.join(model_dir, 'pdbids_train.npy'), allow_pickle=True)
pdbids_test = np.load(os.path.join(model_dir, 'pdbids_test.npy'), allow_pickle=True)

all_paths = np.concatenate([kills_train, kills_test])
all_pdbids = np.concatenate([pdbids_train, pdbids_test])

unique_pdbs = np.unique(all_pdbids)
logger.info(f'Found {len(unique_pdbs)} unique PDBs: {unique_pdbs}')

selected_paths = []
selected_pdbs = []

for pdb in unique_pdbs:
    idx = np.where(all_pdbids == pdb)[0]
    if len(idx) > 0:
        selected_paths.append(all_paths[idx[0]])
        selected_pdbs.append(pdb)

logger.info(f'Loading images...')
images = []
for p in selected_paths:
    with Image.open(p) as img:
        arr = np.array(img).astype('float32') / 255.0
        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=-1)
        images.append(arr)

images = np.array(images)

logger.info(f'Building VAE model and loading weights from {ENCODER_MODEL_PATH}')
encoder = build_encoder(vae_config.LATENT_DIM)
decoder = build_decoder(vae_config.LATENT_DIM)
vae = VAE(encoder, decoder, beta=100.0) # Using BETA=100.0 as previously set
vae.build((None, 128, 128, 1))
vae.load_weights(ENCODER_MODEL_PATH)

logger.info('Reconstructing images...')
z_mean, z_log_var, z = vae.encoder.predict(images)
reconstructed = vae.decoder.predict(z_mean) # using mean for pure reconstruction

# Plotting
n = len(selected_pdbs)
plt.figure(figsize=(2 * n, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(selected_pdbs[i])
    plt.axis('off')
    
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    plt.title('Recon')
    plt.axis('off')

plt.tight_layout()
output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'all_pdb_reconstructions.png')
plt.savefig(output_path)
logger.info(f'Saved to {output_path}')
