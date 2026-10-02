import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image

sys.path.insert(0, '/Users/hikaru/Code/2026_Project/02_HandsOn/vae_pdb_sprit_angle')
from model import VAE, get_encoder, get_decoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PlotAllPdbRecon')

model_dir = sys.argv[1]
output_dir = sys.argv[2]

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

kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
kills_test = fix_paths(np.load(os.path.join(model_dir, 'kills_test.npy'), allow_pickle=True))
pdbids_train = np.load(os.path.join(model_dir, 'pdbids_train.npy'), allow_pickle=True)
pdbids_test = np.load(os.path.join(model_dir, 'pdbids_test.npy'), allow_pickle=True)

all_paths = np.concatenate([kills_train, kills_test])
all_pdbids = np.concatenate([pdbids_train, pdbids_test])

unique_pdbs = np.unique(all_pdbids)
selected_paths = []
selected_pdbs = []

for pdb in unique_pdbs:
    idx = np.where(all_pdbids == pdb)[0]
    if len(idx) > 0:
        selected_paths.append(all_paths[idx[0]])
        selected_pdbs.append(pdb)

images = []
for p in selected_paths:
    with Image.open(p) as img:
        arr = np.array(img).astype('float32') / 255.0
        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=-1)
        images.append(arr)
images = np.array(images)

sys.path.insert(0, '/Users/hikaru/Code/2026_Project/06_Analysis')
from model import VAE, get_encoder, get_decoder

encoder = get_encoder((128, 128, 1), 10)
decoder = get_decoder(10)
vae = VAE(encoder, decoder)
vae.build((None, 128, 128, 1))
vae.load_weights(os.path.join(model_dir, 'vae_best_weights.weights.h5'))

z_mean, _, _ = vae.encoder.predict(images)
reconstructed = vae.decoder.predict(z_mean)

n = len(selected_pdbs)
plt.figure(figsize=(2 * n, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(selected_pdbs[i])
    plt.axis('off')
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    plt.title('Recon')
    plt.axis('off')

plt.tight_layout()
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'all_pdb_reconstructions.png'))
