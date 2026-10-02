import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image

sys.path.insert(0, '/Users/hikaru/Code/2026_Project/02_HandsOn/vae_pdb_sprit_angle')
from model import VAE, get_encoder, get_decoder

model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep'

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

# すべてのPDBに共通して存在しそうな基準角度を検索
target_angle_str = '_000_000_000_'

selected_paths = []
selected_pdbs = []

for pdb in unique_pdbs:
    idx = np.where(all_pdbids == pdb)[0]
    pdb_paths = all_paths[idx]
    
    # 完全に同じ角度（000_000_000など）を探す
    match = [p for p in pdb_paths if target_angle_str in p]
    if len(match) == 0:
        # 見つからない場合は最初の画像をとりあえず使う（本来は揃えたい）
        match = pdb_paths
        
    if len(match) > 0:
        # sn1.6 などの特定のSNを選ぶ
        sn_match = [p for p in match if 'sn1.6' in p]
        if len(sn_match) > 0:
            selected_paths.append(sn_match[0])
        else:
            selected_paths.append(match[0])
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

# Calculate diff (Original - Reconstructed, without absolute value)
diff_images = images - reconstructed

n = len(selected_pdbs)
plt.figure(figsize=(3 * n, 9))

# 色のスケールを揃えるための最大差分値
max_diff = np.max(np.abs(diff_images))

for i in range(n):
    # Original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title(f'{selected_pdbs[i]}\nOriginal', fontsize=14)
    plt.axis('off')
    
    # Reconstructed
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title('Reconstructed', fontsize=14)
    plt.axis('off')
    
    # Difference (Jet cmap is common for zero-centered difference)
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(diff_images[i].squeeze(), cmap='jet', vmin=-max_diff, vmax=max_diff)
    plt.title('Difference', fontsize=14)
    plt.axis('off')

plt.tight_layout()
output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'same_angle_diff_v2.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved to {output_path}')
