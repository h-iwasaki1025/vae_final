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
target_pdb = '1kqm'
target_angle = '_000_000_000_'
target_sn = 'sn1.6'

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

kills_test = fix_paths(np.load(os.path.join(model_dir, 'kills_test.npy'), allow_pickle=True))

# 基準となる画像を1枚探す
base_path = None
all_paths = np.concatenate([fix_paths(np.load(os.path.join(model_dir, "kills_train.npy"), allow_pickle=True)), kills_test])
for p in all_paths:
    if target_pdb in p and target_angle in p and target_sn in p:
        base_path = p
        break

if base_path is None:
    base_path = kills_test[0]

print(f'Base image: {base_path}')

with Image.open(base_path) as img:
    arr = np.array(img).astype('float32') / 255.0
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=-1)
base_img = np.expand_dims(arr, axis=0) # shape: (1, 128, 128, 1)

sys.path.insert(0, '/Users/hikaru/Code/2026_Project/06_Analysis')
from model import VAE, get_encoder, get_decoder

encoder = get_encoder((128, 128, 1), 10)
decoder = get_decoder(10)
vae = VAE(encoder, decoder)
vae.build((None, 128, 128, 1))
vae.load_weights(os.path.join(model_dir, 'vae_best_weights.weights.h5'))

# 基準となる潜在変数 z を取得
z_base_mean, _, _ = vae.encoder.predict(base_img)
z_base = z_base_mean[0] # shape: (10,)

# 走査する範囲とステップ数
n_steps = 11
z_range = np.linspace(-3.0, 3.0, n_steps)

# 10次元すべてについて走査
latent_dim = 10
canvas = np.zeros((latent_dim * 128, n_steps * 128))

# バッチ処理用
batch_z = []
coords = []

for dim in range(latent_dim):
    for step_idx, val in enumerate(z_range):
        z_target = np.copy(z_base)
        # その次元の値を標準偏差分だけ振る(あるいは純粋に -3から3へ)
        # 今回は z_base の値に -3 から 3 を「加算する」か、「上書きする」か。
        # VAEは事前分布がN(0,1)なので、上書きして -3 から 3 まで振るのが一般的
        z_target[dim] = val
        batch_z.append(z_target)
        coords.append((dim, step_idx))

batch_z = np.array(batch_z)
reconstructed = vae.decoder.predict(batch_z, batch_size=256)

for i, (r, c) in enumerate(coords):
    recon_img = reconstructed[i].squeeze()
    y_start = r * 128
    y_end = y_start + 128
    x_start = c * 128
    x_end = x_start + 128
    canvas[y_start:y_end, x_start:x_end] = recon_img

# プロット
plt.figure(figsize=(n_steps * 1.5, latent_dim * 1.5))
plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
plt.title(f'Latent Space Traversal (Base: {target_pdb} {target_angle})', fontsize=20)

# 軸のラベル設定
ticks_x = np.arange(0, n_steps) * 128 + 64
labels_x = [f'{v:.1f}' for v in z_range]
plt.xticks(ticks_x, labels_x, fontsize=12)
plt.xlabel('Latent Value', fontsize=16)

ticks_y = np.arange(0, latent_dim) * 128 + 64
labels_y = [f'Dim {d}' for d in range(latent_dim)]
plt.yticks(ticks_y, labels_y, fontsize=12)
plt.ylabel('Latent Dimension', fontsize=16)

# グリッド線を引く
for x in np.arange(0, n_steps + 1) * 128:
    plt.axvline(x, color='white', linestyle='-', linewidth=0.5)
for y in np.arange(0, latent_dim + 1) * 128:
    plt.axhline(y, color='white', linestyle='-', linewidth=0.5)

output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'latent_traversal_1kqm.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved to {output_path}')
