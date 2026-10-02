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

kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
kills_test = fix_paths(np.load(os.path.join(model_dir, 'kills_test.npy'), allow_pickle=True))

all_paths = np.concatenate([kills_train, kills_test])

# 対象のPDBとノイズレベルでフィルタリング
filtered_paths = [p for p in all_paths if target_pdb in p and target_sn in p]
print(f'Found {len(filtered_paths)} paths for {target_pdb} and {target_sn}')

# パスを辞書に格納: key = (x_angle, z_angle), value = path
path_dict = {}
for p in filtered_paths:
    fname = os.path.basename(p)
    try:
        parts = fname.split('.')[0].split('_')
        # Format usually: 0_1kqm_xxx_yyy_zzz_sn1.6
        if parts[0] in ('0', '1', '2'):
            x_angle = int(parts[2])
            z_angle = int(parts[4])
        else:
            x_angle = int(parts[1])
            z_angle = int(parts[3])
        path_dict[(x_angle, z_angle)] = p
    except Exception as e:
        pass

sys.path.insert(0, '/Users/hikaru/Code/2026_Project/06_Analysis')
from model import VAE, get_encoder, get_decoder

encoder = get_encoder((128, 128, 1), 10)
decoder = get_decoder(10)
vae = VAE(encoder, decoder)
vae.build((None, 128, 128, 1))
vae.load_weights(os.path.join(model_dir, 'vae_best_weights.weights.h5'))

angles = np.arange(0, 360, 10)
img_size = 128

# 大きなカンバスを作成 (Z角度が行、X角度が列)
# 横: X角度 (0~350), 縦: Z角度 (0~350)
canvas = np.zeros((36 * img_size, 36 * img_size))

# まとめて推論するためのリスト
batch_images = []
batch_coords = []

for r, z in enumerate(angles):
    for c, x in enumerate(angles):
        p = path_dict.get((x, z), None)
        if p is not None and os.path.exists(p):
            with Image.open(p) as img:
                arr = np.array(img).astype('float32') / 255.0
                if len(arr.shape) == 2:
                    arr = np.expand_dims(arr, axis=-1)
                batch_images.append(arr)
                batch_coords.append((r, c))

batch_images = np.array(batch_images)
if len(batch_images) > 0:
    z_mean, _, _ = vae.encoder.predict(batch_images, batch_size=256)
    reconstructed = vae.decoder.predict(z_mean, batch_size=256)
    
    for i, (r, c) in enumerate(batch_coords):
        recon_img = reconstructed[i].squeeze()
        # カンバスに埋め込み
        y_start = r * img_size
        y_end = y_start + img_size
        x_start = c * img_size
        x_end = x_start + img_size
        canvas[y_start:y_end, x_start:x_end] = recon_img

# 画像として保存
plt.figure(figsize=(24, 24))
plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
plt.title(f'Reconstruction Map: {target_pdb} (Rows: Z angle 0-350, Cols: X angle 0-350)', fontsize=24)
plt.axis('off')

# 余白をなくして高解像度で保存
output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'angle_map_reconstruction_{target_pdb}.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
print(f'Saved to {output_path}')
