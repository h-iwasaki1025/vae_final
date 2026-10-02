import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA

model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep'
output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep'
os.makedirs(output_dir, exist_ok=True)

# ユーティリティ関数（パス修正）
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

# State をパースする関数
def parse_state_pdbid(m: str, fname: str):
    p = m.split("_")
    if len(p) >= 2 and p[0] in ("0","1","2"):
        return int(p[0]), p[1]
    q = fname.split("_")
    if len(q) >= 2 and q[0] in ("0","1","2"):
        return int(q[0]), q[1]
    return None, None

kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)

y_train = []
angle_x_train = []
angle_z_train = []

for p in kills_train:
    fname = os.path.basename(p)
    # State のパース
    state, _ = parse_state_pdbid(fname.split('_')[0], fname)
    if state is None:
        state = 0
    y_train.append(state)
    
    # Angle のパース
    try:
        parts = fname.split('.')[0].split('_')
        # Format usually: 0_1kqm_xxx_yyy_zzz_sn1.6
        if parts[0] in ('0', '1', '2'):
            x_ang = int(parts[2])
            z_ang = int(parts[4])
        else:
            x_ang = int(parts[1])
            z_ang = int(parts[3])
        angle_x_train.append(x_ang)
        angle_z_train.append(z_ang)
    except Exception:
        angle_x_train.append(0)
        angle_z_train.append(0)

y_train = np.array(y_train)
angle_x_train = np.array(angle_x_train)
angle_z_train = np.array(angle_z_train)

# PCAで10次元に変換
print('Calculating PCA (10 components)...')
pca = PCA(n_components=10)
pca_z_train = pca.fit_transform(z_mean_train)

df = pd.DataFrame(pca_z_train)
df.columns = [f'PCA{i+1}' for i in range(df.shape[1])]

def plot_and_save_matrix(color_array, cmap_name, save_name, title_suffix):
    print(f'Plotting {save_name}...')
    fig = plt.figure(figsize=(15, 15))
    
    cmap_obj = plt.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.min(color_array), vmax=np.max(color_array))
    c_mapped = cmap_obj(norm(color_array))
    
    axes = scatter_matrix(
        df, alpha=0.8, c=c_mapped, s=2, figsize=(15, 15)
    )
    plt.suptitle(f'PCA Latent Space Scatter Matrix (Train) - {title_suffix}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 1. State
plot_and_save_matrix(y_train, 'jet', 'latent_scatter_matrix_train_pca_State.png', 'Color: State')

# 2. Angle X
plot_and_save_matrix(angle_x_train, 'hsv', 'latent_scatter_matrix_train_pca_AngleX.png', 'Color: Angle X')

# 3. Angle Z
plot_and_save_matrix(angle_z_train, 'hsv', 'latent_scatter_matrix_train_pca_AngleZ.png', 'Color: Angle Z')

print('All 3 scatter matrices generated successfully.')
