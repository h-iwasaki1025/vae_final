import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep'
base_output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep/pca_pairwise'
os.makedirs(base_output_dir, exist_ok=True)

for sub in ['State', 'AngleX', 'AngleZ']:
    os.makedirs(os.path.join(base_output_dir, sub), exist_ok=True)

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
    state, _ = parse_state_pdbid(fname.split('_')[0], fname)
    if state is None: state = 0
    y_train.append(state)
    
    try:
        parts = fname.split('.')[0].split('_')
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

print('Calculating PCA (10 components) on full data...')
pca = PCA(n_components=10)
pca_z_train_full = pca.fit_transform(z_mean_train)

# 描画負荷を下げるため、先ほどのご要望に合わせて30度刻みで間引く
step_size = 30
valid_idx = (angle_x_train % step_size == 0) & (angle_z_train % step_size == 0)

pca_z_train = pca_z_train_full[valid_idx]
y_train = y_train[valid_idx]
angle_x_train = angle_x_train[valid_idx]
angle_z_train = angle_z_train[valid_idx]

print(f'Filtered data shape: {pca_z_train.shape} (from {pca_z_train_full.shape})')

def plot_2d(i, j, color_array, cmap_name, save_dir, title_suffix, cbar_label):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # i, j are 0-indexed here
    x_val = pca_z_train[:, i]
    y_val = pca_z_train[:, j]
    
    if cmap_name == 'jet':
        norm = mcolors.Normalize(vmin=np.min(color_array), vmax=np.max(color_array))
        scatter = ax.scatter(x_val, y_val, c=color_array, cmap=cmap_name, s=6, alpha=0.8, norm=norm)
        cbar = plt.colorbar(scatter, ax=ax, ticks=np.unique(color_array))
    else:
        scatter = ax.scatter(x_val, y_val, c=color_array, cmap=cmap_name, s=6, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax)
        
    cbar.set_label(cbar_label, fontsize=12)
    plt.xlabel(f'PCA {i+1}', fontsize=14)
    plt.ylabel(f'PCA {j+1}', fontsize=14)
    plt.title(f'PCA {i+1} vs PCA {j+1} - {title_suffix}', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    save_name = f'pca_{i+1}_vs_{j+1}.png'
    plt.savefig(os.path.join(save_dir, save_name), dpi=100)
    plt.close()

count = 0
for i in range(10):
    for j in range(10):
        if i == j: continue
        
        # 1. State
        plot_2d(i, j, y_train, 'jet', os.path.join(base_output_dir, 'State'), 'Color: State', 'State')
        # 2. Angle X
        plot_2d(i, j, angle_x_train, 'RdYlBu', os.path.join(base_output_dir, 'AngleX'), 'Color: Angle X', 'X Angle (deg)')
        # 3. Angle Z
        plot_2d(i, j, angle_z_train, 'RdYlBu', os.path.join(base_output_dir, 'AngleZ'), 'Color: Angle Z', 'Z Angle (deg)')
        
        count += 3

print(f'Successfully generated {count} pairwise scatter plots.')
