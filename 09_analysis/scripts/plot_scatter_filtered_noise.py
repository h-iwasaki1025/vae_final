import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA

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

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', path)
    if match:
        return match.group(1)
    return "Unknown"

def main():
    model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep'
    output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
    z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)

    y_train = []
    noise_train = []
    
    print("Parsing states and noise levels...")
    for p in kills_train:
        fname = os.path.basename(str(p))
        state, _ = parse_state_pdbid(fname.split('_')[0], fname)
        if state is None:
            state = 0
        y_train.append(state)
        noise_train.append(parse_noise(str(p)))

    y_train = np.array(y_train)
    noise_train = np.array(noise_train)

    # 軸を元のScatter Matrixと一致させるために、PCAは全データに対してフィットする
    print('Calculating PCA (10 components) on full training data to keep axes consistent...')
    pca = PCA(n_components=10)
    pca_z_train = pca.fit_transform(z_mean_train)

    print('Filtering data by noise levels [1.6]...')
    target_noises = ['1.6']
    mask = np.isin(noise_train, target_noises)
    
    pca_filtered = pca_z_train[mask]
    y_filtered = y_train[mask]
    
    print(f"Original points: {len(y_train)}, Filtered points: {len(y_filtered)}")

    df = pd.DataFrame(pca_filtered)
    df.columns = [f'PCA{i+1}' for i in range(df.shape[1])]

    print('Plotting Scatter Matrix...')
    # pandas.plotting.scatter_matrix does not return a single Axes, it returns a matrix of Axes.
    # We must plot it within a Figure context.
    fig = plt.figure(figsize=(15, 15))
    
    cmap_obj = plt.colormaps.get_cmap('jet')
    norm = mcolors.Normalize(vmin=np.min(y_filtered), vmax=np.max(y_filtered))
    c_mapped = cmap_obj(norm(y_filtered))
    
    axes = scatter_matrix(df, alpha=0.8, c=c_mapped, s=2, figsize=(15, 15))
    plt.suptitle('PCA Latent Space Scatter Matrix (Train) - Noise: 1.6 (Color: State)', fontsize=18)
    
    # scatter_matrix の tight_layout の代わり
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, 'latent_scatter_matrix_train_pca_filtered_noise1.6.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print('Saved to', save_path)

if __name__ == '__main__':
    main()
