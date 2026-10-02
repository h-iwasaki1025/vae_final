import os
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

def main():
    model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep'
    output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Train data...")
    kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
    z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)
    
    y_train = []
    for p in kills_train:
        fname = os.path.basename(str(p))
        state, _ = parse_state_pdbid(fname.split('_')[0], fname)
        if state is None:
            state = 0
        y_train.append(state)
    y_train = np.array(y_train)

    print('Calculating PCA on full training data...')
    pca = PCA(n_components=z_mean_train.shape[1])
    pca_train = pca.fit_transform(z_mean_train)

    print("Plotting MAX Train PCA...")
    df = pd.DataFrame(pca_train)
    df.columns = [f'Dim{i+1}' for i in range(df.shape[1])]
    fig = plt.figure(figsize=(15, 15))
    
    cmap_obj = plt.colormaps.get_cmap('jet')
    norm = mcolors.Normalize(vmin=np.min(y_train), vmax=np.max(y_train))
    c_mapped = cmap_obj(norm(y_train))
    
    scatter_matrix(df, alpha=0.8, c=c_mapped, s=2, figsize=(15, 15))
    plt.suptitle('PCA Latent Space (Train) - MAX (Color: State)', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, 'latent_scatter_matrix_train_max_pca.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved to', save_path)

if __name__ == '__main__':
    main()
