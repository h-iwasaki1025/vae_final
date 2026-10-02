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

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', path)
    if match:
        return match.group(1)
    return "Unknown"

def plot_matrix(data, color_info, cmap_name, title, save_path):
    print(f"Plotting {title}...")
    df = pd.DataFrame(data)
    df.columns = [f'Dim{i+1}' for i in range(df.shape[1])]
    fig = plt.figure(figsize=(15, 15))
    
    if cmap_name:
        cmap_obj = plt.colormaps.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=np.min(color_info), vmax=np.max(color_info))
        c_mapped = cmap_obj(norm(color_info))
        scatter_matrix(df, alpha=0.8, c=c_mapped, s=2, figsize=(15, 15))
    else:
        scatter_matrix(df, alpha=0.8, c=color_info, s=2, figsize=(15, 15))
        
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved to', save_path)

def main():
    model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep'
    output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep'
    os.makedirs(output_dir, exist_ok=True)
    
    target_noises = ['1.6']

    # --- TRAIN DATA ---
    print("Loading Train data...")
    kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
    z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)
    
    y_train = []
    noise_train = []
    for p in kills_train:
        fname = os.path.basename(str(p))
        state, _ = parse_state_pdbid(fname.split('_')[0], fname)
        if state is None:
            state = 0
        y_train.append(state)
        noise_train.append(parse_noise(str(p)))
    y_train = np.array(y_train)
    noise_train = np.array(noise_train)
    
    mask_train = np.isin(noise_train, target_noises)
    z_train_filtered = z_mean_train[mask_train]
    y_train_filtered = y_train[mask_train]

    # --- TEST DATA ---
    print("Loading Test data...")
    kills_test = fix_paths(np.load(os.path.join(model_dir, 'kills_test.npy'), allow_pickle=True))
    z_mean_test = np.load(os.path.join(model_dir, 'z_mean_test.npy'), allow_pickle=True)
    pdbids_test = np.load(os.path.join(model_dir, 'pdbids_test.npy'), allow_pickle=True)
    
    noise_test = []
    for p in kills_test:
        noise_test.append(parse_noise(str(p)))
    noise_test = np.array(noise_test)
    
    mask_test = np.isin(noise_test, target_noises)
    z_test_filtered = z_mean_test[mask_test]
    pdb_test_filtered = pdbids_test[mask_test]
    
    print(f"Test PDBs found: {np.unique(pdb_test_filtered)}")
    
    # Test用カラー指定: 1l2o -> blue, 1qvi -> orange, 2ec6 -> green
    test_color_map = {'1l2o': '#1f77b4', '1qvi': '#ff7f0e', '2ec6': '#2ca02c'}
    c_test = [test_color_map.get(pdb, 'gray') for pdb in pdb_test_filtered]

    # --- PCA Fit (Test 全体でfit, main.pyと揃える) ---
    print('Calculating PCA on full test data...')
    import random
    random.seed(720)
    np.random.seed(720)
    n_vis = min(len(z_mean_test), 20000)
    idx = np.random.choice(len(z_mean_test), n_vis, replace=False)
    z_mean_test_vis = z_mean_test[idx]
    
    pca = PCA(n_components=z_mean_test.shape[1])
    pca.fit(z_mean_test_vis)
    
    pca_train_filtered = pca.transform(z_train_filtered)
    pca_test_filtered = pca.transform(z_test_filtered)

    # --- PLOT ---
    # 1. Train Raw Z (y_train_filtered と 'jet' を渡す)
    plot_matrix(z_train_filtered, y_train_filtered, 'jet', 
                'Raw Latent Space (Train) - Noise 1.6 (State)', 
                os.path.join(output_dir, 'latent_scatter_matrix_train_raw_filtered_noise1.6.png'))

    # 2. Train PCA Z (y_train_filtered と 'jet' を渡す)
    plot_matrix(pca_train_filtered, y_train_filtered, 'jet', 
                'PCA Latent Space (Train) - Noise 1.6 (State)', 
                os.path.join(output_dir, 'latent_scatter_matrix_train_pca_filtered_noise1.6.png'))

    # 3. Test Raw Z (c_test と None を渡す)
    plot_matrix(z_test_filtered, c_test, None, 
                'Raw Latent Space (Test) - Noise 1.6 (PDB)', 
                os.path.join(output_dir, 'latent_scatter_matrix_test_raw_filtered_noise1.6.png'))

    # 4. Test PCA Z (c_test と None を渡す)
    plot_matrix(pca_test_filtered, c_test, None, 
                'PCA Latent Space (Test) - Noise 1.6 (PDB)', 
                os.path.join(output_dir, 'latent_scatter_matrix_test_pca_filtered_noise1.6.png'))

if __name__ == '__main__':
    main()
