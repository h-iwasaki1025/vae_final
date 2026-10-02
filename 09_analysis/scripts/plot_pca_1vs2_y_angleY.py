import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

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

def parse_angles(fname):
    parts = fname.split('.')[0].split('_')
    try:
        if parts[0] in ('0', '1', '2'):
            y_ang = int(parts[3])
            z_ang = int(parts[4])
        else:
            y_ang = int(parts[2])
            z_ang = int(parts[3])
        return y_ang, z_ang
    except Exception:
        return 0, 0

def main():
    model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep'
    output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)
    kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
    
    z_mean_test = np.load(os.path.join(model_dir, 'z_mean_test.npy'), allow_pickle=True)
    
    angle_y_train = []
    angle_z_train = []
    for p in kills_train:
        fname = os.path.basename(p)
        y_ang, z_ang = parse_angles(fname)
        angle_y_train.append(y_ang)
        angle_z_train.append(z_ang)
        
    angle_y_train = np.array(angle_y_train)
    angle_z_train = np.array(angle_z_train)
    
    print("Calculating PCA on test data...")
    import random
    random.seed(720)
    np.random.seed(720)
    n_vis = min(len(z_mean_test), 20000)
    idx_fit = np.random.choice(len(z_mean_test), n_vis, replace=False)
    
    pca = PCA(n_components=10)
    pca.fit(z_mean_test[idx_fit])
    
    pca_z_train = pca.transform(z_mean_train)
    
    # sparse filtering (30 deg) for better visualization
    step_size = 30
    valid_idx = (angle_y_train % step_size == 0) & (angle_z_train % step_size == 0)
    
    pca_z_train_filt = pca_z_train[valid_idx]
    y_filt = angle_y_train[valid_idx]
    z_filt = angle_z_train[valid_idx]
    
    # Plot Angle Y
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mcolors.Normalize(vmin=0, vmax=360)
    scatter = ax.scatter(pca_z_train_filt[:, 0], pca_z_train_filt[:, 1], c=y_filt, cmap='hsv', s=6, alpha=0.8, norm=norm)
    cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(0, 361, 60))
    cbar.set_label('Angle Y (deg)', fontsize=12)
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_title('PC1 vs PC2 (Train) - Colored by Angle Y', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_1_vs_2_AngleY_hsv.png'), dpi=200)
    plt.close()
    
    # Plot Angle Z
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_z_train_filt[:, 0], pca_z_train_filt[:, 1], c=z_filt, cmap='hsv', s=6, alpha=0.8, norm=norm)
    cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(0, 361, 60))
    cbar.set_label('Angle Z (deg)', fontsize=12)
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_title('PC1 vs PC2 (Train) - Colored by Angle Z', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_1_vs_2_AngleZ_hsv.png'), dpi=200)
    plt.close()

if __name__ == '__main__':
    main()
