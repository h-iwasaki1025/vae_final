import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    base_out = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep'
    
    out_dir_y = os.path.join(base_out, 'pca_pairwise_test', 'AngleY')
    out_dir_z = os.path.join(base_out, 'pca_pairwise_test', 'AngleZ')
    os.makedirs(out_dir_y, exist_ok=True)
    os.makedirs(out_dir_z, exist_ok=True)
    
    print("Loading Test data...")
    z_mean_test = np.load(os.path.join(model_dir, 'z_mean_test.npy'), allow_pickle=True)
    kills_test = fix_paths(np.load(os.path.join(model_dir, 'kills_test.npy'), allow_pickle=True))
    
    angle_y_test = []
    angle_z_test = []
    for p in kills_test:
        fname = os.path.basename(p)
        y_ang, z_ang = parse_angles(fname)
        angle_y_test.append(y_ang)
        angle_z_test.append(z_ang)
        
    angle_y_test = np.array(angle_y_test)
    angle_z_test = np.array(angle_z_test)
    
    print("Calculating PCA on test data...")
    import random
    random.seed(720)
    np.random.seed(720)
    n_vis = min(len(z_mean_test), 20000)
    idx_fit = np.random.choice(len(z_mean_test), n_vis, replace=False)
    
    pca = PCA(n_components=10)
    pca.fit(z_mean_test[idx_fit])
    
    pca_z_test = pca.transform(z_mean_test)
    
    # 描画のクリアさのため30度間隔のデータに絞る
    step_size = 30
    valid_idx = (angle_y_test % step_size == 0) & (angle_z_test % step_size == 0)
    pca_z_test_filt = pca_z_test[valid_idx]
    y_filt = angle_y_test[valid_idx]
    z_filt = angle_z_test[valid_idx]
    
    def plot_pair(i, j):
        norm = mcolors.Normalize(vmin=0, vmax=360)
        
        # Angle Y
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(pca_z_test_filt[:, i], pca_z_test_filt[:, j], c=y_filt, cmap='hsv', s=6, alpha=0.8, norm=norm)
        cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(0, 361, 60))
        cbar.set_label('Angle Y (deg)', fontsize=12)
        ax.set_xlabel(f'PCA {i+1}', fontsize=12)
        ax.set_ylabel(f'PCA {j+1}', fontsize=12)
        ax.set_title(f'PCA {i+1} vs PCA {j+1} - Angle Y (Test)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_y, f'pca_{i+1}_vs_{j+1}.png'), dpi=100)
        plt.close(fig)

        # Angle Z
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(pca_z_test_filt[:, i], pca_z_test_filt[:, j], c=z_filt, cmap='hsv', s=6, alpha=0.8, norm=norm)
        cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(0, 361, 60))
        cbar.set_label('Angle Z (deg)', fontsize=12)
        ax.set_xlabel(f'PCA {i+1}', fontsize=12)
        ax.set_ylabel(f'PCA {j+1}', fontsize=12)
        ax.set_title(f'PCA {i+1} vs PCA {j+1} - Angle Z (Test)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_z, f'pca_{i+1}_vs_{j+1}.png'), dpi=100)
        plt.close(fig)

    print("Generating individual pair plots...")
    for i in range(10):
        for j in range(10):
            if i == j: continue
            plot_pair(i, j)
            
    print("Done generating 90 pairs for Angle Y and Angle Z.")

if __name__ == '__main__':
    main()
