import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

def main():
    model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep'
    base_output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep/pca_pairwise_test'
    out_dir = os.path.join(base_output_dir, 'PDBID')
    os.makedirs(out_dir, exist_ok=True)

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
    z_mean_test = np.load(os.path.join(model_dir, 'z_mean_test.npy'), allow_pickle=True)
    pdbids_test = np.load(os.path.join(model_dir, 'pdbids_test.npy'), allow_pickle=True)

    angle_x_test = []
    angle_z_test = []

    for p in kills_test:
        fname = os.path.basename(p)
        try:
            parts = fname.split('.')[0].split('_')
            if parts[0] in ('0', '1', '2'):
                x_ang = int(parts[2])
                z_ang = int(parts[4])
            else:
                x_ang = int(parts[1])
                z_ang = int(parts[3])
            angle_x_test.append(x_ang)
            angle_z_test.append(z_ang)
        except Exception:
            angle_x_test.append(0)
            angle_z_test.append(0)

    angle_x_test = np.array(angle_x_test)
    angle_z_test = np.array(angle_z_test)

    print('Calculating PCA (10 components) on test data...')
    pca = PCA(n_components=10)
    
    import random
    random.seed(720)
    np.random.seed(720)
    n_vis = min(len(z_mean_test), 20000)
    idx_fit = np.random.choice(len(z_mean_test), n_vis, replace=False)
    pca.fit(z_mean_test[idx_fit])
    
    pca_z_test_full = pca.transform(z_mean_test)

    # 間引き (Sparse)
    step_size = 30
    valid_idx = (angle_x_test % step_size == 0) & (angle_z_test % step_size == 0)

    pca_z_sub = pca_z_test_full[valid_idx]
    pdbids_sub = pdbids_test[valid_idx]

    print(f"Filtered data shape: {pca_z_sub.shape} (from {pca_z_test_full.shape})")

    unique_pdbs = sorted(list(set(pdbids_sub)))
    pdb_to_idx = {pid: i for i, pid in enumerate(unique_pdbs)}
    pdb_color_array = np.array([pdb_to_idx[pid] for pid in pdbids_sub])
    cmap_name = 'tab10' if len(unique_pdbs) <= 10 else 'tab20'
    cmap = plt.get_cmap(cmap_name)

    def plot_2d_pdb(i, j):
        fig, ax = plt.subplots(figsize=(8, 6))
        x_val = pca_z_sub[:, i]
        y_val = pca_z_sub[:, j]
        
        # rgba colors list directly
        colors = [cmap(pdb_to_idx[pid]) for pid in pdbids_sub]
        ax.scatter(x_val, y_val, c=colors, s=6, alpha=0.8, edgecolors='none')
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=pid,
                                  markerfacecolor=cmap(pdb_to_idx[pid]), markersize=8)
                           for pid in unique_pdbs]
        ax.legend(handles=legend_elements, title="PDB ID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax.set_xlabel(f'PCA {i+1}', fontsize=14)
        ax.set_ylabel(f'PCA {j+1}', fontsize=14)
        ax.set_title(f'PCA {i+1} vs PCA {j+1} - Color: PDB ID (Test)', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_name = f'pca_{i+1}_vs_{j+1}.png'
        plt.savefig(os.path.join(out_dir, save_name), dpi=100)
        plt.close(fig)

    count = 0
    for i in range(10):
        for j in range(10):
            if i == j: continue
            plot_2d_pdb(i, j)
            count += 1
            
    print(f"Successfully generated {count} pairwise scatter plots.")

if __name__ == "__main__":
    main()
