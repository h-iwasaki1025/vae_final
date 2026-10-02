import os
import sys
import numpy as np
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

def generate_pca_pairs_for_model(model_name):
    print(f"=====================================")
    print(f"Starting analysis for: {model_name}")
    print(f"=====================================")

    model_dir = f'/Users/hikaru/Code/2026_Project/04_Savemodel/{model_name}'
    base_output_dir = f'/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_{model_name}'
    
    # Check if necessary files exist
    if not os.path.exists(os.path.join(model_dir, 'z_mean_test.npy')):
        print(f"Error: z_mean_test.npy not found in {model_dir}. Trying to fallback to 07_result...")
        if not os.path.exists(os.path.join(base_output_dir, 'z_mean_test.npy')):
            print(f"Error: Data not found. Skipping {model_name}.")
            return
        else:
            data_dir = base_output_dir
    else:
        data_dir = model_dir

    # --- 1. Load Train Data ---
    print("Loading train data...")
    z_mean_train = np.load(os.path.join(data_dir, 'z_mean_train.npy'), allow_pickle=True)
    kills_train = fix_paths(np.load(os.path.join(data_dir, 'kills_train.npy'), allow_pickle=True))
    
    y_train = []
    angle_x_train = []
    angle_z_train = []
    for p in kills_train:
        fname = os.path.basename(p)
        # state
        parts = fname.split("_")
        state = 0
        if len(parts) >= 2 and parts[0] in ("0","1","2"):
            state = int(parts[0])
        y_train.append(state)
        # angles
        try:
            parts2 = fname.split('.')[0].split('_')
            if parts2[0] in ('0', '1', '2'):
                x_ang = int(parts2[2])
                z_ang = int(parts2[4])
            else:
                x_ang = int(parts2[1])
                z_ang = int(parts2[3])
            angle_x_train.append(x_ang)
            angle_z_train.append(z_ang)
        except Exception:
            angle_x_train.append(0)
            angle_z_train.append(0)

    y_train = np.array(y_train)
    angle_x_train = np.array(angle_x_train)
    angle_z_train = np.array(angle_z_train)

    # --- 2. Load Test Data ---
    print("Loading test data...")
    z_mean_test = np.load(os.path.join(data_dir, 'z_mean_test.npy'), allow_pickle=True)
    kills_test = fix_paths(np.load(os.path.join(data_dir, 'kills_test.npy'), allow_pickle=True))
    pdbids_test = np.load(os.path.join(data_dir, 'pdbids_test.npy'), allow_pickle=True)

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

    # --- 3. PCA Space (Fitted on Test Data like main.py) ---
    print("Calculating PCA (10 components) on test data...")
    pca = PCA(n_components=10)
    import random
    random.seed(720)
    np.random.seed(720)
    n_vis = min(len(z_mean_test), 20000)
    idx_fit = np.random.choice(len(z_mean_test), n_vis, replace=False)
    pca.fit(z_mean_test[idx_fit])

    pca_z_train_full = pca.transform(z_mean_train)
    pca_z_test_full = pca.transform(z_mean_test)

    # --- 4. Sparse Filtering (30 degrees) ---
    step_size = 30
    valid_idx_train = (angle_x_train % step_size == 0) & (angle_z_train % step_size == 0)
    pca_z_train = pca_z_train_full[valid_idx_train]
    y_train = y_train[valid_idx_train]
    angle_x_train = angle_x_train[valid_idx_train]
    angle_z_train = angle_z_train[valid_idx_train]
    print(f"Filtered Train data shape: {pca_z_train.shape}")

    valid_idx_test = (angle_x_test % step_size == 0) & (angle_z_test % step_size == 0)
    pca_z_test = pca_z_test_full[valid_idx_test]
    pdbids_test = pdbids_test[valid_idx_test]
    print(f"Filtered Test data shape: {pca_z_test.shape}")

    # --- 5. Generate Plots ---
    out_dir_train = os.path.join(base_output_dir, 'pca_pairwise_train')
    out_dir_test = os.path.join(base_output_dir, 'pca_pairwise_test')
    
    for sub in ['State', 'AngleX', 'AngleZ']:
        os.makedirs(os.path.join(out_dir_train, sub), exist_ok=True)
    os.makedirs(os.path.join(out_dir_test, 'PDBID'), exist_ok=True)

    # 5-1. Train Plots Function
    def plot_2d_train(i, j, color_array, cmap_name, save_dir, title_suffix, cbar_label):
        fig, ax = plt.subplots(figsize=(8, 6))
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
        ax.set_xlabel(f'PCA {i+1}', fontsize=14)
        ax.set_ylabel(f'PCA {j+1}', fontsize=14)
        ax.set_title(f'PCA {i+1} vs PCA {j+1} - {title_suffix}', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pca_{i+1}_vs_{j+1}.png'), dpi=100)
        plt.close(fig)

    # 5-2. Test PDBID Plot Function
    unique_pdbs = sorted(list(set(pdbids_test)))
    pdb_to_idx = {pid: idx for idx, pid in enumerate(unique_pdbs)}
    cmap_pdb_name = 'tab10' if len(unique_pdbs) <= 10 else 'tab20'
    cmap_pdb = plt.get_cmap(cmap_pdb_name)

    def plot_2d_test_pdb(i, j):
        fig, ax = plt.subplots(figsize=(8, 6))
        x_val = pca_z_test[:, i]
        y_val = pca_z_test[:, j]
        
        # Exact matched colors
        colors = [cmap_pdb(pdb_to_idx[pid]) for pid in pdbids_test]
        ax.scatter(x_val, y_val, c=colors, s=6, alpha=0.8, edgecolors='none')
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=pid,
                                  markerfacecolor=cmap_pdb(pdb_to_idx[pid]), markersize=8)
                           for pid in unique_pdbs]
        ax.legend(handles=legend_elements, title="PDB ID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax.set_xlabel(f'PCA {i+1}', fontsize=14)
        ax.set_ylabel(f'PCA {j+1}', fontsize=14)
        ax.set_title(f'PCA {i+1} vs PCA {j+1} - Color: PDB ID (Test)', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_test, 'PDBID', f'pca_{i+1}_vs_{j+1}.png'), dpi=100)
        plt.close(fig)

    print("Generating plots...")
    count_train = 0
    count_test = 0
    for i in range(10):
        for j in range(10):
            if i == j: continue
            
            # Train
            plot_2d_train(i, j, y_train, 'jet', os.path.join(out_dir_train, 'State'), 'Color: State', 'State')
            plot_2d_train(i, j, angle_x_train, 'RdYlBu', os.path.join(out_dir_train, 'AngleX'), 'Color: Angle X', 'X Angle (deg)')
            plot_2d_train(i, j, angle_z_train, 'RdYlBu', os.path.join(out_dir_train, 'AngleZ'), 'Color: Angle Z', 'Z Angle (deg)')
            count_train += 3
            
            # Test
            plot_2d_test_pdb(i, j)
            count_test += 1
            
    print(f"Done for {model_name}. Generated {count_train} Train plots and {count_test} Test plots.")

if __name__ == "__main__":
    models = ["20260609x_y_ep", "20260609y_z_ep"]
    for model in models:
        generate_pca_pairs_for_model(model)
