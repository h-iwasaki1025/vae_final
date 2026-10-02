import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import multiprocessing as mp

MODEL_DIR = sys.argv[1]
OUTPUT_DIR = os.path.join(sys.argv[2], "scatter_matrices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', str(path))
    if match: return float(match.group(1))
    return 0.0

def parse_angles(fname):
    parts = fname.split('.')[0].split('_')
    try:
        # Format: 0_1l2o_000_140_010_sn0.6 (X_Y_Z)
        if parts[0] in ('0', '1', '2'):
            x_ang = int(parts[3])
            z_ang = int(parts[4])
        else:
            x_ang = int(parts[2])
            z_ang = int(parts[3])
        return x_ang, z_ang
    except Exception:
        return 0, 0

def plot_scatter_matrix(args):
    data, labels, label_type, title, save_path = args
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    if label_type in ['angleX', 'angleZ']:
        cmap = 'hsv'; vmin, vmax = 0, 360
    elif label_type == 'noise':
        cmap = 'jet'; vmin, vmax = np.min(labels), np.max(labels)
    elif label_type == 'state':
        cmap = matplotlib.colors.ListedColormap(['#3b4cc0', '#b40426']); vmin, vmax = -0.5, 1.5
    else: # pdbid
        cmap = 'tab10'
        unique_labels = list(set(labels))
        unique_labels.sort()
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels = np.array([label_map[lbl] for lbl in labels])
        vmin, vmax = -0.5, len(unique_labels) - 0.5

    for i in range(10):
        for j in range(10):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[:, i], bins=30, color='gray', alpha=0.7)
            else:
                sc = ax.scatter(data[:, j], data[:, i], c=labels, cmap=cmap, vmin=vmin, vmax=vmax, s=2, alpha=0.5)
            if i == 9: ax.set_xlabel(f'Dim {j+1}')
            else: ax.set_xticks([])
            if j == 0: ax.set_ylabel(f'Dim {i+1}')
            else: ax.set_yticks([])

    fig.suptitle(title, fontsize=32)
    if label_type not in ['state', 'pdbid']:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sc, cax=cbar_ax)
    else:
        handles, leg_labels = sc.legend_elements()
        if label_type == 'pdbid':
            # Extract numerical part carefully to index unique_labels
            clean_labels = []
            for lbl in leg_labels:
                match = re.search(r'\d+', lbl)
                idx = int(match.group()) if match else 0
                clean_labels.append(unique_labels[idx] if idx < len(unique_labels) else str(idx))
            leg_labels = clean_labels
        if label_type == 'state':
            leg_labels = [f"State {int(float(re.search(r'-?\d+\.?\d*', l).group()))}" for l in leg_labels]
        fig.legend(handles, leg_labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=16, title=label_type.capitalize(), title_fontsize=20)
        
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    print("Running 10D Scatter Matrices...")
    z_train = np.load(os.path.join(MODEL_DIR, "z_mean_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"), allow_pickle=True)
    kills_train = np.load(os.path.join(MODEL_DIR, "kills_train.npy"), allow_pickle=True)
    
    z_test = np.load(os.path.join(MODEL_DIR, "z_mean_test.npy"), allow_pickle=True)
    pdbids_test = np.load(os.path.join(MODEL_DIR, "pdbids_test.npy"), allow_pickle=True)
    kills_test = np.load(os.path.join(MODEL_DIR, "kills_test.npy"), allow_pickle=True)
    
    # Check for additional sn1.6 inference data
    sn16_train_exists = os.path.exists(os.path.join(MODEL_DIR, "z_mean_train_sn1.6.npy"))
    sn16_test_exists = os.path.exists(os.path.join(MODEL_DIR, "z_mean_test_sn1.6.npy"))
    
    if sn16_train_exists:
        z_train_sn16 = np.load(os.path.join(MODEL_DIR, "z_mean_train_sn1.6.npy"))
        idx_train_sn16 = np.load(os.path.join(MODEL_DIR, "indices_train_sn1.6.npy"))
        y_train_sn16 = y_train[idx_train_sn16]
        noise_train_sn16 = np.full(len(z_train_sn16), 1.6)
        kills_train_sn16 = np.load(os.path.join(MODEL_DIR, "kills_train_sn1.6.npy"))
        angleX_train_sn16 = np.array([parse_angles(os.path.basename(p))[0] for p in kills_train_sn16])
        angleZ_train_sn16 = np.array([parse_angles(os.path.basename(p))[1] for p in kills_train_sn16])
    
    if sn16_test_exists:
        z_test_sn16 = np.load(os.path.join(MODEL_DIR, "z_mean_test_sn1.6.npy"))
        idx_test_sn16 = np.load(os.path.join(MODEL_DIR, "indices_test_sn1.6.npy"))
        pdbids_test_sn16 = pdbids_test[idx_test_sn16]
        noise_test_sn16 = np.full(len(z_test_sn16), 1.6)
        kills_test_sn16 = np.load(os.path.join(MODEL_DIR, "kills_test_sn1.6.npy"))
        angleX_test_sn16 = np.array([parse_angles(os.path.basename(p))[0] for p in kills_test_sn16])
        angleZ_test_sn16 = np.array([parse_angles(os.path.basename(p))[1] for p in kills_test_sn16])
    
    noise_train = np.array([parse_noise(p) for p in kills_train])
    angleX_train = np.array([parse_angles(os.path.basename(p))[0] for p in kills_train])
    angleZ_train = np.array([parse_angles(os.path.basename(p))[1] for p in kills_train])
    
    noise_test = np.array([parse_noise(p) for p in kills_test])
    angleX_test = np.array([parse_angles(os.path.basename(p))[0] for p in kills_test])
    angleZ_test = np.array([parse_angles(os.path.basename(p))[1] for p in kills_test])

    np.random.seed(720)
    sample_indices = np.random.choice(len(z_test), min(20000, len(z_test)), replace=False)
    pca = PCA(n_components=10)
    pca.fit(z_test[sample_indices])
    
    z_pca_train = pca.transform(z_train)
    z_pca_test = pca.transform(z_test)
    
    tasks = []
    datasets = {
        'train': {
            'raw': {'all': z_train},
            'pca': {'all': z_pca_train},
            'labels': {'all': {'state': y_train, 'noise': noise_train, 'angleX': angleX_train, 'angleZ': angleZ_train}},
            'noise_array': {'all': noise_train}
        },
        'test': {
            'raw': {'all': z_test},
            'pca': {'all': z_pca_test},
            'labels': {'all': {'pdbid': pdbids_test, 'noise': noise_test, 'angleX': angleX_test, 'angleZ': angleZ_test}},
            'noise_array': {'all': noise_test}
        }
    }
    
    if sn16_train_exists:
        datasets['train']['raw']['sn1.6'] = z_train_sn16
        datasets['train']['pca']['sn1.6'] = pca.transform(z_train_sn16)
        datasets['train']['labels']['sn1.6'] = {'state': y_train_sn16, 'noise': noise_train_sn16, 'angleX': angleX_train_sn16, 'angleZ': angleZ_train_sn16}
        datasets['train']['noise_array']['sn1.6'] = noise_train_sn16
        
    if sn16_test_exists:
        datasets['test']['raw']['sn1.6'] = z_test_sn16
        datasets['test']['pca']['sn1.6'] = pca.transform(z_test_sn16)
        datasets['test']['labels']['sn1.6'] = {'pdbid': pdbids_test_sn16, 'noise': noise_test_sn16, 'angleX': angleX_test_sn16, 'angleZ': angleZ_test_sn16}
        datasets['test']['noise_array']['sn1.6'] = noise_test_sn16
    
    tasks = []
    for ds_name, ds_data in datasets.items():
        for space_name in ['raw', 'pca']:
            for filter_name in ['all', 'sn1.6']:
                if filter_name in ds_data[space_name]:
                    data_full = ds_data[space_name][filter_name]
                    noise_full = ds_data['noise_array'][filter_name]
                    labels_dict = ds_data['labels'][filter_name]
                else:
                    # If not explicitly provided, derive from 'all'
                    data_full = ds_data[space_name]['all']
                    noise_full = ds_data['noise_array']['all']
                    labels_dict = ds_data['labels']['all']
                
                mask = (noise_full == 1.6) if filter_name == 'sn1.6' and not ('sn16_exists' in locals() and ds_name == 'train' and sn16_train_exists) else np.ones(len(data_full), dtype=bool)
                data_plot = data_full[mask]
                if len(data_plot) == 0: continue
                
                for label_name, label_full in labels_dict.items():
                    labels_plot = label_full[mask]
                    title = f"10D Scatter Matrix | {ds_name.upper()} | {space_name.upper()} | {filter_name.upper()} | Color: {label_name}"
                    save_dir = os.path.join(OUTPUT_DIR, filter_name)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"scatter_{ds_name}_{space_name}_{filter_name}_{label_name}_rdylbu.png")
                    tasks.append((data_plot, labels_plot, label_name, title, save_path))

    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        pool.map(plot_scatter_matrix, tasks)

if __name__ == '__main__':
    main()
