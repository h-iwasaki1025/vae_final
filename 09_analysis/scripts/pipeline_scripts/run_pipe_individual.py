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
OUTPUT_DIR = os.path.join(sys.argv[2], "pca_individual")

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', str(path))
    if match: return float(match.group(1))
    return 0.0

def parse_angles(fname):
    parts = fname.split('.')[0].split('_')
    try:
        if parts[0] in ('0', '1', '2'):
            x_ang = int(parts[3])
            z_ang = int(parts[4])
        else:
            x_ang = int(parts[2])
            z_ang = int(parts[3])
        return x_ang, z_ang
    except Exception:
        return 0, 0

def plot_single_pair(args):
    data_x, data_y, labels, pc_x, pc_y, label_type, title, save_path = args
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if label_type in ['angleX', 'angleZ']: cmap = 'hsv'; vmin, vmax = 0, 360
    elif label_type == 'noise': cmap = 'jet'; vmin, vmax = np.min(labels), np.max(labels)
    elif label_type == 'state': cmap = matplotlib.colors.ListedColormap(['#3b4cc0', '#b40426']); vmin, vmax = -0.5, 1.5
    else:
        cmap = 'tab10'
        unique_labels = list(set(labels))
        unique_labels.sort()
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        labels = np.array([label_map[lbl] for lbl in labels])
        vmin, vmax = -0.5, len(unique_labels) - 0.5

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(data_x, data_y, c=labels, cmap=cmap, vmin=vmin, vmax=vmax, s=5, alpha=0.6)
    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.title(title, fontsize=12)
    if label_type not in ['state', 'pdbid']:
        plt.colorbar(sc, label=label_type)
    else:
        handles, leg_labels = sc.legend_elements()
        if label_type == 'pdbid':
            clean_labels = []
            for lbl in leg_labels:
                match = re.search(r'\d+', lbl)
                idx = int(match.group()) if match else 0
                clean_labels.append(unique_labels[idx] if idx < len(unique_labels) else str(idx))
            leg_labels = clean_labels
        plt.legend(handles, leg_labels, loc='center left', bbox_to_anchor=(1, 0.5), title=label_type.upper())
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    print("Running PCA Individual Plots...")
    z_train = np.load(os.path.join(MODEL_DIR, "z_mean_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"), allow_pickle=True)
    kills_train = np.load(os.path.join(MODEL_DIR, "kills_train.npy"), allow_pickle=True)
    z_test = np.load(os.path.join(MODEL_DIR, "z_mean_test.npy"), allow_pickle=True)
    pdbids_test = np.load(os.path.join(MODEL_DIR, "pdbids_test.npy"), allow_pickle=True)
    kills_test = np.load(os.path.join(MODEL_DIR, "kills_test.npy"), allow_pickle=True)
    
    noise_train = np.array([parse_noise(p) for p in kills_train])
    angleX_train = np.array([parse_angles(os.path.basename(p))[0] for p in kills_train])
    angleZ_train = np.array([parse_angles(os.path.basename(p))[1] for p in kills_train])
    
    noise_test = np.array([parse_noise(p) for p in kills_test])
    angleX_test = np.array([parse_angles(os.path.basename(p))[0] for p in kills_test])
    angleZ_test = np.array([parse_angles(os.path.basename(p))[1] for p in kills_test])

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

    np.random.seed(720)
    sample_indices = np.random.choice(len(z_test), min(20000, len(z_test)), replace=False)
    pca = PCA(n_components=10)
    pca.fit(z_test[sample_indices])
    
    z_pca_train = pca.transform(z_train)
    z_pca_test = pca.transform(z_test)
    
    datasets = {
        'train': {
            'pca': {'all': z_pca_train},
            'labels': {'all': {'state': y_train, 'noise': noise_train, 'angleX': angleX_train, 'angleZ': angleZ_train}},
            'noise_array': {'all': noise_train}
        },
        'test': {
            'pca': {'all': z_pca_test},
            'labels': {'all': {'pdbid': pdbids_test, 'noise': noise_test, 'angleX': angleX_test, 'angleZ': angleZ_test}},
            'noise_array': {'all': noise_test}
        }
    }
    
    if sn16_train_exists:
        datasets['train']['pca']['sn1.6'] = pca.transform(z_train_sn16)
        datasets['train']['labels']['sn1.6'] = {'state': y_train_sn16, 'noise': noise_train_sn16, 'angleX': angleX_train_sn16, 'angleZ': angleZ_train_sn16}
        datasets['train']['noise_array']['sn1.6'] = noise_train_sn16
        
    if sn16_test_exists:
        datasets['test']['pca']['sn1.6'] = pca.transform(z_test_sn16)
        datasets['test']['labels']['sn1.6'] = {'pdbid': pdbids_test_sn16, 'noise': noise_test_sn16, 'angleX': angleX_test_sn16, 'angleZ': angleZ_test_sn16}
        datasets['test']['noise_array']['sn1.6'] = noise_test_sn16
    
    tasks = []
    for ds_name, ds_data in datasets.items():
        for filter_name in ['all', 'sn1.6']:
            if filter_name in ds_data['pca']:
                data_full = ds_data['pca'][filter_name]
                noise_full = ds_data['noise_array'][filter_name]
                labels_dict = ds_data['labels'][filter_name]
            else:
                data_full = ds_data['pca']['all']
                noise_full = ds_data['noise_array']['all']
                labels_dict = ds_data['labels']['all']
            
            mask = (noise_full == 1.6) if filter_name == 'sn1.6' and not ('sn16_exists' in locals() and ds_name == 'train' and sn16_train_exists) else np.ones(len(data_full), dtype=bool)
            data_plot = data_full[mask]
            if len(data_plot) == 0: continue
            
            for label_name, label_full in labels_dict.items():
                labels_plot = label_full[mask]
                out_folder = os.path.join(OUTPUT_DIR, ds_name, filter_name, label_name)
                for i in range(10):
                    for j in range(10):
                        if i == j: continue
                        title = f"PCA {j+1} vs {i+1} | {ds_name.upper()} | {filter_name.upper()} | Color: {label_name}"
                        save_path = os.path.join(out_folder, f"pca_{j+1}_vs_{i+1}_rdylbu.png")
                        tasks.append((data_plot[:, j], data_plot[:, i], labels_plot, j+1, i+1, label_name, title, save_path))

    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        pool.map(plot_single_pair, tasks, chunksize=10)

if __name__ == '__main__':
    main()
