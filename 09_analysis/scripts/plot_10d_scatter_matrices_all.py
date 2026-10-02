import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import multiprocessing as mp

MODEL_DIR  = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep"
OUTPUT_DIR = "/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep/scatter_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', str(path))
    if match: return float(match.group(1))
    return 0.0

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

def plot_scatter_matrix(data, labels, label_type, title, save_path):
    # data: (N, 10)
    # labels: (N,)
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    if label_type in ['angleY', 'angleZ']:
        cmap = 'hsv'
        vmin, vmax = 0, 360
    elif label_type == 'noise':
        cmap = 'jet'
        vmin, vmax = np.min(labels), np.max(labels)
    elif label_type == 'state':
        cmap = 'tab10'
        vmin, vmax = -0.5, 9.5
    else: # pdbid
        cmap = 'tab10'
        unique_labels = list(set(labels))
        unique_labels.sort()
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels = np.array([label_map[lbl] for lbl in labels])
        vmin, vmax = -0.5, 9.5

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
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    if label_type not in ['state', 'pdbid']:
        fig.colorbar(sc, cax=cbar_ax)
    else:
        fig.colorbar(sc, cax=cbar_ax, ticks=np.arange(len(set(labels))))
        
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def worker(args):
    plot_scatter_matrix(*args)

def main():
    print("Loading data...")
    z_train = np.load(os.path.join(MODEL_DIR, "z_mean_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"), allow_pickle=True)
    kills_train = np.load(os.path.join(MODEL_DIR, "kills_train.npy"), allow_pickle=True)
    
    z_test = np.load(os.path.join(MODEL_DIR, "z_mean_test.npy"), allow_pickle=True)
    pdbids_test = np.load(os.path.join(MODEL_DIR, "pdbids_test.npy"), allow_pickle=True)
    kills_test = np.load(os.path.join(MODEL_DIR, "kills_test.npy"), allow_pickle=True)
    
    # Parse attributes
    noise_train = np.array([parse_noise(p) for p in kills_train])
    angleY_train = np.array([parse_angles(os.path.basename(p))[0] for p in kills_train])
    angleZ_train = np.array([parse_angles(os.path.basename(p))[1] for p in kills_train])
    
    noise_test = np.array([parse_noise(p) for p in kills_test])
    angleY_test = np.array([parse_angles(os.path.basename(p))[0] for p in kills_test])
    angleZ_test = np.array([parse_angles(os.path.basename(p))[1] for p in kills_test])

    # PCA Fitting
    print("Fitting PCA on Test data...")
    np.random.seed(720)
    sample_indices = np.random.choice(len(z_test), min(20000, len(z_test)), replace=False)
    pca = PCA(n_components=10)
    pca.fit(z_test[sample_indices])
    
    z_pca_train = pca.transform(z_train)
    z_pca_test = pca.transform(z_test)
    
    tasks = []
    
    datasets = {
        'train': {
            'raw': z_train, 'pca': z_pca_train,
            'labels': {
                'state': y_train, 'noise': noise_train, 'angleY': angleY_train, 'angleZ': angleZ_train
            },
            'noise_array': noise_train
        },
        'test': {
            'raw': z_test, 'pca': z_pca_test,
            'labels': {
                'pdbid': pdbids_test, 'noise': noise_test, 'angleY': angleY_test, 'angleZ': angleZ_test
            },
            'noise_array': noise_test
        }
    }
    
    for ds_name, ds_data in datasets.items():
        for space_name in ['raw', 'pca']:
            data_full = ds_data[space_name]
            
            for filter_name in ['all', 'sn1.6']:
                if filter_name == 'sn1.6':
                    mask = (ds_data['noise_array'] == 1.6)
                else:
                    mask = np.ones(len(data_full), dtype=bool)
                
                data_plot = data_full[mask]
                
                for label_name, label_full in ds_data['labels'].items():
                    labels_plot = label_full[mask]
                    
                    title = f"10D Scatter Matrix | {ds_name.upper()} | {space_name.upper()} | {filter_name.upper()} | Color: {label_name}"
                    filename = f"scatter_{ds_name}_{space_name}_{filter_name}_{label_name}.png"
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    
                    tasks.append((data_plot, labels_plot, label_name, title, save_path))

    print(f"Total matrices to generate: {len(tasks)}")
    
    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        pool.map(worker, tasks)
        
    print("Done generating 10D Scatter Matrices.")

if __name__ == '__main__':
    main()
