import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import multiprocessing as mp
import itertools

MODEL_DIR  = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep"
OUTPUT_DIR = "/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep/pca_individual"

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

def plot_single_pair(args):
    data_x, data_y, labels, pc_x, pc_y, label_type, title, save_path = args
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        labels = np.array([label_map[lbl] for lbl in labels])
        vmin, vmax = -0.5, 9.5

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(data_x, data_y, c=labels, cmap=cmap, vmin=vmin, vmax=vmax, s=5, alpha=0.6)
    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.title(title, fontsize=12)
    
    if label_type not in ['state', 'pdbid']:
        plt.colorbar(sc, label=label_type)
    else:
        plt.colorbar(sc, ticks=np.arange(len(set(labels))), label=label_type)
        
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    print("Loading data for PCA individual plots...")
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
    
    datasets = {
        'train': {
            'pca': z_pca_train,
            'labels': {
                'state': y_train, 'noise': noise_train, 'angleY': angleY_train, 'angleZ': angleZ_train
            },
            'noise_array': noise_train
        },
        'test': {
            'pca': z_pca_test,
            'labels': {
                'pdbid': pdbids_test, 'noise': noise_test, 'angleY': angleY_test, 'angleZ': angleZ_test
            },
            'noise_array': noise_test
        }
    }
    
    tasks = []
    
    for ds_name, ds_data in datasets.items():
        data_full = ds_data['pca']
        
        for filter_name in ['all', 'sn1.6']:
            if filter_name == 'sn1.6':
                mask = (ds_data['noise_array'] == 1.6)
            else:
                mask = np.ones(len(data_full), dtype=bool)
            
            data_plot = data_full[mask]
            
            for label_name, label_full in ds_data['labels'].items():
                labels_plot = label_full[mask]
                
                out_folder = os.path.join(OUTPUT_DIR, ds_name, filter_name, label_name)
                
                # Generate all 90 pairs (i!=j)
                for i in range(10):
                    for j in range(10):
                        if i == j: continue
                        
                        title = f"PCA {j+1} vs {i+1} | {ds_name.upper()} | {filter_name.upper()} | Color: {label_name}"
                        filename = f"pca_{j+1}_vs_{i+1}.png"
                        save_path = os.path.join(out_folder, filename)
                        
                        # Note: we plot data_plot[:, j] on X-axis and data_plot[:, i] on Y-axis
                        args = (data_plot[:, j], data_plot[:, i], labels_plot, j+1, i+1, label_name, title, save_path)
                        tasks.append(args)

    print(f"Total individual PCA plots to generate: {len(tasks)}")
    
    # Process in parallel
    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        # Use chunksize for efficiency with many small tasks
        pool.map(plot_single_pair, tasks, chunksize=10)
        
    print("Done generating individual PCA plots.")

if __name__ == '__main__':
    main()
