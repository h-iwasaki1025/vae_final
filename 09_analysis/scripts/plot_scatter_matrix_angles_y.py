import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

def plot_matrix(data, color_info, cmap_name, title, save_path):
    print(f"Plotting {title}...")
    df = pd.DataFrame(data)
    df.columns = [f'Dim{i+1}' for i in range(df.shape[1])]
    fig = plt.figure(figsize=(15, 15))
    
    cmap_obj = plt.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=360)
    c_mapped = cmap_obj(norm(color_info))
    
    scatter_matrix(df, alpha=0.8, c=c_mapped, s=2, figsize=(15, 15))
        
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved to', save_path)

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
    
    # 描画が重すぎる＆重なって見えないのを防ぐため、30度ごとのデータに間引く
    step_size = 30
    valid_idx = (angle_y_train % step_size == 0) & (angle_z_train % step_size == 0)
    
    pca_z_train_filt = pca_z_train[valid_idx]
    y_filt = angle_y_train[valid_idx]
    z_filt = angle_z_train[valid_idx]
    
    plot_matrix(pca_z_train_filt, y_filt, 'hsv', 
                'PCA Latent Space (Train) - Colored by Angle Y', 
                os.path.join(output_dir, 'latent_scatter_matrix_train_pca_AngleY.png'))
                
    plot_matrix(pca_z_train_filt, z_filt, 'hsv', 
                'PCA Latent Space (Train) - Colored by Angle Z', 
                os.path.join(output_dir, 'latent_scatter_matrix_train_pca_AngleZ.png'))

if __name__ == '__main__':
    main()
