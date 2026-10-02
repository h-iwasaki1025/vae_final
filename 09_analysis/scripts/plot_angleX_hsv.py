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

def main():
    model_name = "20260609x_z_ep"
    model_dir = f'/Users/hikaru/Code/2026_Project/04_Savemodel/{model_name}'
    # 出力先は新しいフォルダ (pca_pairwise_train 内に作成)
    base_output_dir = f'/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_{model_name}'
    
    if not os.path.exists(os.path.join(model_dir, 'z_mean_train.npy')):
        data_dir = base_output_dir
    else:
        data_dir = model_dir

    print("Loading train data...")
    z_mean_train = np.load(os.path.join(data_dir, 'z_mean_train.npy'), allow_pickle=True)
    kills_train = fix_paths(np.load(os.path.join(data_dir, 'kills_train.npy'), allow_pickle=True))
    
    angle_x_train = []
    angle_z_train = []
    for p in kills_train:
        fname = os.path.basename(p)
        try:
            parts = fname.split('.')[0].split('_')
            if parts[0] in ('0', '1', '2'):
                x_ang = int(parts[2])
                z_ang = int(parts[4])
            else:
                x_ang = int(parts[1])
                z_ang = int(parts[3])
            angle_x_train.append(x_ang)
            angle_z_train.append(z_ang)
        except Exception:
            angle_x_train.append(0)
            angle_z_train.append(0)

    angle_x_train = np.array(angle_x_train)
    angle_z_train = np.array(angle_z_train)

    print("Calculating PCA (10 components) on train data...")
    pca = PCA(n_components=10)
    
    # PCAの構築は今までと同様にTestデータにFitするか、TrainにFitするか。
    # 以前の `plot_pca_pairs.py` ではTrainで構築していましたが、最新の `main.py` および `run_all_pca_pairs.py` はTestで構築しています。
    # ユーザーが求めているのは "pca_pairwise/AngleX" であり、直近の Train用の画像なので、今回は `main.py` の動作に合わせてTestベースのPCA空間に射影します。
    # もしくは単純化のためTrain全体でFitしても大差ありませんが、一貫性のためTestデータからFitします。
    z_mean_test = np.load(os.path.join(data_dir, 'z_mean_test.npy'), allow_pickle=True)
    import random
    random.seed(720)
    np.random.seed(720)
    n_vis = min(len(z_mean_test), 20000)
    idx_fit = np.random.choice(len(z_mean_test), n_vis, replace=False)
    pca.fit(z_mean_test[idx_fit])

    pca_z_train_full = pca.transform(z_mean_train)

    # 間引き (30度)
    step_size = 30
    valid_idx = (angle_x_train % step_size == 0) & (angle_z_train % step_size == 0)
    
    pca_z_train = pca_z_train_full[valid_idx]
    angle_x_sub = angle_x_train[valid_idx]

    print(f"Filtered Train data shape: {pca_z_train.shape}")

    # 新しいフォルダ
    out_dir = os.path.join(base_output_dir, 'pca_pairwise_train', 'AngleX_hsv')
    os.makedirs(out_dir, exist_ok=True)

    # プロット関数 (HSVを使用)
    def plot_2d_angleX(i, j):
        fig, ax = plt.subplots(figsize=(8, 6))
        x_val = pca_z_train[:, i]
        y_val = pca_z_train[:, j]
        
        # hsv の正規化 (角度は一般的に 0-360)
        norm = mcolors.Normalize(vmin=np.min(angle_x_sub), vmax=np.max(angle_x_sub))
        
        scatter = ax.scatter(x_val, y_val, c=angle_x_sub, cmap='hsv', s=6, alpha=0.8, norm=norm, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('X Angle (deg)', fontsize=12)
        
        ax.set_xlabel(f'PCA {i+1}', fontsize=14)
        ax.set_ylabel(f'PCA {j+1}', fontsize=14)
        ax.set_title(f'PCA {i+1} vs PCA {j+1} - Color: Angle X (hsv)', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pca_{i+1}_vs_{j+1}.png'), dpi=100)
        plt.close(fig)

    print("Generating HSV plots for AngleX...")
    count = 0
    for i in range(10):
        for j in range(10):
            if i == j: continue
            plot_2d_angleX(i, j)
            count += 1
            
    print(f"Generated {count} plots in {out_dir}")

if __name__ == "__main__":
    main()
