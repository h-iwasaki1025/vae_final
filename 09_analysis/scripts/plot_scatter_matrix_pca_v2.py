import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA

sys.path.insert(0, '/Users/hikaru/Code/2026_Project/06_Analysis')
import utils

model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep'
output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep'
os.makedirs(output_dir, exist_ok=True)

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

kills_train = fix_paths(np.load(os.path.join(model_dir, 'kills_train.npy'), allow_pickle=True))
z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)

# y (State) をパース
y_train = []
for p in kills_train:
    fname = os.path.basename(p)
    state, _ = utils.parse_state_pdbid(fname.split('_')[0], fname)
    if state is None:
        state = 0
    y_train.append(state)
y_train = np.array(y_train)

# PCAで10次元に変換
print('Calculating PCA (10 components)...')
pca = PCA(n_components=10)
pca_z_train = pca.fit_transform(z_mean_train)

print(f'Drawing PCA Scatter Matrix for Train data. pca_z shape: {pca_z_train.shape}')

df = pd.DataFrame(pca_z_train)
# 1-indexed for PCA labels (PCA1 to PCA10)
df.columns = [f'PCA{i+1}' for i in range(df.shape[1])]

fig = plt.figure(figsize=(15, 15))
cmap_obj = plt.colormaps.get_cmap('jet')
norm = mcolors.Normalize(vmin=np.min(y_train), vmax=np.max(y_train))
c_mapped = cmap_obj(norm(y_train))

axes = scatter_matrix(
    df, alpha=0.8, c=c_mapped, s=2, figsize=(15, 15)
)
plt.suptitle('PCA Latent Space Scatter Matrix (Train)', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])

save_name = 'latent_scatter_matrix_train_pca.png'
save_path = os.path.join(output_dir, save_name)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print('PCA Scatter matrix generation complete.')
