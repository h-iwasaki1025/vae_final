import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

print(f'Drawing Scatter Matrix for Train data. z_mean shape: {z_mean_train.shape}')

utils.check_senzai(
    z_mean_train, y_train,
    CMAP='jet', AL=0.8, size=2,
    save_name='latent_scatter_matrix_train.png', out_dir=output_dir, pdf_fig_list=None,
    title='Sampled Latent Space Scatter Matrix (Train)'
)
print('Scatter matrix generation complete.')
