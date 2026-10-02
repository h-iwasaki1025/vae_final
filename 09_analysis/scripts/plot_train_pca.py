import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm

model_dir = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep"
output_dir = "/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep"
os.makedirs(output_dir, exist_ok=True)

def fix_paths(paths):
    fixed = []
    for p in paths:
        p_str = str(p)
        if p_str.startswith("/Volumes/hikaru/"):
            fixed.append(p_str.replace("/Volumes/hikaru/", "/Users/hikaru/", 1))
        elif p_str.startswith("/Users/hikarui./"):
            fixed.append(p_str.replace("/Users/hikarui./", "/Users/hikaru/", 1))
        else:
            fixed.append(p_str)
    return np.array(fixed)

kills_train = fix_paths(np.load(os.path.join(model_dir, "kills_train.npy"), allow_pickle=True))
pdbids_train = np.load(os.path.join(model_dir, "pdbids_train.npy"), allow_pickle=True)
z_mean_train = np.load(os.path.join(model_dir, "z_mean_train.npy"), allow_pickle=True)

angles_train = []
snrs_train = []
for p in kills_train:
    fname = os.path.basename(p)
    try:
        parts = fname.split(".")[0].split("_")
        if parts[0] in ("0", "1", "2"):
            sn_str = parts[-1]
            z_ang = int(parts[4])
        else:
            sn_str = parts[-1]
            z_ang = int(parts[3])
        sn_val = float(sn_str.replace("sn", ""))
        angles_train.append(z_ang)
        snrs_train.append(sn_val)
    except Exception:
        angles_train.append(0)
        snrs_train.append(1.0)

angles_train = np.array(angles_train)
snrs_train = np.array(snrs_train)

pca = PCA(n_components=2)
z_train_pca = pca.fit_transform(z_mean_train)

unique_pdbs = np.unique(pdbids_train)
colors = cm.get_cmap("tab10", len(unique_pdbs))

plt.figure(figsize=(10, 8))
for i, pdb in enumerate(unique_pdbs):
    idx = np.where(pdbids_train == pdb)[0]
    plt.scatter(z_train_pca[idx, 0], z_train_pca[idx, 1], label=pdb, color=colors(i), s=10, alpha=0.5)

plt.xlabel("Principal Component 1", fontsize=14)
plt.ylabel("Principal Component 2", fontsize=14)
plt.title("PCA of Latent Space (Train Data) - by PDB", fontsize=16)
plt.legend(title="PDB IDs", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_2d_train_pdb.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_train_pca[:, 0], z_train_pca[:, 1], c=angles_train, cmap="hsv", s=10, alpha=0.5)
cbar = plt.colorbar(scatter)
cbar.set_label("Z Angle (degrees)", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=14)
plt.ylabel("Principal Component 2", fontsize=14)
plt.title("PCA of Latent Space (Train Data) - by Z Angle", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_2d_train_angle.png"), dpi=150)
plt.close()

if "1qvi" in unique_pdbs:
    plt.figure(figsize=(10, 8))
    idx_others = np.where(pdbids_train != "1qvi")[0]
    idx_1qvi = np.where(pdbids_train == "1qvi")[0]
    
    plt.scatter(z_train_pca[idx_others, 0], z_train_pca[idx_others, 1], color="lightgray", s=10, alpha=0.3, label="Others")
    plt.scatter(z_train_pca[idx_1qvi, 0], z_train_pca[idx_1qvi, 1], color="green", s=10, alpha=0.8, label="1qvi")
    
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.title("PCA of Latent Space (Train Data) - 1qvi Highlight", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_2d_train_1qvi_green.png"), dpi=150)
    plt.close()
    print("Generated 1qvi highlight plot.")
else:
    print("1qvi not found in train data, skipping 1qvi highlight plot.")

print("Train PCA plots generated successfully.")
