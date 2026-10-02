import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix
from matplotlib.backends.backend_pdf import PdfPages
import datetime

# --- General Utility ---
def print_top_counts(pdbids, k=10):
    from collections import Counter
    c = Counter(pdbids)
    print("unique pdb:", len(c))
    print("top:", c.most_common(k))

def make_pdf_title_page(date_str=None):
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    fig = plt.figure(figsize=(8.5, 6))
    plt.axis("off")
    plt.text(0.5, 0.5, f"Date: {date_str}", ha="center", va="center", fontsize=30, fontweight="bold")
    return fig

# --- Pre-processing & Data Helpers ---

def parse_state_pdbid(m: str, fname: str):
    p = m.split("_")
    if len(p) >= 2 and p[0] in ("0","1","2"):
        return int(p[0]), p[1]
    q = fname.split("_")
    if len(q) >= 2 and q[0] in ("0","1","2"):
        return int(q[0]), q[1]
    return None, None

def parse_phi(fname: str):
    parts = fname.split("_")
    if parts[0] in ("0","1","2"):
        return int(parts[2])
    return int(parts[1])

# --- Plotting Helpers ---

def plot_test_images(
    x_test, n=10, start_idx=1, pixel_size=128,
    save_name=None, out_dir=None, pdf_fig_list=None, title="Test Image Samples"
):
    j = start_idx
    fig = plt.figure(figsize=(n, 3))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        # Avoid indexing out of bounds
        if j < len(x_test):
            plt.imshow(x_test[j].reshape(pixel_size, pixel_size), cmap='gray')
        ax.axis('off')
        ax.set_title(f"idx={j}", fontsize=8)
        j += 4
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_name and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Image saved: {save_path}")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)

def plot_recon_kl(history, out_dir=None, save_name=None, pdf_fig_list=None):
    if isinstance(history, dict):
        h = history
    elif hasattr(history, 'history'):
        h = history.history
    else:
        return
        
    epochs = range(1, len(h["loss"]) + 1)
    fig = plt.figure()
    
    if "recon_loss" in h and "kl_loss" in h:
        plt.plot(epochs, h["recon_loss"], label="recon_loss")
        plt.plot(epochs, h["kl_loss"], label="kl_loss")
        if "val_recon_loss" in h: plt.plot(epochs, h["val_recon_loss"], label="val_recon_loss")
        if "val_kl_loss" in h:    plt.plot(epochs, h["val_kl_loss"], label="val_kl_loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Recon vs KL")
    plt.legend()
    plt.tight_layout()
    
    if save_name and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)

def plot_loss(
    result, ylim=None, show_val=True, show_acc=True,
    title="Learning Curve", save_name=None, out_dir=None, pdf_fig_list=None
):
    if isinstance(result, dict):
        history = result
    elif hasattr(result, 'history'):
        history = result.history
    else:
        return
        
    fig = plt.figure(figsize=(7,5))
    plt.plot(history['loss'], label='loss', color='r')
    if show_val and 'val_loss' in history:
        plt.plot(history['val_loss'], label='val_loss', color='b')
    if show_acc and 'accuracy' in history:
        plt.plot(history['accuracy'], label='accuracy', color='g')
    if show_acc and 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='val_accuracy', color='m')
        
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel('loss / accuracy')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_name and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)

# --- Post-Analysis Plotting ---

def check_senzai(
    z_mean, ColorMap, CMAP, AL, size,
    save_name=None, out_dir=None, pdf_fig_list=None, title="Latent Space Scatter Matrix"
):
    df = pd.DataFrame(z_mean)
    df.columns = [f'z{i}' for i in range(df.shape[1])]
    fig = plt.figure(figsize=(15, 15))
    
    # Map scalar values to RGBA colors for scatter_matrix
    if isinstance(ColorMap, (list, np.ndarray)) and len(ColorMap) > 0 and isinstance(ColorMap[0], (int, float, np.integer, np.floating)):
        cmap_obj = plt.colormaps.get_cmap(CMAP)
        norm = mcolors.Normalize(vmin=np.min(ColorMap), vmax=np.max(ColorMap))
        c_mapped = cmap_obj(norm(ColorMap))
    else:
        c_mapped = ColorMap
        
    axes = scatter_matrix(
        df.iloc[:, :], alpha=AL, c=c_mapped, s=5, figsize=(15, 15)
    )
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_name and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)
    return df

def PCA_graph(
    pca_z_mean, ColorMap, CMAP, X, Y, AL, size,
    save_name=None, out_dir=None, pdf_fig_list=None, title="2D PCA of Latent Space",
    xlabel="PC1", ylabel="PC2"
):
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(111)
    c_array = np.array(ColorMap)
    if c_array.ndim > 1:
        if c_array.shape[1] == 1:
            c_array = c_array.flatten()
            
    mappable = ax1.scatter(pca_z_mean[:, 0], pca_z_mean[:, 1], c=c_array, cmap=CMAP, alpha=AL, s=size)
    
    unique_labels = np.unique(ColorMap)
    if len(unique_labels) <= 3:
        handles, _ = mappable.legend_elements()
        legend_labels = [f"State {int(lbl)}" for lbl in unique_labels]
        ax1.legend(handles, legend_labels, title="State")
    else:
        fig.colorbar(mappable, ax=ax1)
        
    ax1.set_xlim(X * -1, X)
    ax1.set_ylim(Y * -1, Y)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel(ylabel, fontsize=18)
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_name and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)
    return pca_z_mean

def PCA_graph_2(
    pca_z_mean, ColorMap, CMAP, X, Y, AL, size, deg,
    save_name=None, out_dir=None, pdf_fig_list=None, title="Rotated 2D PCA of Latent Space"
):
    deg_rad = np.deg2rad(deg)
    cos, sin = np.cos(deg_rad), np.sin(deg_rad)
    dp = np.zeros((len(pca_z_mean), 2))
    for i in range(len(pca_z_mean)):
        dp[i, 0] = pca_z_mean[i, 0] * cos - pca_z_mean[i, 1] * sin
        dp[i, 1] = pca_z_mean[i, 0] * sin + pca_z_mean[i, 1] * cos
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(111)
    mappable = ax1.scatter(dp[:, 0], dp[:, 1], c=ColorMap, cmap=CMAP, alpha=AL, s=size)
    
    unique_labels = np.unique(ColorMap)
    if len(unique_labels) <= 3:
        handles, _ = mappable.legend_elements()
        legend_labels = [f"State {int(lbl)}" for lbl in unique_labels]
        ax1.legend(handles, legend_labels, title="State")
    else:
        fig.colorbar(mappable, ax=ax1)
        
    ax1.set_xlim(X * -1, X)
    ax1.set_ylim(Y * -1, Y)
    ax1.set_xlabel("PC1 (rotated)", fontsize=18)
    ax1.set_ylabel("PC2 (rotated)", fontsize=18)
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_name and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)
    return dp

def hitstgram_2D(
    pca_z_mean, zyouken, z0, bins, rMin, rMax,
    save_name=None, out_dir=None, pdf_fig_list=None, title="2-State Histogram"
):
    red = []
    blue = []
    for i, x in enumerate(pca_z_mean):
        if zyouken[i] == 0:
            red.append(x[z0])
        else:
            blue.append(x[z0])
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    ax1.hist(red, alpha=0.5, color='r', label='State 0', bins=bins, range=(rMin, rMax), density=True)
    ax1.hist(blue, alpha=0.5, color='b', label='State 1', bins=bins, range=(rMin, rMax), density=True)
    ax1.legend()
    ax1.set_xlabel(f"PC{z0}", fontsize=16)
    ax1.set_ylabel("Density", fontsize=16)
    plt.title(title)
    plt.tight_layout()
    if save_name and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)

def imscatter(x, y, image_list, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        im_list = []
        for p in image_list:
            if isinstance(p, str) or hasattr(p, "startswith") or (hasattr(p, "__fspath__")):
                img = plt.imread(str(p))
            else:
                img = p # Assume it's a numpy array
            
            # Normalize array to 0-1 if it's float, or keep as is. Add colormap if grayscale
            cmap = 'gray' if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1) else None
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[:, :, 0] # remove channel dim for OffsetImage
                
            im_list.append(OffsetImage(img, zoom=zoom, cmap=cmap))
            
        x, y = np.atleast_1d(x, y)
        for x0, y0, im in zip(x, y, im_list):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
    except Exception as e:
        print(f"Warning: Failed to create imscatter overlays: {e}")
    return ax

def make_decoder(x, y, im_list, Xz, Yz, zoom,
                 save_name="latent_imscatter.png", out_dir=None, pdf_fig_list=None, dpi=200):
    fig, ax = plt.subplots(figsize=(30, 30))
    imscatter(x, y, im_list, ax=ax, zoom=zoom)

    ax.set_xlim(-Xz, Xz)
    ax.set_ylim(-Yz, Yz)
    ax.set_xlabel("PC1", fontsize=50)
    ax.set_ylabel("PC2", fontsize=50)
    ax.tick_params(labelsize=30)
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)

def plot_pca_pairs_all(
    z_mean, labels, out_dir=None, pdf_fig_list=None,
    prefix="pca_scatter", alpha=0.5, size=8, max_dim=10, cmap_name='RdYlBu_r'
):
    mask = (labels == 0) | (labels == 1)
    z = z_mean[mask]
    labs = labels[mask]

    n_comp_actual = min(z.shape[1], max_dim)
    pca = PCA(n_components=n_comp_actual)
    z_pca = pca.fit_transform(z)
    N_dim = z_pca.shape[1]

    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    color_array = cmap(norm(labs))

    for i in range(N_dim):
        for j in range(i+1, N_dim):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                z_pca[:, i], z_pca[:, j],
                c=color_array, alpha=alpha, s=size, edgecolors='none'
            )
            ax.set_xlabel(f'PC{i+1}', fontsize=15)
            ax.set_ylabel(f'PC{j+1}', fontsize=15)
            ax.set_title(f"PCA: PC{i+1} vs PC{j+1}", fontsize=16)
            ax.grid(True, linestyle=':', alpha=0.4)
            plt.tight_layout()

            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                fname = f"{prefix}_PC{i+1}_PC{j+1}.png"
                save_path = os.path.join(out_dir, fname)
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if pdf_fig_list is not None:
                pdf_fig_list.append(fig)
            plt.close(fig)

def plot_pca_variance(z_mean, out_dir=None, pdf_fig_list=None, n_components=None, threshold=0.90, title="PCA Explained Variance"):
    if n_components is None:
        n_components = z_mean.shape[1]
    n_components = min(n_components, z_mean.shape[1], z_mean.shape[0])
    pca = PCA(n_components=n_components)
    pca_z = pca.fit_transform(z_mean)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    fig = plt.figure(figsize=(10, 6))
    components = np.arange(1, len(explained_variance_ratio) + 1)
    plt.bar(components, explained_variance_ratio, alpha=0.6, label='Individual Explained Variance')
    plt.plot(components, cumulative_variance, marker='o', color='r', label='Cumulative Explained Variance')

    k = np.argmax(cumulative_variance >= threshold) + 1
    plt.axvline(k, color='g', linestyle='--', label=f'{k} PCs = {threshold*100:.0f}% var')
    plt.text(k + 0.2, 0.5, f'{k} components\\ncover {threshold*100:.0f}%', color='green')
    plt.xticks(components)
    plt.xlabel('Principal Component', fontsize=14)
    plt.ylabel('Explained Variance Ratio', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.legend(loc='best')

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"pca_variance_plot_{ts}.png"
        plt.savefig(os.path.join(out_dir, save_name), dpi=300, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)
    return pca, explained_variance_ratio, cumulative_variance, k

def plot_pca_by_pdbid_state(
    z_mean_test, y_test, pdbids_test, state=0,
    out_dir=None, pdf_fig_list=None, basename="pca_pdb", dpi=300, pca_z=None,
    xlabel="PC1", ylabel="PC2"
):
    mask  = np.array(y_test) == state
    if not np.any(mask):
        print(f"No data points found for state={state}. Skipping plot_pca_by_pdbid_state.")
        return
        
    if pca_z is None:
        pca_z_filtered = PCA(n_components=2).fit_transform(z_mean_test[mask])
    else:
        # すでに外で全体のPCA変換が行われている場合は、マスクで該当する点だけ切り出す
        pca_z_filtered = pca_z[mask]
        
    pdbs  = [pid for pid, flag in zip(pdbids_test, mask) if flag]
    unique = sorted(set(pdbs))
    unique_all = sorted(set(pdbids_test))
    n_pdb_all  = len(unique_all)
    
    # Ensure there are at least two PDBs to compare
    if len(unique) == 0:
        return
        
    cmap = plt.get_cmap("tab10" if n_pdb_all <= 10 else "tab20")
    
    fig1, axes = plt.subplots(max(2, (len(unique) + 2) // 3), 3, figsize=(15, 4 * max(2, (len(unique) + 2) // 3)), sharex=True, sharey=True)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    
    for i, pid in enumerate(unique):
        if i >= len(axes): break
        ax = axes[i]
        idx = [j for j, p in enumerate(pdbs) if p == pid]
        color_idx = unique_all.index(pid)
        ax.scatter(
            pca_z_filtered[idx, 0], pca_z_filtered[idx, 1],
            color=cmap(color_idx), s=10, alpha=0.6, edgecolors='none'
        )
        ax.set_title(pid, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.4)
        
    for j in range(len(unique), len(axes)):
        fig1.delaxes(axes[j])
        
    state_title = f"state={state}" if state is not None else ""
    fig1.suptitle(f"PCA — individual PDBs ({state_title})", fontsize=16)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
 
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        f1 = os.path.join(out_dir, f"{basename}_individual_state{state}.png")
        fig1.savefig(f1, dpi=dpi, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig1)
    plt.close(fig1)
 
    fig2, ax = plt.subplots(figsize=(8, 8))
    for i, pid in enumerate(unique):
        idx = [j for j, p in enumerate(pdbs) if p == pid]
        color_idx = unique_all.index(pid)
        if color_idx < cmap.N:
            color = cmap(color_idx)
        else:
            color = 'black'
        ax.scatter(
            pca_z_filtered[idx, 0], pca_z_filtered[idx, 1],
            color=color, s=10, alpha=0.6, label=pid, edgecolors='none'
        )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f"PCA — all PDBs ({state_title})", fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    if len(unique) > 0:
        ax.legend(
            title="PDB ID",
            bbox_to_anchor=(1.02, 1),
            loc='upper left', borderaxespad=0.5,
            fontsize=9, title_fontsize=10
        )
    fig2.tight_layout()
    if out_dir:
        f2 = os.path.join(out_dir, f"{basename}_all_state{state}.png")
        fig2.savefig(f2, dpi=dpi, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig2)
    plt.close(fig2)


def plot_pca_trajectories_by_pdb(
    z_mean_test, r_test, pdbids_test, y_test,
    out_dir=None, pdf_fig_list=None, basename="06_pca_trajectories", dpi=300, pca_z=None
):
    """
    PDB IDごとに、回転角度順に点を結んだ「軌跡（Trajectory）」を描画する。
    """
    if pca_z is None:
        pca_z = PCA(n_components=2).fit_transform(z_mean_test)
    pdbs = np.array(pdbids_test)
    rots = np.array(r_test)
    unique_pdbs = sorted(set(pdbids_test))
    n_pdb = len(unique_pdbs)
    
    if n_pdb == 0:
        return

    # サブプロットの配置
    cols = 3
    rows = (n_pdb + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True, sharey=True)
    if n_pdb == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    # Tab10/Tab20 colormap for PDBs
    cmap_pdb = plt.get_cmap("tab10" if n_pdb <= 10 else "tab20")
    for i, pid in enumerate(unique_pdbs):
        ax = axes[i]
        mask = pdbs == pid
        p_z = pca_z[mask]
        p_r = rots[mask]
        p_y = np.array(y_test)[mask]
        
        # 回転角度(r_test)でソート
        sort_idx = np.argsort(p_r)
        p_z_sorted = p_z[sort_idx]
        p_r_sorted = p_r[sort_idx]
        p_y_sorted = p_y[sort_idx]
        
        # --- (A) 統合プロット用の描画 (PDBごとの色) ---
        c = cmap_pdb(i)
        ax.scatter(p_z_sorted[:, 0], p_z_sorted[:, 1],
                   color=c, s=15, alpha=0.8, edgecolors='none')
        ax.set_title(f"PDB: {pid}", fontsize=12, color=c)
        ax.grid(True, linestyle=':', alpha=0.5)

        # --- (B) 個別プロットの作成と保存 ---
        fig_ind, ax_ind = plt.subplots(figsize=(8, 7))
        # 角度(Rotation)で色付けした散布図。
        sc = ax_ind.scatter(p_z_sorted[:, 0], p_z_sorted[:, 1], c=p_r_sorted, 
                            cmap='RdYlBu_r', s=5, alpha=0.8, edgecolors='none')
        
        ax_ind.set_xlabel("PC1", fontsize=12)
        ax_ind.set_ylabel("PC2", fontsize=12)
        ax_ind.set_title(f"Trajectory: {pid}\n(Colored by Rotation Angle)", fontsize=14)
        plt.colorbar(sc, ax=ax_ind, label='Rotation Angle')
        ax_ind.grid(True, linestyle=':', alpha=0.5)
        
        if out_dir:
            ind_path = os.path.join(out_dir, f"trajectory_{pid}.png")
            fig_ind.savefig(ind_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_ind)
        
    for j in range(n_pdb, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle("PCA Trajectories by PDB ID (Ordered by Rotation Angle)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{basename}.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if pdf_fig_list is not None:
        pdf_fig_list.append(fig)
    plt.close(fig)

def HeatMap(data, zyouken, wide, div, z0, z1,
            out_dir=None, prefix="heatmap", dpi=200):
    test = data
    count = int((wide * 2) / div + 1)
    
    zen1 = zen2 = 0
    red  = np.zeros((count, count))
    blue = np.zeros((count, count))

    for i in range(len(test)):
        x = test[i][z0]
        y = test[i][z1]
        label = int(zyouken[i])
        xi = int((x + wide) / div)
        yi = int((wide - y) / div)
        if 0 <= xi < count and 0 <= yi < count:
            if label == 1:
                red[yi][xi] += 1; zen1 += 1
            elif label == 0:
                blue[yi][xi] += 1; zen2 += 1

    red2  = red  / np.max(red)  if np.max(red)  != 0 else red
    blue2 = blue / np.max(blue) if np.max(blue) != 0 else blue

    ticks = [int(count * i / 10) for i in range(11)]
    tick_labels = ['-5','-4','-3','-2','-1','0','1','2','3','4','5']

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _plot(mat, cmap, title, fname, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(12, 10))
        hm = ax.pcolor(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(hm)
        ax.invert_yaxis()
        ax.set_xlabel("Z"+str(z0), fontsize=25)
        ax.set_ylabel("Z"+str(z1), fontsize=25)
        ax.set_xticks(ticks); ax.set_xticklabels(tick_labels)
        ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)
        ax.set_title(title)
        if out_dir:
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    _plot(blue2, cm.Reds,  "Label = 0", f"{prefix}_z{z0}z{z1}_label0.png")
    _plot(red2,  cm.Blues, "Label = 1", f"{prefix}_z{z0}z{z1}_label1.png")

    diff = red2 - blue2
    vmax = np.max(np.abs(diff)) if np.max(np.abs(diff)) != 0 else 1.0
    _plot(diff, cm.bwr, "Difference: Label1 - Label0",
          f"{prefix}_z{z0}z{z1}_diff.png", vmin=-vmax, vmax=vmax)

def check_latent_collapse(encoder, x, pixel_size=128, max_n=20000, title_prefix="train", out_dir=None, pdf_fig_list=None):
    n = min(len(x), max_n)
    idx = np.random.choice(len(x), n, replace=False)
    xs = x[idx]

    if xs.ndim == 3:
        xs = xs[..., None]

    out = encoder.predict(xs, batch_size=min(1024, n), verbose=0)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        z_mean, z_log_var = out[0], out[1]
    else:
        z_mean, z_log_var = out, None

    z_mean = np.asarray(z_mean)
    per_dim_std = z_mean.std(axis=0)
    overall_std = z_mean.std()
    
    print(f"[{title_prefix}] z_mean shape: {z_mean.shape}")
    print(f"[{title_prefix}] overall std(z_mean): {overall_std:.6g}")
    print(f"[{title_prefix}] per-dim std min/median/max: "
          f"{per_dim_std.min():.6g} / {np.median(per_dim_std):.6g} / {per_dim_std.max():.6g}")

    if z_log_var is not None:
        z_log_var = np.asarray(z_log_var)
        print(f"[{title_prefix}] z_log_var mean: {z_log_var.mean():.6g}, std: {z_log_var.std():.6g}")

    n_comp_actual = min(2, z_mean.shape[1], z_mean.shape[0])
    if n_comp_actual >= 2:
        pca = PCA(n_components=2)
        z2 = pca.fit_transform(z_mean)
        evr = pca.explained_variance_ratio_
        print(f"[{title_prefix}] PCA explained variance ratio: {evr[0]:.4f}, {evr[1]:.4f}")

        fig = plt.figure(figsize=(7,6))
        plt.scatter(z2[:,0], z2[:,1], s=2, alpha=0.6)
        plt.title(f"{title_prefix}: PCA(z_mean)  EVR={evr[0]:.3f},{evr[1]:.3f}")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{title_prefix}_latent_collapse.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        if pdf_fig_list is not None:
            pdf_fig_list.append(fig)
        plt.close(fig)
    else:
        evr = [np.nan, np.nan]

    return {"z_mean": z_mean, "z_log_var": z_log_var, "pca_evr": evr, "per_dim_std": per_dim_std}

def plot_training_history(csv_path, out_dir=None, dpi=300):
    if not os.path.exists(csv_path):
        print(f"Warning: training history file not found at {csv_path}")
        return
        
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    df = pd.read_csv(csv_path)
    
    if 'epoch' not in df.columns:
        print("Warning: 'epoch' column not found in training history.")
        return
        
    epochs = df['epoch'].values
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    metrics = [
        ('loss', 'val_loss', 'Total Loss'),
        ('reconstruction_loss', 'val_reconstruction_loss', 'Reconstruction Loss'),
        ('kl_loss', 'val_kl_loss', 'KL Loss')
    ]
    
    for i, (train_col, val_col, title) in enumerate(metrics):
        ax = axes[i]
        if train_col in df.columns:
            ax.plot(epochs, df[train_col], label='Train', color='tab:blue')
        if val_col in df.columns:
            val_df = df.dropna(subset=[val_col])
            val_data = pd.to_numeric(val_df[val_col], errors='coerce')
            val_df = val_df.dropna(subset=[val_col])
            if not val_data.empty and val_data.notna().any():
                ax.plot(val_df['epoch'], val_data[val_data.notna()], label='Validation', color='tab:orange', marker='o', markersize=3)
                
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
    axes[-1].set_xlabel('Epoch', fontsize=12)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(50))
    
    plt.tight_layout()
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "training_history_plot.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_pca_pairs_sparse(pca_z, y, angle_x, angle_z, out_dir, step_size=30, prefix="train"):
    """
    10次元PCAスコアのすべてのペアワイズ2次元散布図を、State/AngleX/AngleZの3パターンで一括出力する。
    データ密集を避けるため、angle_x と angle_z を用いて step_size 刻みで間引く。
    """
    import logging
    logger = logging.getLogger("AnalysisMain")
    base_output_dir = os.path.join(out_dir, f"pca_pairwise_{prefix}")
    os.makedirs(base_output_dir, exist_ok=True)
    for sub in ['State', 'AngleX', 'AngleZ']:
        os.makedirs(os.path.join(base_output_dir, sub), exist_ok=True)

    # フィルタリング
    valid_idx = (angle_x % step_size == 0) & (angle_z % step_size == 0)
    pca_z_sub = pca_z[valid_idx]
    y_sub = y[valid_idx]
    ax_sub = angle_x[valid_idx]
    az_sub = angle_z[valid_idx]

    logger.info(f"Generating PCA pairs ({prefix}). Filtered data: {len(pca_z_sub)} (step={step_size})")

    n_comp = pca_z.shape[1]
    
    def _plot_2d(i, j, color_array, cmap_name, save_dir, title_suffix, cbar_label):
        fig, ax = plt.subplots(figsize=(8, 6))
        x_val = pca_z_sub[:, i]
        y_val = pca_z_sub[:, j]
        
        if cmap_name == 'jet' and len(np.unique(color_array)) < 10:
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
        
        save_name = f'pca_{i+1}_vs_{j+1}.png'
        plt.savefig(os.path.join(save_dir, save_name), dpi=100)
        plt.close(fig)

    count = 0
    for i in range(n_comp):
        for j in range(n_comp):
            if i == j: continue
            _plot_2d(i, j, y_sub, 'jet', os.path.join(base_output_dir, 'State'), 'Color: State', 'State')
            _plot_2d(i, j, ax_sub, 'RdYlBu', os.path.join(base_output_dir, 'AngleX'), 'Color: Angle X', 'X Angle (deg)')
            _plot_2d(i, j, az_sub, 'RdYlBu', os.path.join(base_output_dir, 'AngleZ'), 'Color: Angle Z', 'Z Angle (deg)')
            count += 3

    logger.info(f"Finished generating {count} PCA pair plots in {base_output_dir}")
