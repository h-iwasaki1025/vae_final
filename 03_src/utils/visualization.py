import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CMAP_NAME = "RdYlBu_r"

def PCA_graph(pca_z, ColorMap, CMAP=CMAP_NAME, X=5, Y=5, AL=1.0, size=2, save_name=None, out_dir=None, title="2D PCA"):
    fig = plt.figure(figsize=(6, 6))
    sc = plt.scatter(pca_z[:, 0], pca_z[:, 1], c=ColorMap, cmap=CMAP, s=size, alpha=AL)
    plt.xlim(-X, X)
    plt.ylim(-Y, Y)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.colorbar(sc)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if out_dir and save_name:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, save_name), dpi=150)
    plt.close(fig)

def check_senzai(z_mean, ColorMap, CMAP=CMAP_NAME, AL=1.0, size=2, save_name=None, out_dir=None, title="Scatter Matrix"):
    dim = z_mean.shape[1]
    if dim <= 1:
        return
    cols = [f"z_{i+1}" for i in range(dim)]
    df = pd.DataFrame(z_mean, columns=cols)
    df["label"] = ColorMap

    g = sns.pairplot(df, vars=cols, hue="label", palette=CMAP, plot_kws={"s": size, "alpha": AL}, corner=True)
    g.fig.suptitle(title, y=1.02)
    
    if out_dir and save_name:
        os.makedirs(out_dir, exist_ok=True)
        g.savefig(os.path.join(out_dir, save_name), dpi=150)
    plt.close(g.fig)
