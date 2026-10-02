import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

# tab10 の代表色定義 (Index 0: 青 #1f77b4, Index 3: 赤 #d62728)
TAB10_BLUE = "#1f77b4"
TAB10_RED = "#d62728"

# デフォルトのカスタムカラーマップ (青と赤)
TAB10_RED_BLUE_CMAP = ListedColormap([TAB10_BLUE, TAB10_RED])

def get_custom_cmap(cmap_name="tab10_red_blue", colors=None):
    if colors is not None and len(colors) > 0:
        return ListedColormap(colors)
    if cmap_name == "tab10_red_blue":
        return TAB10_RED_BLUE_CMAP
    return plt.get_cmap(cmap_name)

def PCA_graph(pca_z, ColorMap, CMAP="tab10_red_blue", X=5, Y=5, AL=1.0, size=3, save_name=None, out_dir=None, title="2D PCA"):
    fig = plt.figure(figsize=(6, 6))
    
    cmap_obj = get_custom_cmap(CMAP)
    sc = plt.scatter(pca_z[:, 0], pca_z[:, 1], c=ColorMap, cmap=cmap_obj, s=size, alpha=AL)
    plt.xlim(-X, X)
    plt.ylim(-Y, Y)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.colorbar(sc, ticks=[0, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if out_dir and save_name:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, save_name), dpi=150)
    plt.close(fig)

def check_senzai(z_mean, ColorMap, CMAP="tab10_red_blue", AL=1.0, size=2, save_name=None, out_dir=None, title="Scatter Matrix"):
    dim = z_mean.shape[1]
    if dim <= 1:
        return
    cols = [f"z_{i+1}" for i in range(dim)]
    df = pd.DataFrame(z_mean, columns=cols)
    df["label"] = ColorMap

    palette = [TAB10_BLUE, TAB10_RED] if CMAP == "tab10_red_blue" else CMAP
    g = sns.pairplot(df, vars=cols, hue="label", palette=palette, plot_kws={"s": size, "alpha": AL}, corner=True)
    g.fig.suptitle(title, y=1.02)
    
    if out_dir and save_name:
        os.makedirs(out_dir, exist_ok=True)
        g.savefig(os.path.join(out_dir, save_name), dpi=150)
    plt.close(g.fig)
