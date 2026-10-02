import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

# tab10 の元コード準拠カラー定義
# State 0 (Post-Power Stroke)    -> 赤 #d62728 (tab10 Red)
# State 1 (Post-Recovery Stroke) -> 青 #1f77b4 (tab10 Blue)
# State 2 (Pre-Power Stroke)     -> 緑 #2ca02c (tab10 Green)
COLOR_STATE_0_RED = "#d62728"
COLOR_STATE_1_BLUE = "#1f77b4"
COLOR_STATE_2_GREEN = "#2ca02c"

STATE_COLOR_LIST = [COLOR_STATE_0_RED, COLOR_STATE_1_BLUE, COLOR_STATE_2_GREEN]
STATE_CMAP = ListedColormap(STATE_COLOR_LIST)

def get_state_color_list(labels):
    """State 0 -> 赤, State 1 -> 青, State 2 -> 緑 のカラーリストを自動割り当て"""
    colors = []
    for st in labels:
        st_int = int(st)
        if st_int == 0:
            colors.append(COLOR_STATE_0_RED)
        elif st_int == 1:
            colors.append(COLOR_STATE_1_BLUE)
        elif st_int == 2:
            colors.append(COLOR_STATE_2_GREEN)
        else:
            colors.append(COLOR_STATE_0_RED)
    return colors

def PCA_graph(pca_z, ColorMap, CMAP="tab10_state", X=5, Y=5, AL=1.0, size=3, save_name=None, out_dir=None, title="2D PCA"):
    fig = plt.figure(figsize=(6, 6))
    
    color_vals = get_state_color_list(ColorMap)
    sc = plt.scatter(pca_z[:, 0], pca_z[:, 1], c=color_vals, s=size, alpha=AL)
    
    plt.xlim(-X, X)
    plt.ylim(-Y, Y)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    
    # 凡例の追加 (State 0: 赤, State 1: 青, State 2: 緑)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_STATE_0_RED, markersize=8, label="State 0 (Red)"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_STATE_1_BLUE, markersize=8, label="State 1 (Blue)"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_STATE_2_GREEN, markersize=8, label="State 2 (Green)")
    ]
    plt.legend(handles=handles, loc="upper right", fontsize=8)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if out_dir and save_name:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, save_name), dpi=150)
    plt.close(fig)

def check_senzai(z_mean, ColorMap, CMAP="tab10_state", AL=1.0, size=2, save_name=None, out_dir=None, title="Scatter Matrix"):
    dim = z_mean.shape[1]
    if dim <= 1:
        return
    cols = [f"z_{i+1}" for i in range(dim)]
    df = pd.DataFrame(z_mean, columns=cols)
    
    # State文字列マッピング
    state_names = {0: "State 0 (Red)", 1: "State 1 (Blue)", 2: "State 2 (Green)"}
    df["State"] = [state_names.get(int(st), f"State {st}") for st in ColorMap]

    palette = {"State 0 (Red)": COLOR_STATE_0_RED, "State 1 (Blue)": COLOR_STATE_1_BLUE, "State 2 (Green)": COLOR_STATE_2_GREEN}
    g = sns.pairplot(df, vars=cols, hue="State", palette=palette, plot_kws={"s": size, "alpha": AL}, corner=True)
    g.fig.suptitle(title, y=1.02)
    
    if out_dir and save_name:
        os.makedirs(out_dir, exist_ok=True)
        g.savefig(os.path.join(out_dir, save_name), dpi=150)
    plt.close(g.fig)
