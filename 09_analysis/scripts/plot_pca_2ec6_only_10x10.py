#!/usr/bin/env python3
"""2ec6 only: PCA individual pairwise plots (10x10) + one matrix figure."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, "/Users/hikaru/Code/2026_Project/06_Analysis_refactored")
from analysis.individual import plot_single_pair

MODEL_DIR = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260603_z_noise10"
OUT_ROOT = (
    "/Users/hikaru/Code/2026_Project/07_result/"
    "20260607_z_noise10_20260603_z_noise10/angle_error_audit/"
    "pca_individual_2ec6"
)
OUT_PAIR = os.path.join(OUT_ROOT, "test", "all", "angleZ")
OUT_MATRIX = os.path.join(OUT_ROOT, "test", "all")


def main():
    os.makedirs(OUT_PAIR, exist_ok=True)

    z_test = np.load(os.path.join(MODEL_DIR, "z_mean_test.npy"))
    pdbids = np.load(os.path.join(MODEL_DIR, "pdbids_test.npy"), allow_pickle=True).astype(str)
    angles = np.load(os.path.join(MODEL_DIR, "z_angle_test.npy")).astype(int)

    mask = pdbids == "2ec6"
    z_2 = z_test[mask]
    ang_2 = angles[mask]
    print(f"2ec6 samples: {len(z_2)}")

    np.random.seed(720)
    sample_indices = np.random.choice(len(z_test), min(20000, len(z_test)), replace=False)
    pca = PCA(n_components=10, random_state=42)
    pca.fit(z_test[sample_indices])
    z_pca = pca.transform(z_2)
    print("explained_var:", np.round(pca.explained_variance_ratio_, 4))

    config = {
        "plot_settings": {
            "global": {"show_legend": True, "figsize_square": [6, 5], "fontsize": 12},
            "colors": {"pdb_colors": {"2ec6": "#fc8d62", "1l2o": "#1f77b4"}},
        },
        "individual": {"alpha": 0.8, "marker_size": 5},
    }

    tasks = []
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            pc_x, pc_y = j + 1, i + 1
            title = f"PCA {pc_x} vs {pc_y} | TEST | 2ec6 only | Color: angleZ"
            save_path = os.path.join(OUT_PAIR, f"pca_{pc_x}_vs_{pc_y}.png")
            tasks.append(
                (
                    z_pca[:, j],
                    z_pca[:, i],
                    ang_2,
                    pc_x,
                    pc_y,
                    "angleZ",
                    title,
                    save_path,
                    config,
                )
            )

    print(f"generating {len(tasks)} pairwise plots (serial)...")
    for t in tasks:
        plot_single_pair(t)

    print("generating 10x10 matrix...")
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(10):
        for j in range(10):
            ax = axes[i, j]
            if i == j:
                ax.hist(z_pca[:, i], bins=40, color="#fc8d62", alpha=0.85)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"PC{i+1}", fontsize=9)
            else:
                ax.scatter(
                    z_pca[:, j],
                    z_pca[:, i],
                    c=ang_2,
                    cmap="hsv",
                    vmin=0,
                    vmax=360,
                    s=2,
                    alpha=0.55,
                    rasterized=True,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            if i == 9:
                ax.set_xlabel(f"PC{j+1}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"PC{i+1}", fontsize=8)

    fig.suptitle(
        "PCA 10×10 | TEST | 2ec6 only | Color: angleZ (HSV)\n"
        f"Model: 20260603_z_noise10 | N={len(z_2)}",
        fontsize=16,
        y=0.995,
    )
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 360)),
        cax=cax,
    )
    cb.set_label("Angle Z (deg)")
    os.makedirs(OUT_MATRIX, exist_ok=True)
    matrix_path = os.path.join(OUT_MATRIX, "pca_10x10_matrix_2ec6_angleZ.png")
    fig.savefig(matrix_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(OUT_ROOT, "pca_explained_variance.txt"), "w") as f:
        f.write("PC\tratio\tcumsum\n")
        c = 0.0
        for i, r in enumerate(pca.explained_variance_ratio_, 1):
            c += r
            f.write(f"{i}\t{r:.6f}\t{c:.6f}\n")

    n_png = len([x for x in os.listdir(OUT_PAIR) if x.endswith(".png")])
    print("pair plots:", n_png)
    print("OUT_PAIR:", OUT_PAIR)
    print("MATRIX:", matrix_path)
    print("ROOT:", OUT_ROOT)


if __name__ == "__main__":
    main()
