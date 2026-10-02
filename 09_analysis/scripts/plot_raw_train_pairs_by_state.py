#!/usr/bin/env python3
"""TRAIN raw z pairwise scatters colored by State. 10x10 minus diagonal = 90 plots."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260603_z_noise10"
OUT = (
    "/Users/hikaru/Code/2026_Project/07_result/"
    "20260607_z_noise10_20260603_z_noise10/angle_error_audit/"
    "raw_individual_train_by_state"
)
C0 = "#1f77b4"
C1 = "#d62728"


def main():
    os.makedirs(OUT, exist_ok=True)
    Z = np.load(os.path.join(MODEL_DIR, "z_mean_train.npy"))
    y = np.load(os.path.join(MODEL_DIR, "y_train.npy")).astype(int)
    n_dim = Z.shape[1]
    Z0, Z1 = Z[y == 0], Z[y == 1]

    n = 0
    for i in range(n_dim):  # y-axis
        for j in range(n_dim):  # x-axis
            if i == j:
                continue
            x_lab, y_lab = j + 1, i + 1
            save_path = os.path.join(OUT, f"z{x_lab}_vs_z{y_lab}.png")
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            ax.scatter(
                Z1[:, j], Z1[:, i], s=6, c=C1, alpha=0.35, linewidths=0,
                rasterized=True, label="State1",
            )
            ax.scatter(
                Z0[:, j], Z0[:, i], s=6, c=C0, alpha=0.35, linewidths=0,
                rasterized=True, label="State0",
            )
            ax.set_xlabel(f"z{x_lab}")
            ax.set_ylabel(f"z{y_lab}")
            ax.set_title(f"TRAIN raw  |  z{x_lab} vs z{y_lab}  |  colored by State")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="best", framealpha=0.9)
            fig.tight_layout()
            fig.savefig(save_path, dpi=140)
            plt.close(fig)
            n += 1
            if n % 15 == 0:
                print(f"{n}/90")
    print(f"saved {n} pngs -> {OUT}")


if __name__ == "__main__":
    main()
