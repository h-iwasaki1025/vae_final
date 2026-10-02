import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class SaveEveryNEpochs(tf.keras.callbacks.Callback):
    """
    20エポックごとに、YYYYMMDD_experiment_name フォルダを生成し、
    - 06_results/models/YYYYMMDD_experiment_name/ に重み (.h5)
    - 06_results/images/YYYYMMDD_experiment_name/ に再構成比較画像 (.png)
    - 06_results/graphs/YYYYMMDD_experiment_name/ に学習グラフ (.png)
    を分けて自動保存。
    """
    def __init__(self, x_samples, experiment_name="vae_pdb_multi_v2", every=20, base_dir="06_results"):
        super().__init__()
        self.every = int(every)
        self.experiment_name = experiment_name
        self.x_samples = x_samples
        
        # 今日の日付 (YYYYMMDD) プレフィックスを生成
        today_str = datetime.now().strftime("%Y%m%d")
        folder_name = f"{today_str}_{experiment_name}"
        
        self.models_dir = os.path.join(base_dir, "models", folder_name)
        self.images_dir = os.path.join(base_dir, "images", folder_name)
        self.graphs_dir = os.path.join(base_dir, "graphs", folder_name)
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        print(f"[Callback Init] Output directories prepared under {folder_name}:")
        print(f"  - Models: {self.models_dir}")
        print(f"  - Images: {self.images_dir}")
        print(f"  - Graphs: {self.graphs_dir}")

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % self.every != 0:
            return

        # 1. モデルの重み保存 -> 06_results/models/YYYYMMDD_experiment_name/
        weight_path = os.path.join(self.models_dir, f"vae_epoch_{ep:04d}.h5")
        self.model.save_weights(weight_path)

        # 2. 再構成比較画像の保存 -> 06_results/images/YYYYMMDD_experiment_name/
        decoded = self.model.predict(self.x_samples, verbose=0)
        n = min(len(self.x_samples), 8)
        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        for i in range(n):
            axes[0, i].imshow(self.x_samples[i].squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(decoded[i].squeeze(), cmap="gray")
            axes[1, i].axis("off")
        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Reconstructed")
        plt.tight_layout()
        
        img_path = os.path.join(self.images_dir, f"recon_epoch_{ep:04d}.png")
        plt.savefig(img_path)
        plt.close(fig)

        print(f"\n[Callback Epoch {ep:04d}] Saved weights -> {weight_path}")
        print(f"[Callback Epoch {ep:04d}] Saved recon image -> {img_path}")
