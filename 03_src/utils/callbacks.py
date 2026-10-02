import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class SaveEveryNEpochs(tf.keras.callbacks.Callback):
    """指定エポック（20エポック）毎にモデル重み(models/)と再構築画像・グラフ(graphs/)を分けて保存"""
    def __init__(self, x_samples, experiment_name="vae_pdb_multi_v2", every=20, base_dir="06_results"):
        super().__init__()
        self.every = int(every)
        self.experiment_name = experiment_name
        self.x_samples = x_samples
        
        self.models_dir = os.path.join(base_dir, "models", experiment_name)
        self.graphs_dir = os.path.join(base_dir, "graphs", experiment_name)
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % self.every != 0:
            return

        # 1. モデルの重み保存 -> 06_results/models/
        weight_path = os.path.join(self.models_dir, f"vae_epoch_{ep:04d}.h5")
        self.model.save_weights(weight_path)

        # 2. 再構築画像・比較グラフの保存 -> 06_results/graphs/
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
        
        img_path = os.path.join(self.graphs_dir, f"recon_epoch_{ep:04d}.png")
        plt.savefig(img_path)
        plt.close(fig)
        print(f"\n[Callback Epoch {ep:04d}] Saved model weights -> {weight_path}")
        print(f"[Callback Epoch {ep:04d}] Saved reconstruction graph -> {img_path}")
