import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class SaveEveryNEpochs(tf.keras.callbacks.Callback):
    """指定エポック（デフォルト20）毎にモデル重みと評価画像を自動保存"""
    def __init__(self, x_samples, every=20, out_dir="06_results/vae_pdb_multi_v2"):
        super().__init__()
        self.every = int(every)
        self.out_dir = out_dir
        self.x_samples = x_samples
        self.weights_dir = os.path.join(out_dir, "weights")
        self.images_dir = os.path.join(out_dir, "reconstruction_images")
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % self.every != 0:
            return

        # 1. 重みの保存
        weight_path = os.path.join(self.weights_dir, f"vae_epoch_{ep:04d}.h5")
        self.model.save_weights(weight_path)

        # 2. 再構築画像の保存
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
        print(f"\n[Callback] Saved weights and reconstruction image at epoch {ep} -> {self.out_dir}")
