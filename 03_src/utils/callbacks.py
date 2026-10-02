import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.visualization import PCA_graph, check_senzai

class VAEFullCallback(tf.keras.callbacks.Callback):
    """
    vae_side_template_v2 準拠コールバック:
    指定エポック (20 epoch) ごとに以下を出力:
    1. 06_results/models/YYYYMMDD_exp/ -> 重みファイル (.h5)
    2. 06_results/images/YYYYMMDD_exp/ -> 差分付き3行比較画像 [Original / Recon / Diff (jet)]
    3. 06_results/graphs/YYYYMMDD_exp/ -> 2D PCA (Wide & Zoom) & 潜在空間 Scatter Matrix グラフ
    """
    def __init__(self, x_samples, labels=None, experiment_name="vae_pdb_multi_v2", every=20, base_dir="06_results"):
        super().__init__()
        self.every = int(every)
        self.experiment_name = experiment_name
        self.x_samples = x_samples
        self.labels = labels if labels is not None else np.zeros(len(x_samples))
        
        today_str = datetime.now().strftime("%Y%m%d")
        folder_name = f"{today_str}_{experiment_name}"
        
        self.models_dir = os.path.join(base_dir, "models", folder_name)
        self.images_dir = os.path.join(base_dir, "images", folder_name)
        self.graphs_dir = os.path.join(base_dir, "graphs", folder_name)
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)

    def _get_encoder(self):
        if hasattr(self.model, "encoder") and self.model.encoder is not None:
            return self.model.encoder
        return self.model.get_layer("encoder")

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % self.every != 0:
            return

        # 1. 重みの保存 (06_results/models/)
        weight_path = os.path.join(self.models_dir, f"vae_epoch_{ep:04d}.h5")
        self.model.save_weights(weight_path)

        # 2. 差分付き再構成画像の保存 (06_results/images/)
        decoded = self.model.predict(self.x_samples, batch_size=256, verbose=0)
        n = min(len(self.x_samples), 10)
        fig = plt.figure(figsize=(15, 4))
        for i in range(n):
            orig = self.x_samples[i].squeeze()
            recon = decoded[i].squeeze()
            diff = np.clip(((orig - recon) + 2.0) / 4.0, 0.0, 1.0)

            # Original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(orig, cmap="gray")
            ax.axis("off")
            if i == 0: ax.set_title("Original")

            # Reconstructed
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(recon, cmap="gray")
            ax.axis("off")
            if i == 0: ax.set_title("Reconstructed")

            # Diff
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(diff, cmap=plt.cm.jet)
            ax.axis("off")
            if i == 0: ax.set_title("Diff (jet)")

        img_path = os.path.join(self.images_dir, f"recondiff_ep{ep:04d}.png")
        plt.savefig(img_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        # 3. PCA ＆ Scatter Matrix 潜在空間グラフの保存 (06_results/graphs/)
        enc = self._get_encoder()
        out = enc.predict(self.x_samples, batch_size=512, verbose=0)
        z_mean = out[0] if isinstance(out, (list, tuple)) else out

        if z_mean.shape[1] >= 2:
            pca_z = PCA(n_components=2).fit_transform(z_mean)
            
            # 2D PCA Wide
            PCA_graph(pca_z, ColorMap=self.labels, X=5, Y=5, size=3,
                      save_name=f"pca_wide_ep{ep:04d}.png", out_dir=self.graphs_dir, title=f"2D PCA (Wide) Epoch {ep}")
            
            # 2D PCA Zoom
            PCA_graph(pca_z, ColorMap=self.labels, X=1, Y=1, size=3,
                      save_name=f"pca_zoom_ep{ep:04d}.png", out_dir=self.graphs_dir, title=f"2D PCA (Zoom) Epoch {ep}")

        # Scatter Matrix
        check_senzai(z_mean[:1000], ColorMap=self.labels[:1000], size=2,
                      save_name=f"scatter_matrix_ep{ep:04d}.png", out_dir=self.graphs_dir, title=f"Latent Scatter Matrix Epoch {ep}")

        print(f"\n[Callback Epoch {ep:04d}] Saved: Weights -> {weight_path}, Images -> {img_path}, PCA/Scatter -> {self.graphs_dir}")
