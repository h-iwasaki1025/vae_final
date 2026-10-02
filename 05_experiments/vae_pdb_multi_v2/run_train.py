import os
import yaml
import sys
from pathlib import Path

# パス追加 (03_src をインポート可能にする)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "03_src"))

from models.vae_7layer import get_encoder, get_decoder, VAE
from utils.callbacks import SaveEveryNEpochs

def main():
    # 1. Config 読み込み
    config_path = PROJECT_ROOT / "01_config" / "vae_pdb_multi_v2.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"=== Starting Experiment: {config['experiment_name']} ===")
    print(f"Dataset path: {config['dataset']['data_dir']}")
    print(f"Save interval: Every {config['training']['save_every_epochs']} epochs")

    # 2. モデル構築
    latent_dim = config['model']['latent_dim']
    pixel_size = config['dataset']['pixel_size']
    beta = config['model']['beta']

    encoder = get_encoder(input_shape=(pixel_size, pixel_size, 1), latent_dim=latent_dim)
    decoder = get_decoder(latent_dim=latent_dim)
    vae = VAE(encoder, decoder, beta=beta)
    
    import tensorflow as tf
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']))

    print("Model architecture ready!")
    print(f"Weights and images will be saved to {config['output']['results_dir']} every {config['training']['save_every_epochs']} epochs.")

if __name__ == "__main__":
    main()
