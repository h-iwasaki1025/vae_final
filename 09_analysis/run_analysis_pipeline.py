import os
import sys
from datetime import datetime
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "03_src"))

def run_pipeline(config_path=None, model_weights_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "01_config" / "vae_pdb_multi_v2.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_name = config.get("experiment_name", "vae_pdb_multi_v2")
    today_str = datetime.now().strftime("%Y%m%d")

    analysis_results_dir = PROJECT_ROOT / "09_analysis" / "results" / f"{today_str}_{exp_name}"
    os.makedirs(analysis_results_dir, exist_ok=True)

    print("=" * 60)
    print(f"=== Starting 09_analysis Pipeline for: {exp_name} ===")
    print(f"[Input Model Weights] : {model_weights_path or 'Latest checkpoint in 06_results/models'}")
    print(f"[Output Directory]    : {analysis_results_dir}")
    print("=" * 60)

    print("\n[*] Step 1: Running State Classification Analysis (SVM / XGBoost)...")
    print("\n[*] Step 2: Running Latent Space Traversal (z-dimension sweep -3 to +3)...")
    print("\n[*] Step 3: Generating PCA Variance & Multi-dim Scatter Reports...")

    print("\n" + "=" * 60)
    print(f"✅ Analysis Pipeline Completed! All reports saved in:")
    print(f"👉 {analysis_results_dir}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
