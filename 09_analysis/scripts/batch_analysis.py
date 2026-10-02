import os
import sys
import glob
import json
import gc
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BatchAnalysis")

BASE_SAVE_DIR = "/Users/hikaru/Code/2026_Project/04_Savemodel/"
BASE_RESULT_DIR = "/Users/hikaru/Code/2026_Project/07_result/"
ANALYSIS_DIR = "/Users/hikaru/Code/2026_Project/06_Analysis/"

# 解析対象モデルディレクトリの一覧 (全10モデル)
TARGET_MODELS = [
    # アニーリング 7層モデル群 (8/28)
    "20260828_anneal_b001_psi_seed420",
    "20260828_anneal_b100_psi_seed420",
    "20260828_anneal_b001_multi_seed420",
    "20260828_anneal_b100_multi_seed420",
    # 3層モデル群 (8/27)
    "20260827_02_3layer_psi_seed420",
    "20260827_03_3layer_psi_seed720",
    "20260827_02_3layer_rot90_seed420",
    "20260827_03_3layer_rot90_seed720",
    # 標準 7層モデル群 (8/25)
    "20260825_psi_noise_seed420_side",
    "20260825_01_psi_noise_seed421_side"
]

def run_analysis_for_model(model_folder):
    save_path = os.path.join(BASE_SAVE_DIR, model_folder)
    out_path = os.path.join(BASE_RESULT_DIR, model_folder)
    
    if not os.path.exists(save_path):
        logger.warning(f"Savemodel directory not found: {save_path}. Skipping.")
        return

    os.makedirs(out_path, exist_ok=True)
    logger.info("=" * 60)
    logger.info(f"Starting batch analysis for: {model_folder}")
    logger.info(f"Target Savemodel: {save_path}")
    logger.info(f"Target Result Dir (07_result): {out_path}")
    logger.info("=" * 60)

    # config.json の引数・パスを書き換え (/Users/hikaru/Code/2026_Project/07_result/ 下に保存)
    cfg_file = os.path.join(ANALYSIS_DIR, "config.json")
    cfg_data = {}
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg_data = json.load(f)
        except Exception:
            cfg_data = {}

    cfg_data["save_model_dir"] = save_path
    cfg_data["output_dir"] = out_path
    cfg_data["enable_svm"] = False  # SVM解析は除外

    with open(cfg_file, "w") as f:
        json.dump(cfg_data, f, indent=4)

    # 独立したサブプロセスで main.py を呼び出し (メモリ領域の完全解放)
    cmd = [sys.executable, os.path.join(ANALYSIS_DIR, "main.py")]
    try:
        res = subprocess.run(cmd, cwd=ANALYSIS_DIR, check=True)
        logger.info(f"Successfully finished batch analysis for {model_folder}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred during analysis for {model_folder}: {e}")

    gc.collect()

def main():
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    logger.info(f"Starting sequential batch analysis runner for {len(TARGET_MODELS)} models")
    logger.info(f"All output files will be saved in: {BASE_RESULT_DIR}")
    logger.info("SVM classification: EXCLUDED (per user instruction)")

    for i, m in enumerate(TARGET_MODELS, 1):
        logger.info(f"\n---> Processing model [{i}/{len(TARGET_MODELS)}]: {m}")
        run_analysis_for_model(m)

    logger.info("\n🎉 All 10 models batch analysis completed successfully!")

if __name__ == "__main__":
    main()
