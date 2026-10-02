import os
import json
import logging
import re
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from data_loader import OnTheFlyPDBDataLoader, load_and_preprocess_image
from model import Sampling # Import Sampling to register the custom layer in Keras
import utils # This is the newly copied utils_v2.py

# ログの設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AnalysisMain")


def make_auto_output_dir(base_path, model_name="analysis"):
    """
    Creates an output directory under base_path using the current date (YYYYMMDD) 
    and handles automatic numbering if the directory already exists.
    """
    today = datetime.now().strftime("%Y%m%d")
    dir_base = os.path.join(base_path, today)
    out_dir = f"{dir_base}_z_noise10" # yyy_sensitivity の make_auto_dir スタイルに命名規則を統一
    
    # モデル名をサブフォルダ名に付加
    out_dir = f"{out_dir}_{model_name}"
    
    n = 1
    while os.path.exists(out_dir):
        out_dir = f"{dir_base}_{n:02d}_z_noise10_{model_name}"
        n += 1
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_inference(encoder, dataset_paths, batch_size, pixel_size, output_dir, mode_name):
    """
    Runs inference on a specific split (train or test) and returns the latent vectors (z_mean).
    Memory-safe sequential loading to avoid tf.data deadlock.
    """
    logger.info(f"--- Running inference on {mode_name} dataset ({len(dataset_paths)} images) ---")
    from PIL import Image
    
    from concurrent.futures import ThreadPoolExecutor
    
    def load_single_image(args):
        idx, p = args
        with Image.open(str(p)) as img:
            img_resized = img.resize((pixel_size, pixel_size), Image.Resampling.BILINEAR)
            arr = np.array(img_resized).astype(np.float32) / 255.0
            if len(arr.shape) == 2:
                arr = np.expand_dims(arr, axis=-1)
            return idx, arr

    logger.info("Loading images in parallel using ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(load_single_image, enumerate(dataset_paths)))
        
    results.sort(key=lambda x: x[0])
    imgs = [r[1] for r in results]
    logger.info(f"Successfully loaded {len(imgs)} images.")
            
    x_arr = np.array(imgs)
    logger.info(f"Finished loading images. Running model prediction...")
    
    # 潜在ベクトルの推論 (z_mean の取得)
    out = encoder.predict(x_arr, batch_size=batch_size, verbose=1)
    if isinstance(out, (list, tuple)) and len(out) >= 1:
        z_mean = out[0]
    else:
        z_mean = out

    np.save(os.path.join(output_dir, f"z_mean_{mode_name}.npy"), z_mean)
    logger.info(f"Saved inference z_mean array shape: {z_mean.shape}")
    return z_mean


def main():
    # 1. config.json の読み込み
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        logger.error(f"config.json not found in {os.path.dirname(__file__)}")
        return

    logger.info("Loading config.json...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # 環境によるパスの差異を自動調整（VolumesをUsersに読み替え、サーバー環境に合わせる）
    def adjust_path(p_str):
        if not p_str:
            return p_str
        p_str = str(p_str)
        if p_str.startswith("/Volumes/hikaru/"):
            return p_str.replace("/Volumes/hikaru/", "/Users/hikaru/", 1)
        elif p_str.startswith("/Users/hikarui./"):
            return p_str.replace("/Users/hikarui./", "/Users/hikaru/", 1)
        return p_str

    data_path = adjust_path(config["DATA_PATH"])
    encoder_model_path = adjust_path(config["ENCODER_MODEL_PATH"])
    base_result_dir = adjust_path(config.get("BASE_RESULT_DIR", "/Users/hikaru/Code/2026_Project/07_result/"))
    pixel_size = config.get("PIXEL_SIZE", 128)
    batch_size = config.get("BATCH_SIZE", 256)
    latent_dim = config.get("LATENT_DIM", 10)
    sn_levels = config["SN_LEVELS"]
    train_pdbs = config["TRAIN_PDBIDS"]
    test_pdbs = config["TEST_PDBIDS"]
    target_angle_idx = config.get("TARGET_ANGLE_INDEX", 1)  # 0: xxx, 1: yyy, 2: zzz
    colormap_name = config.get("COLORMAP_NAME", "RdYlBu_r")
    
    # PCA軸の設定を取得 (X=1, Y=2 など)
    pca_axis = (1, 2) # Override to PC1 vs PC2 based on user request
    
    # パース設定 & フィルター設定の取得
    parser_config = config.get("FILENAME_PARSER", None)
    angle_filter = config.get("ANGLE_FILTER", None)

    # エンコーダの親フォルダ名を抽出して、出力フォルダの末尾に付加する（例: y_all_batch=128 など）
    model_name = os.path.basename(os.path.dirname(encoder_model_path))
    if not model_name:
        model_name = "analysis"

    # 日付ベースの自動連番付き出力ディレクトリを作成
    output_dir = make_auto_output_dir(base_result_dir, model_name)

    logger.info(f"Target encoder model: {encoder_model_path}")
    logger.info(f"Target data path: {data_path}")
    logger.info(f"Final output directory: {output_dir}")

    # --- Plot Training History ---
    model_dir = os.path.dirname(encoder_model_path)
    csv_path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(csv_path):
        utils.plot_training_history(csv_path, out_dir=output_dir)
        logger.info(f"Plotted training history from {csv_path}")

    # 2. データの取得（スキャンまたはモデルフォルダからのnpy読み込み）
    use_model_npy = (data_path is None or data_path == "")
    skip_inference = False
    z_mean_train = None
    z_mean_test = None
    
    if use_model_npy:
        model_dir = os.path.dirname(encoder_model_path)
        logger.info(f"DATA_PATH が指定されていないため、モデルフォルダから npy メタデータをロードします: {model_dir}")
        
        # 各種 npy ファイルの読み込み
        kills_train = np.load(os.path.join(model_dir, "kills_train.npy"), allow_pickle=True)
        kills_test = np.load(os.path.join(model_dir, "kills_test.npy"), allow_pickle=True)
        
        # 実行環境と npy 内のパスのズレを動的に修正（VolumesをUsersに読み替える）
        def fix_paths(paths):
            fixed = []
            for p in paths:
                p_str = str(p)
                if p_str.startswith("/Volumes/hikaru/"):
                    fixed.append(p_str.replace("/Volumes/hikaru/", "/Users/hikaru/", 1))
                elif p_str.startswith("/Users/hikarui./"):
                    fixed.append(p_str.replace("/Users/hikarui./", "/Users/hikaru/", 1))
                else:
                    fixed.append(p_str)
            return np.array(fixed)

        kills_train = fix_paths(kills_train)
        kills_test = fix_paths(kills_test)
        
        y_train = np.load(os.path.join(model_dir, "y_train.npy"))
        y_test = np.load(os.path.join(model_dir, "y_test.npy"))
        
        # 3軸の角度情報をスタックして angles 配列を構築 (N, 3)
        x_angle_train = np.load(os.path.join(model_dir, "x_angle_train.npy"))
        y_angle_train = np.load(os.path.join(model_dir, "y_angle_train.npy"))
        z_angle_train = np.load(os.path.join(model_dir, "z_angle_train.npy"))
        train_angles = np.column_stack([x_angle_train, y_angle_train, z_angle_train])
        
        x_angle_test = np.load(os.path.join(model_dir, "x_angle_test.npy"))
        y_angle_test = np.load(os.path.join(model_dir, "y_angle_test.npy"))
        z_angle_test = np.load(os.path.join(model_dir, "z_angle_test.npy"))
        test_angles = np.column_stack([x_angle_test, y_angle_test, z_angle_test])
        
        pdbids_train = np.load(os.path.join(model_dir, "pdbids_train.npy"), allow_pickle=True)
        pdbids_test = np.load(os.path.join(model_dir, "pdbids_test.npy"), allow_pickle=True)
        
        train_info = {
            "paths": kills_train,
            "y": y_train,
            "angles": train_angles,
            "pdbids": pdbids_train
        }
        test_info = {
            "paths": kills_test,
            "y": y_test,
            "angles": test_angles,
            "pdbids": pdbids_test
        }
        logger.info(f"Loaded from model npy. Train: {len(kills_train)} paths, Test: {len(kills_test)} paths.")
        
        z_mean_train_path = os.path.join(model_dir, "z_mean_train.npy")
        z_mean_test_path = os.path.join(model_dir, "z_mean_test.npy")
        if os.path.exists(z_mean_train_path) and os.path.exists(z_mean_test_path):
            logger.info("抽出済みの潜在変数データ (z_mean) が見つかりました。画像の再読み込みと推論をスキップします。")
            z_mean_train = np.load(z_mean_train_path)
            z_mean_test = np.load(z_mean_test_path)
            skip_inference = True
            
    else:
        # データセットのスキャン（パース・アングルフィルタを設定）
        loader = OnTheFlyPDBDataLoader(
            data_path=data_path,
            sn_levels=sn_levels,
            allowed_train_pdbs=train_pdbs,
            allowed_test_pdbs=test_pdbs,
            pixel_size=pixel_size,
            parser_config=parser_config,
            angle_filter=angle_filter
        )
        train_info, test_info = loader.scan_dataset()
        if train_info is None or test_info is None:
            logger.error("Failed to load data paths.")
            return

    # 2.5. リピートによるデータフィルタリング (repeat0~10など)
    allowed_repeats = config.get("ALLOWED_REPEATS", None)
    if allowed_repeats is not None:
        logger.info(f"Filtering dataset by repeats: {allowed_repeats}")
        
        def extract_repeat_number(path_str):
            # _rep07.tif または _repeat07.tif のような表記から数値を抽出
            match = re.search(r'_rep(?:eat)?(\d+)', path_str)
            if match:
                return int(match.group(1))
            return None

        def filter_by_repeats(info):
            paths = info["paths"]
            y = info["y"]
            angles = info["angles"]
            pdbids = info["pdbids"]
            
            keep_indices = []
            for idx, p in enumerate(paths):
                rep_num = extract_repeat_number(str(p))
                if rep_num is None or rep_num in allowed_repeats:
                    keep_indices.append(idx)
                    
            keep_indices = np.array(keep_indices, dtype=int)
            return {
                "paths": np.array(paths)[keep_indices],
                "y": np.array(y)[keep_indices],
                "angles": np.array(angles)[keep_indices],
                "pdbids": np.array(pdbids)[keep_indices]
            }
            
        train_info = filter_by_repeats(train_info)
        test_info = filter_by_repeats(test_info)
        logger.info(f"After repeat filtering - Train: {len(train_info['paths'])}, Test: {len(test_info['paths'])}")

    # メタデータ配列を保存
    np.save(os.path.join(output_dir, "y_train.npy"), train_info["y"])
    np.save(os.path.join(output_dir, "y_test.npy"), test_info["y"])
    np.save(os.path.join(output_dir, "kills_train.npy"), train_info["paths"])
    np.save(os.path.join(output_dir, "kills_test.npy"), test_info["paths"])
    np.save(os.path.join(output_dir, "pdbids_train.npy"), train_info["pdbids"])
    np.save(os.path.join(output_dir, "pdbids_test.npy"), test_info["pdbids"])
    np.save(os.path.join(output_dir, "x_angle_test.npy"), test_info["angles"][:, 0])
    np.save(os.path.join(output_dir, "y_angle_test.npy"), test_info["angles"][:, 1])
    np.save(os.path.join(output_dir, "z_angle_test.npy"), test_info["angles"][:, 2])
    logger.info("Saved metadata files to output directory.")

    # 3. エンコーダモデルとデコーダモデルのロード
    logger.info("Loading encoder model...")
    if not os.path.exists(encoder_model_path):
        logger.error(f"Encoder model path does not exist: {encoder_model_path}")
        return

    decoder = None
    if encoder_model_path.endswith(".weights.h5") or "weights" in os.path.basename(encoder_model_path):
        logger.info("重みファイルが指定されたため、VAEモデル構造を構築してパラメータをロードします。")
        from model import get_encoder, get_decoder, VAE
        input_shape = (pixel_size, pixel_size, 1)
        encoder = get_encoder(input_shape, latent_dim)
        decoder = get_decoder(latent_dim)
        vae = VAE(encoder, decoder)
        dummy_input = tf.zeros((1, pixel_size, pixel_size, 1))
        _ = vae(dummy_input)
        vae.load_weights(encoder_model_path)
        encoder = vae.encoder
        decoder = vae.decoder
        logger.info("VAE の重みからモデルを正常に抽出しました。")
    else:
        logger.info("保存済みモデルからエンコーダを直接ロードします。（デコーダはNoneとなります）")
        encoder = keras.models.load_model(
            encoder_model_path,
            custom_objects={"Sampling": Sampling}
        )

    # 4. 推論の実行
    if not skip_inference:
        z_mean_train = run_inference(
            encoder=encoder, dataset_paths=train_info["paths"],
            batch_size=batch_size, pixel_size=pixel_size,
            output_dir=output_dir, mode_name="train"
        )
        z_mean_test = run_inference(
            encoder=encoder, dataset_paths=test_info["paths"],
            batch_size=batch_size, pixel_size=pixel_size,
            output_dir=output_dir, mode_name="test"
        )

    # 5. 共通の PCA フィッティング (学習時 callbacks の再現)
    logger.info("Establishing common PCA space using Test dataset as reference (reproducing learning callback logic)...")
    from sklearn.decomposition import PCA
    import random
    
    # 学習時と同じシード値 720 を適用して、完全に同じサンプリング/シャッフル順序を再現する
    random.seed(720)
    np.random.seed(720)
    tf.random.set_seed(720)
    
    # 学習時の callbacks 設定と同様、x_test (z_mean_test) から np.random.choice を用いて
    # 最大 20,000 点をサンプリング (現在の z_mean_test の件数は 6480 なので、全データがシャッフルされた順序で取得される)
    n_vis = min(len(z_mean_test), 20000)
    idx = np.random.choice(len(z_mean_test), n_vis, replace=False)
    z_mean_test_vis = z_mean_test[idx]
    
    # ユーザーが指定したPCA軸を含むように、また全体の分散を確認できるように全ての成分(latent_dim)でPCAをfit
    pca_test = PCA(n_components=latent_dim)
    pca_test.fit(z_mean_test_vis)
    
    # テストデータおよびトレーニングデータをPCA空間に投影
    df_test = pd.DataFrame(z_mean_test)
    df_test.columns = [f'z{i}' for i in range(df_test.shape[1])]
    pca_z_mean_test_full = pca_test.transform(df_test)
    
    df_train = pd.DataFrame(z_mean_train)
    df_train.columns = [f'z{i}' for i in range(df_train.shape[1])]
    pca_z_mean_train_full = pca_test.transform(df_train)

    # ユーザー指定のX軸、Y軸成分（1-indexed）を抽出して2次元の描画用配列を生成
    pca_axis = (1, 2)
    idx_x = pca_axis[0] - 1
    idx_y = pca_axis[1] - 1
    pca_z_mean_test = np.column_stack([pca_z_mean_test_full[:, idx_x], pca_z_mean_test_full[:, idx_y]])
    pca_z_mean_train = np.column_stack([pca_z_mean_train_full[:, idx_x], pca_z_mean_train_full[:, idx_y]])
    
    pca_max_x = max(np.abs(pca_z_mean_test[:, 0]).max(), np.abs(pca_z_mean_train[:, 0]).max()) * 1.05
    pca_max_y = max(np.abs(pca_z_mean_test[:, 1]).max(), np.abs(pca_z_mean_train[:, 1]).max()) * 1.05

    xlabel = f"PC{pca_axis[0]}"
    ylabel = f"PC{pca_axis[1]}"

    # 6. 元の utils_v2.py (現在の utils.py) を使用して、元々の可視化グラフを再現
    logger.info("Generating original style visualizations...")

    # idx_rep_test setup (include rep 1~3)
    def is_rep123(p):
        p_str = str(p)
        reps = ["_rep01", "_repeat01", "_rep02", "_repeat02", "_rep03", "_repeat03", 
                "_rep1_", "_repeat1_", "_rep2_", "_repeat2_", "_rep3_", "_repeat3_"]
        return any(rep in p_str for rep in reps)
        
    idx_rep_test = [i for i, p in enumerate(test_info["paths"]) if is_rep123(p)]
    idx_rep_train = [i for i, p in enumerate(train_info["paths"]) if is_rep123(p)]
    
    # --- (A) Testデータ用の各種プロット ---
    logger.info("Plotting Test Dataset plots...")
    # 1. 潜在空間散布図行列 (check_senzai)
    # --- MAX RAW ---
    utils.check_senzai(z_mean_test, test_info["y"], CMAP=colormap_name, AL=0.8, size=2,
                       save_name="latent_scatter_matrix_test_max_raw.png", out_dir=output_dir, pdf_fig_list=None,
                       title="Latent Space Scatter Matrix (Test - Max Raw)")
    # --- MAX PCA ---
    utils.check_senzai(pca_z_mean_test_full, test_info["y"], CMAP=colormap_name, AL=0.8, size=2,
                       save_name="latent_scatter_matrix_test_max_pca.png", out_dir=output_dir, pdf_fig_list=None,
                       title="Latent Space Scatter Matrix (Test - Max PCA)")
    
    # --- REP RAW ---
    if len(idx_rep_test) > 0:
        utils.check_senzai(z_mean_test[idx_rep_test], test_info["y"][idx_rep_test], CMAP=colormap_name, AL=0.8, size=2,
                           save_name="latent_scatter_matrix_test_rep1to3_raw.png", out_dir=output_dir, pdf_fig_list=None,
                           title="Latent Space Scatter Matrix (Test - Rep 1-3 Raw)")
        # --- REP PCA ---
        utils.check_senzai(pca_z_mean_test_full[idx_rep_test], test_info["y"][idx_rep_test], CMAP=colormap_name, AL=0.8, size=2,
                           save_name="latent_scatter_matrix_test_rep1to3_pca.png", out_dir=output_dir, pdf_fig_list=None,
                           title="Latent Space Scatter Matrix (Test - Rep 1-3 PCA)")
    
    # --- PDB RAW & PCA (10D Scatter Matrix) ---
    unique_pdbs_test = sorted(list(set(test_info["pdbids"])))
    pdb_to_idx_test = {pid: i for i, pid in enumerate(unique_pdbs_test)}
    pdb_idx_test = np.array([pdb_to_idx_test[pid] for pid in test_info["pdbids"]])
    
    # Generate explicit color list so utils.check_senzai doesn't normalize it as a continuous variable
    import matplotlib.pyplot as plt
    cmap_pdb_obj = plt.get_cmap("tab10" if len(unique_pdbs_test) <= 10 else "tab20")
    pdb_colors_test = [cmap_pdb_obj(pdb_to_idx_test[pid]) for pid in test_info["pdbids"]]

    utils.check_senzai(z_mean_test, pdb_colors_test, CMAP=None, AL=0.8, size=2,
                       save_name="latent_scatter_matrix_test_max_raw_pdb.png", out_dir=output_dir, pdf_fig_list=None,
                       title="Latent Space Scatter Matrix by PDB (Test - Max Raw)")
    utils.check_senzai(pca_z_mean_test_full, pdb_colors_test, CMAP=None, AL=0.8, size=2,
                       save_name="latent_scatter_matrix_test_max_pca_pdb.png", out_dir=output_dir, pdf_fig_list=None,
                       title="Latent Space Scatter Matrix by PDB (Test - Max PCA)")

    if len(idx_rep_test) > 0:
        pdb_colors_test_rep = [pdb_colors_test[i] for i in idx_rep_test]
        utils.check_senzai(z_mean_test[idx_rep_test], pdb_colors_test_rep, CMAP=None, AL=0.8, size=2,
                           save_name="latent_scatter_matrix_test_rep1to3_raw_pdb.png", out_dir=output_dir, pdf_fig_list=None,
                           title="Latent Space Scatter Matrix by PDB (Test - Rep 1-3 Raw)")
        utils.check_senzai(pca_z_mean_test_full[idx_rep_test], pdb_colors_test_rep, CMAP=None, AL=0.8, size=2,
                           save_name="latent_scatter_matrix_test_rep1to3_pca_pdb.png", out_dir=output_dir, pdf_fig_list=None,
                           title="Latent Space Scatter Matrix by PDB (Test - Rep 1-3 PCA)")

    # --- PCA Pairwise (Sparse) ---
    logger.info("Generating sparse PCA pairwise plots (Test)...")
    utils.plot_pca_pairs_sparse(
        pca_z=pca_z_mean_test_full,
        y=test_info["y"],
        angle_x=test_info["angles"][:, 0],
        angle_z=test_info["angles"][:, 2],
        out_dir=output_dir,
        step_size=30,
        prefix="test"
    )

    # 2. 2D PCA プロット (PCA_graph)
    # --- MAX ---
    utils.PCA_graph(pca_z_mean_test, test_info["y"], CMAP=colormap_name, X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
                    save_name="pca_2d_test_max_state.png", out_dir=output_dir, pdf_fig_list=None,
                    title=f"2D PCA of Latent Space (Test Max: {xlabel} vs {ylabel})",
                    xlabel=xlabel, ylabel=ylabel)
    utils.PCA_graph(pca_z_mean_test, test_info["angles"][:, 0], CMAP="RdYlBu_r", X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
                    save_name="pca_2d_test_max_angle.png", out_dir=output_dir, pdf_fig_list=None,
                    title=f"2D PCA of Latent Space by Angle (Test Max: {xlabel} vs {ylabel})",
                    xlabel=xlabel, ylabel=ylabel)
    # --- REP ---
    if len(idx_rep_test) > 0:
        utils.PCA_graph(pca_z_mean_test[idx_rep_test], test_info["y"][idx_rep_test], CMAP=colormap_name, X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
                        save_name="pca_2d_test_rep1to3_state.png", out_dir=output_dir, pdf_fig_list=None,
                        title=f"2D PCA of Latent Space (Test Rep 1-3: {xlabel} vs {ylabel})",
                        xlabel=xlabel, ylabel=ylabel)
        utils.PCA_graph(pca_z_mean_test[idx_rep_test], test_info["angles"][idx_rep_test, 0], CMAP="RdYlBu_r", X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
                        save_name="pca_2d_test_rep1to3_angle.png", out_dir=output_dir, pdf_fig_list=None,
                        title=f"2D PCA of Latent Space by Angle (Test Rep 1-3: {xlabel} vs {ylabel})",
                        xlabel=xlabel, ylabel=ylabel)

    # --- 1qvi Highlight ---
    def plot_pca_highlight_1qvi(pca_z, y, pdbids, save_name, title, x_limit=None, y_limit=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap("RdYlBu")
        colors = []
        for state_val, pid in zip(y, pdbids):
            if pid == "1qvi":
                colors.append("green")
            elif state_val == 0:
                colors.append(cmap(0.0))
            elif state_val == 1:
                colors.append(cmap(1.0))
            else:
                colors.append("black")
        # Plot 1qvi on top by sorting or just plotting everything
        ax.scatter(pca_z[:, 0], pca_z[:, 1], c=colors, s=5, alpha=0.8, edgecolors='none')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if x_limit: ax.set_xlim(-x_limit, x_limit)
        if y_limit: ax.set_ylim(-y_limit, y_limit)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

    plot_pca_highlight_1qvi(pca_z_mean_test, test_info["y"], test_info["pdbids"], 
                            "pca_2d_test_max_1qvi_green.png", f"2D PCA (Test Max 1qvi Highlight: {xlabel} vs {ylabel})", x_limit=pca_max_x, y_limit=pca_max_y)
    if len(idx_rep_test) > 0:
        plot_pca_highlight_1qvi(pca_z_mean_test[idx_rep_test], test_info["y"][idx_rep_test], np.array(test_info["pdbids"])[idx_rep_test], 
                                "pca_2d_test_rep1to3_1qvi_green.png", f"2D PCA (Test Rep 1-3 1qvi Highlight: {xlabel} vs {ylabel})", x_limit=pca_max_x, y_limit=pca_max_y)

    # --- PDB Color Highlight ---
    def plot_pca_by_pdb_custom(pca_z, pdbids, save_name, title, x_limit=None, y_limit=None):
        fig, ax = plt.subplots(figsize=(8, 7))
        unique_pdbs = sorted(list(set(pdbids)))
        cmap = plt.get_cmap("tab10" if len(unique_pdbs) <= 10 else "tab20")
        for i, pid in enumerate(unique_pdbs):
            idx = [j for j, p in enumerate(pdbids) if p == pid]
            ax.scatter(pca_z[idx, 0], pca_z[idx, 1], c=[cmap(i)], s=10, alpha=0.8, label=pid, edgecolors='none')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if x_limit: ax.set_xlim(-x_limit, x_limit)
        if y_limit: ax.set_ylim(-y_limit, y_limit)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

    plot_pca_by_pdb_custom(pca_z_mean_test, test_info["pdbids"], "pca_2d_test_max_pdb.png", f"2D PCA by PDB (Test Max: {xlabel} vs {ylabel})", x_limit=pca_max_x, y_limit=pca_max_y)
    if len(idx_rep_test) > 0:
        plot_pca_by_pdb_custom(pca_z_mean_test[idx_rep_test], np.array(test_info["pdbids"])[idx_rep_test], "pca_2d_test_rep1to3_pdb.png", f"2D PCA by PDB (Test Rep 1-3: {xlabel} vs {ylabel})", x_limit=pca_max_x, y_limit=pca_max_y)
    
    # 3. 2D Histogram of PC1 (hitstgram_2D)
    utils.hitstgram_2D(
        pca_z_mean_test, test_info["y"], 0, bins=30, rMin=-5, rMax=5,
        save_name="histogram_test.png", out_dir=output_dir, pdf_fig_list=None,
        title=f"2-State Histogram of {xlabel} (Test)"
    )
    
    # 4. サムネイル付き散布図 (make_decoder)
    # --- MAX version ---
    logger.info("Generating Max latent imscatter (Test)...")
    if len(idx_rep_test) > 0:
        paths_sub_test = np.array(test_info["paths"])[idx_rep_test]
        pca_sub_test = pca_z_mean_test[idx_rep_test]
    else:
        paths_sub_test = test_info["paths"]
        pca_sub_test = pca_z_mean_test

    logger.info("Generating Repeat 1-3 latent imscatter (Test)...")
    utils.make_decoder(
        pca_sub_test[:, 0], pca_sub_test[:, 1], paths_sub_test, 5, 5, 0.2,
        save_name="latent_imscatter_test_rep1to3.png", out_dir=output_dir, pdf_fig_list=None
    )
    if decoder is not None:
        recon_test_max = decoder.predict(z_mean_test, batch_size=256, verbose=0)
        utils.make_decoder(
            pca_z_mean_test[:, 0], pca_z_mean_test[:, 1], recon_test_max, 5, 5, 0.2,
            save_name="latent_imscatter_test_max_reconstructed.png", out_dir=output_dir, pdf_fig_list=None
        )
    
    # 5. PDBIDごとのPCAプロット (plot_pca_by_pdbid_state)
    for state in [0, 1, 2]:
        utils.plot_pca_by_pdbid_state(
            z_mean_test, test_info["y"], test_info["pdbids"], state=state,
            out_dir=output_dir, pdf_fig_list=None, basename=f"pca_pdb_all_state{state}_test",
            pca_z=pca_z_mean_test, xlabel=xlabel, ylabel=ylabel
        )
        
    # 6. 軌跡 (plot_pca_trajectories_by_pdb)
    utils.plot_pca_trajectories_by_pdb(
        z_mean_test, test_info["angles"][:, target_angle_idx], test_info["pdbids"], test_info["y"],
        out_dir=output_dir, pdf_fig_list=None, basename="06_pca_trajectories_test",
        pca_z=pca_z_mean_test
    )
    
    # 7. HeatMap (HeatMap)
    utils.HeatMap(
        pca_z_mean_test, test_info["y"], 5, 0.2, 0, 1,
        out_dir=output_dir, prefix="HeatMap_test"
    )

    # --- (B) Trainデータ用の各種プロット (同一PCA軸を使用) ---
    logger.info("Plotting Train Dataset plots...")
    # 1. 潜在空間散布図行列 (check_senzai)
    utils.check_senzai(
        z_mean_train, train_info["y"],
        CMAP=colormap_name, AL=0.8, size=2,
        save_name="latent_scatter_matrix_train.png", out_dir=output_dir, pdf_fig_list=None,
        title="Sampled Latent Space Scatter Matrix (Train)"
    )

    if len(idx_rep_train) > 0:
        utils.check_senzai(z_mean_train[idx_rep_train], train_info["y"][idx_rep_train], CMAP=colormap_name, AL=0.8, size=2,
                           save_name="latent_scatter_matrix_train_rep1to3_raw.png", out_dir=output_dir, pdf_fig_list=None,
                           title="Latent Space Scatter Matrix (Train - Rep 1-3 Raw)")
        # --- REP PCA ---
        utils.check_senzai(pca_z_mean_train_full[idx_rep_train], train_info["y"][idx_rep_train], CMAP=colormap_name, AL=0.8, size=2,
                           save_name="latent_scatter_matrix_train_rep1to3_pca.png", out_dir=output_dir, pdf_fig_list=None,
                           title="Latent Space Scatter Matrix (Train - Rep 1-3 PCA)")
    
    # --- PCA Pairwise (Sparse) ---
    logger.info("Generating sparse PCA pairwise plots (Train)...")
    utils.plot_pca_pairs_sparse(
        pca_z=pca_z_mean_train_full,
        y=train_info["y"],
        angle_x=train_info["angles"][:, 0],
        angle_z=train_info["angles"][:, 2],
        out_dir=output_dir,
        step_size=30,
        prefix="train"
    )

    # 2. 2D PCA プロット (PCA_graph)
    utils.PCA_graph(
        pca_z_mean_train, train_info["y"], CMAP=colormap_name, X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
        save_name="pca_2d_train.png", out_dir=output_dir, pdf_fig_list=None,
        title=f"2D PCA of Latent Space (Train: {xlabel} vs {ylabel})",
        xlabel=xlabel, ylabel=ylabel
    )
    utils.PCA_graph(
        pca_z_mean_train, train_info["angles"][:, 0], CMAP="RdYlBu_r", X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
        save_name="pca_2d_train_max_angle.png", out_dir=output_dir, pdf_fig_list=None,
        title=f"2D PCA of Latent Space by Angle (Train: {xlabel} vs {ylabel})",
        xlabel=xlabel, ylabel=ylabel
    )
    if len(idx_rep_train) > 0:
        utils.PCA_graph(pca_z_mean_train[idx_rep_train], train_info["y"][idx_rep_train], CMAP=colormap_name, X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
                        save_name="pca_2d_train_rep1to3_state.png", out_dir=output_dir, pdf_fig_list=None,
                        title=f"2D PCA of Latent Space (Train Rep 1-3: {xlabel} vs {ylabel})",
                        xlabel=xlabel, ylabel=ylabel)
        utils.PCA_graph(pca_z_mean_train[idx_rep_train], train_info["angles"][idx_rep_train, 0], CMAP="RdYlBu_r", X=pca_max_x, Y=pca_max_y, AL=0.8, size=5,
                        save_name="pca_2d_train_rep1to3_angle.png", out_dir=output_dir, pdf_fig_list=None,
                        title=f"2D PCA of Latent Space by Angle (Train Rep 1-3: {xlabel} vs {ylabel})",
                        xlabel=xlabel, ylabel=ylabel)

    plot_pca_by_pdb_custom(pca_z_mean_train, train_info["pdbids"], "pca_2d_train_max_pdb.png", f"2D PCA by PDB (Train Max: {xlabel} vs {ylabel})", x_limit=pca_max_x, y_limit=pca_max_y)
    if len(idx_rep_train) > 0:
        plot_pca_by_pdb_custom(pca_z_mean_train[idx_rep_train], np.array(train_info["pdbids"])[idx_rep_train], "pca_2d_train_rep1to3_pdb.png", f"2D PCA by PDB (Train Rep 1-3: {xlabel} vs {ylabel})", x_limit=pca_max_x, y_limit=pca_max_y)
    
    # 3. 2D Histogram of PC1 (hitstgram_2D)
    utils.hitstgram_2D(
        pca_z_mean_train, train_info["y"], 0, bins=30, rMin=-5, rMax=5,
        save_name="histogram_train.png", out_dir=output_dir, pdf_fig_list=None,
        title="2-State Histogram of PC1 (Train)"
    )
    
    # 4. サムネイル付き散布図 (make_decoder)
    if len(idx_rep_train) > 0:
        paths_sub_train = np.array(train_info["paths"])[idx_rep_train]
        pca_sub_train = pca_z_mean_train[idx_rep_train]
    else:
        paths_sub_train = train_info["paths"]
        pca_sub_train = pca_z_mean_train

    logger.info("Generating Repeat 1-3 latent imscatter (Train)...")
    utils.make_decoder(
        pca_sub_train[:, 0], pca_sub_train[:, 1], paths_sub_train, 5, 5, 0.2,
        save_name="latent_imscatter_train_rep1to3.png", out_dir=output_dir, pdf_fig_list=None
    )
    if decoder is not None:
        recon_train_max = decoder.predict(z_mean_train, batch_size=256, verbose=0)
        utils.make_decoder(
            pca_z_mean_train[:, 0], pca_z_mean_train[:, 1], recon_train_max, 5, 5, 0.2,
            save_name="latent_imscatter_train_max_reconstructed.png", out_dir=output_dir, pdf_fig_list=None
        )
    # --- Repeat 1 version ---
    logger.info("Generating Repeat 1 latent imscatter (Train)...")
    idx_rep1_train = [i for i, p in enumerate(train_info["paths"]) if "_rep01" in str(p) or "_repeat01" in str(p)]
    if len(idx_rep1_train) > 0:
        pca_sub_train = pca_z_mean_train[idx_rep1_train]
        kills_sub_train = np.array(train_info["paths"])[idx_rep1_train]
        utils.make_decoder(
            pca_sub_train[:, 0], pca_sub_train[:, 1], kills_sub_train, 5, 5, 0.2,
            save_name="latent_imscatter_train_rep1_original.png", out_dir=output_dir, pdf_fig_list=None
        )
        if decoder is not None:
            z_sub_train = z_mean_train[idx_rep1_train]
            recon_train_rep1 = decoder.predict(z_sub_train, batch_size=256, verbose=0)
            utils.make_decoder(
                pca_sub_train[:, 0], pca_sub_train[:, 1], recon_train_rep1, 5, 5, 0.2,
                save_name="latent_imscatter_train_rep1_reconstructed.png", out_dir=output_dir, pdf_fig_list=None
            )
    
    # 5. PDBIDごとのPCAプロット (plot_pca_by_pdbid_state)
    for state in [0, 1, 2]:
        utils.plot_pca_by_pdbid_state(
            z_mean_train, train_info["y"], train_info["pdbids"], state=state,
            out_dir=output_dir, pdf_fig_list=None, basename=f"pca_pdb_all_state{state}_train",
            pca_z=pca_z_mean_train
        )
        
    # 6. 軌跡 (plot_pca_trajectories_by_pdb)
    utils.plot_pca_trajectories_by_pdb(
        z_mean_train, train_info["angles"][:, target_angle_idx], train_info["pdbids"], train_info["y"],
        out_dir=output_dir, pdf_fig_list=None, basename="06_pca_trajectories_train",
        pca_z=pca_z_mean_train
    )
    
    # 7. HeatMap (HeatMap)
    utils.HeatMap(
        pca_z_mean_train, train_info["y"], 5, 0.2, 0, 1,
        out_dir=output_dir, prefix="HeatMap_train"
    )

    logger.info(f"Successfully finished analysis. Results are saved in {output_dir}")


if __name__ == "__main__":
    main()
