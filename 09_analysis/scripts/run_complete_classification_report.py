import os
import re
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

BASE_SAVE_DIR = "/Users/hikaru/Code/2026_Project/04_Savemodel/"
BASE_DATA_PSI = "/Users/hikaru/Code/2026_Project/05_Data/pdb_PSI_noise/"
BASE_DATA_MULTI = "/Users/hikaru/Code/2026_Project/05_Data/PDBMulti_new/"
BASE_DATA_ROT90 = "/Users/hikaru/Code/2026_Project/05_Data/PDBSidefix_rot90/"
REPORT_OUT_DIR = "/Users/hikaru/Code/2026_Project/07_result/classifier_summary_report/"

# (表示用ラベル, Savemodelフォルダ名, HandsOnコードフォルダ名, データセットパス)
TARGET_MODELS = [
    ("Anneal-0.01-PSI", "20260828_anneal_b001_psi_seed420", "vae_7layer_anneal_b001_psi", BASE_DATA_PSI),
    ("Anneal-1.00-PSI", "20260828_anneal_b100_psi_seed420", "vae_7layer_anneal_b100_psi", BASE_DATA_PSI),
    ("Anneal-0.01-Multi", "20260828_anneal_b001_multi_seed420", "vae_7layer_anneal_b001_multi", BASE_DATA_MULTI),
    ("Anneal-1.00-Multi", "20260828_anneal_b100_multi_seed420", "vae_7layer_anneal_b100_multi", BASE_DATA_MULTI),
    ("3Layer-PSI-s420", "20260827_02_3layer_psi_seed420", "vae_3layer_psi_noise", BASE_DATA_PSI),
    ("3Layer-PSI-s720", "20260827_03_3layer_psi_seed720", "vae_3layer_psi_noise", BASE_DATA_PSI),
    ("3Layer-rot90-s420", "20260827_02_3layer_rot90_seed420", "vae_3layer_rot90", BASE_DATA_ROT90),
    ("3Layer-rot90-s720", "20260827_03_3layer_rot90_seed720", "vae_3layer_rot90", BASE_DATA_ROT90),
]

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def extract_angle(kill_path):
    p_str = str(kill_path)
    fname = os.path.basename(p_str)
    parts = fname.split(".")[0].split("_")
    nums = [p for p in parts if p.isdigit()]
    if len(nums) >= 3:
        return int(nums[2])
    elif len(nums) >= 1:
        return int(nums[-1])
    return 0

def load_data_loader_module(handson_folder):
    folder_path = os.path.join("/Users/hikaru/Code/2026_Project/02_HandsOn/", handson_folder)
    dl_path = os.path.join(folder_path, "data_loader.py")
    spec = importlib.util.spec_from_file_location("custom_data_loader", dl_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PDBDataLoader

def evaluate_model(model_label, model_folder, handson_folder, data_path):
    save_path = os.path.join(BASE_SAVE_DIR, model_folder)
    encoder_path = os.path.join(save_path, "encoder.h5")

    if not os.path.exists(encoder_path):
        print(f"[{model_label}] encoder.h5 not found in {save_path}. Skipping.")
        return None

    y_train_path = os.path.join(save_path, "y_train.npy")
    y_test_path = os.path.join(save_path, "y_test.npy")
    r_test_path = os.path.join(save_path, "r_test.npy")
    kills_test_path = os.path.join(save_path, "kills_test.npy")

    if not (os.path.exists(y_train_path) and os.path.exists(y_test_path)):
        print(f"[{model_label}] y_train / y_test missing. Skipping.")
        return None

    y_train = np.load(y_train_path).astype(int)
    y_test = np.load(y_test_path).astype(int)

    # 角度データの取得
    if os.path.exists(r_test_path):
        angles_test = np.load(r_test_path)
    elif os.path.exists(kills_test_path):
        kills_test = np.load(kills_test_path, allow_pickle=True)
        angles_test = np.array([extract_angle(k) for k in kills_test])
    else:
        angles_test = np.zeros(len(y_test))

    # Encoderのロード
    try:
        custom_objects = {"Sampling": Sampling}
        encoder = tf.keras.models.load_model(encoder_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"[{model_label}] Failed to load encoder: {e}")
        return None

    # 動的モジュールロードでデータロード
    try:
        PDBDataLoader = load_data_loader_module(handson_folder)
        sn_levels = ["0.6", "0.8", "1.0", "1.2", "1.4", "1.6"]
        loader = PDBDataLoader(data_path, sn_levels, pixel_size=128)
        (train_info, test_info) = loader.load_data()
        x_train, x_test = train_info[0], test_info[0]
    except Exception as e:
        print(f"[{model_label}] Data loading error: {e}")
        return None

    # Encoderから z_mean を推論
    z_train, _, _ = encoder.predict(x_train, batch_size=256, verbose=0)
    z_test, _, _ = encoder.predict(x_test, batch_size=256, verbose=0)

    # RandomForest 分類器で State (0 vs 1) を分類
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(z_train, y_train)
    y_pred = clf.predict(z_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 誤判定サンプルの抽出
    misclassified_mask = (y_test != y_pred)
    misclassified_angles = angles_test[misclassified_mask]
    total_errors = int(np.sum(misclassified_mask))

    # グラフ生成
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 混同行列プロット
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["State 0", "State 1"], yticklabels=["State 0", "State 1"])
    axes[0].set_title(f"Confusion Matrix ({model_label})\nAccuracy: {acc*100:.2f}%")
    axes[0].set_xlabel("Predicted State")
    axes[0].set_ylabel("True State")

    # 2. 誤判定が発生した角度のヒストグラム
    if total_errors > 0:
        axes[1].hist(misclassified_angles, bins=20, color="crimson", edgecolor="black", alpha=0.7)
        axes[1].set_title(f"Misclassified Angles Distribution\n(Total Errors: {total_errors}/{len(y_test)})")
        axes[1].set_xlabel("Angle (degree/index)")
        axes[1].set_ylabel("Error Count")
    else:
        axes[1].text(0.5, 0.5, "100% Accuracy!\nZero Misclassifications", 
                     ha="center", va="center", fontsize=14, color="green")
        axes[1].set_title("Misclassified Angles Distribution")

    plt.tight_layout()
    plot_path = os.path.join(REPORT_OUT_DIR, f"{model_folder}_classifier_report.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print(f"[{model_label}] Completed! Accuracy: {acc*100:.2f}%, Errors: {total_errors}/{len(y_test)}")

    return {
        "label": model_label,
        "folder": model_folder,
        "accuracy": acc,
        "cm": cm,
        "total_test": len(y_test),
        "total_errors": total_errors,
        "plot_path": plot_path,
        "misclassified_angles": misclassified_angles
    }

def main():
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    results = []

    print("Starting full classification & confusion matrix evaluation...")
    for label, folder, handson, data_path in TARGET_MODELS:
        res = evaluate_model(label, folder, handson, data_path)
        if res:
            results.append(res)

    # 統合 Markdown レポート作成
    report_md_path = os.path.join(REPORT_OUT_DIR, "FULL_CLASSIFICATION_REPORT.md")
    with open(report_md_path, "w") as f:
        f.write("# 🧪 VAEモデル構造状態(State 0/1) 分類精度・混同行列・誤判定角度レポート\n\n")
        f.write("## 1. 全モデル精度・混同行列まとめ比較一覧\n\n")
        f.write("| モデル名 | 実験条件 | 分類精度 (Accuracy) | 正解数 / テスト総数 | 誤判定数 | 混同行列 [[TN, FP], [FN, TP]] |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")

        for r in results:
            tn, fp, fn, tp = r["cm"].ravel() if r["cm"].size == 4 else (0, 0, 0, 0)
            f.write(f"| **{r['label']}** | `{r['folder']}` | **{r['accuracy']*100:.2f}%** | {r['total_test']-r['total_errors']}/{r['total_test']} | {r['total_errors']} | `[[TN:{tn}, FP:{fp}], [FN:{fn}, TP:{tp}]]` |\n")

        f.write("\n\n## 2. 誤判定が発生した角度の傾向分析 & 詳細ヒストグラム\n\n")
        for r in results:
            f.write(f"### 🔹 {r['label']} (`{r['folder']}`)\n")
            f.write(f"* **分類精度 (Accuracy)**: **{r['accuracy']*100:.2f}%**\n")
            f.write(f"* **テストデータ数**: {r['total_test']} 件 (誤判定: {r['total_errors']} 件 / 正解: {r['total_test']-r['total_errors']} 件)\n")
            
            if r['total_errors'] > 0:
                angles = r['misclassified_angles']
                unique, counts = np.unique(angles, return_counts=True)
                top_err_angles = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]
                err_str = ", ".join([f"角度 {int(a)}° ({c}件)" for a, c in top_err_angles])
                f.write(f"* **誤判定集中角度 Top 5**: {err_str}\n")
            else:
                f.write("* **誤判定傾向**: 誤判定なし (100% 正解)\n")
            
            f.write(f"* **個別グラフ画像**: `{r['plot_path']}`\n\n")

    print(f"\nReport successfully generated at: {report_md_path}")

if __name__ == "__main__":
    main()
