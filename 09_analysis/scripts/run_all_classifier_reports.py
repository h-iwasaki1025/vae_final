import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf

BASE_SAVE_DIR = "/Users/hikaru/Code/2026_Project/04_Savemodel/"
REPORT_OUT_DIR = "/Users/hikaru/Code/2026_Project/07_result/classifier_summary_report/"

# 解析対象モデル (アニーリング4種 + 3層モデル4種)
TARGET_MODELS = [
    ("Anneal-0.01-PSI", "20260828_anneal_b001_psi_seed420"),
    ("Anneal-1.00-PSI", "20260828_anneal_b100_psi_seed420"),
    ("Anneal-0.01-Multi", "20260828_anneal_b001_multi_seed420"),
    ("Anneal-1.00-Multi", "20260828_anneal_b100_multi_seed420"),
    ("3Layer-PSI-s420", "20260827_02_3layer_psi_seed420"),
    ("3Layer-PSI-s720", "20260827_03_3layer_psi_seed720"),
    ("3Layer-rot90-s420", "20260827_02_3layer_rot90_seed420"),
    ("3Layer-rot90-s720", "20260827_03_3layer_rot90_seed720"),
]

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

def analyze_model(model_label, model_folder):
    save_path = os.path.join(BASE_SAVE_DIR, model_folder)
    if not os.path.exists(save_path):
        return None

    y_train_path = os.path.join(save_path, "y_train.npy")
    y_test_path = os.path.join(save_path, "y_test.npy")
    r_test_path = os.path.join(save_path, "r_test.npy")
    kills_test_path = os.path.join(save_path, "kills_test.npy")

    if not (os.path.exists(y_train_path) and os.path.exists(y_test_path)):
        return None

    y_train = np.load(y_train_path).astype(int)
    y_test = np.load(y_test_path).astype(int)

    # 潜在ベクトル z_mean の読み込み or 推論
    z_train_path = os.path.join(save_path, "z_mean_train.npy")
    z_test_path = os.path.join(save_path, "z_mean_test.npy")

    if os.path.exists(z_train_path) and os.path.exists(z_test_path):
        z_train = np.load(z_train_path)
        z_test = np.load(z_test_path)
    else:
        # Encoder から潜在表現を得る
        encoder_path = os.path.join(save_path, "encoder.h5")
        if not os.path.exists(encoder_path):
            return None
        try:
            encoder = tf.keras.models.load_model(encoder_path, compile=False)
        except Exception:
            return None
        # この場合はz_testのロードは上位データから行う
        return None

    # 角度情報の取得
    if os.path.exists(r_test_path):
        angles_test = np.load(r_test_path)
    elif os.path.exists(kills_test_path):
        kills_test = np.load(kills_test_path, allow_pickle=True)
        angles_test = np.array([extract_angle(k) for k in kills_test])
    else:
        angles_test = np.zeros(len(y_test))

    # ランダムフォレスト/分類器学習
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(z_train, y_train)
    y_pred = clf.predict(z_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 誤判別サンプルの抽出
    misclassified_mask = (y_test != y_pred)
    misclassified_angles = angles_test[misclassified_mask]
    total_misclassified = int(np.sum(misclassified_mask))

    # 各モデル個別のプロット作成
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 混同行列
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["State 0", "State 1"], yticklabels=["State 0", "State 1"])
    axes[0].set_title(f"Confusion Matrix ({model_label})\nAccuracy: {acc*100:.2f}%")
    axes[0].set_xlabel("Predicted State")
    axes[0].set_ylabel("True State")

    # 2. 誤判定が発生した角度のヒストグラム
    if len(misclassified_angles) > 0:
        axes[1].hist(misclassified_angles, bins=20, color="crimson", edgecolor="black", alpha=0.7)
        axes[1].set_title(f"Misclassified Angles Distribution\n(Total Errors: {total_misclassified}/{len(y_test)})")
        axes[1].set_xlabel("Angle (degree/index)")
        axes[1].set_ylabel("Error Count")
    else:
        axes[1].text(0.5, 0.5, "Zero Misclassifications! (100% Acc)", 
                     ha="center", va="center", fontsize=12, color="green")
        axes[1].set_title("Misclassified Angles Distribution")

    plt.tight_layout()
    plot_path = os.path.join(REPORT_OUT_DIR, f"{model_folder}_classifier_analysis.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()

    return {
        "label": model_label,
        "folder": model_folder,
        "accuracy": acc,
        "cm": cm,
        "total_test": len(y_test),
        "total_errors": total_misclassified,
        "plot_path": plot_path,
        "misclassified_angles": misclassified_angles
    }

def main():
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    results = []

    print("Running classification error analysis across all models...")
    for label, folder in TARGET_MODELS:
        res = analyze_model(label, folder)
        if res:
            results.append(res)

    # 統合レポート markdown の生成
    report_md_path = os.path.join(REPORT_OUT_DIR, "CLASSIFICATION_SUMMARY_REPORT.md")
    with open(report_md_path, "w") as f:
        f.write("# VAE Model State Classification & Error Angle Report\n\n")
        f.write("## 1. 全モデル精度・混同行列まとめ\n\n")
        f.write("| モデル | 条件 | 分類精度 (Accuracy) | 正解数 / 全数 | 誤判定数 | 混同行列 [[TN, FP], [FN, TP]] |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")

        for r in results:
            tn, fp, fn, tp = r["cm"].ravel() if r["cm"].size == 4 else (0, 0, 0, 0)
            f.write(f"| **{r['label']}** | `{r['folder']}` | **{r['accuracy']*100:.2f}%** | {r['total_test']-r['total_errors']}/{r['total_test']} | {r['total_errors']} | `[[TN:{tn}, FP:{fp}], [FN:{fn}, TP:{tp}]]` |\n")

        f.write("\n\n## 2. 誤判定が発生した角度の傾向分析\n\n")
        for r in results:
            f.write(f"### 🔹 {r['label']} ({r['folder']})\n")
            f.write(f"- **分類精度**: {r['accuracy']*100:.2f}%\n")
            f.write(f"- **誤判定サンプル数**: {r['total_errors']} / {r['total_test']} 件\n")
            if r['total_errors'] > 0:
                angles = r['misclassified_angles']
                unique, counts = np.unique(angles, return_counts=True)
                top_err_angles = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]
                err_str = ", ".join([f"角度 {a}° ({c}件)" for a, c in top_err_angles])
                f.write(f"- **誤判定が最も集中した角度 Top 5**: {err_str}\n")
            else:
                f.write("- **誤判定**: なし (エラー0件)\n")
            f.write(f"- **個別グラフ**: `07_result/classifier_summary_report/{r['folder']}_classifier_analysis.png`\n\n")

    print(f"Report generated successfully at: {report_md_path}")

if __name__ == "__main__":
    main()
