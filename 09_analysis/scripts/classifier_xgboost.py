import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import xgboost as xgb
import re
from collections import Counter

def parse_noise_level(path):
    # 例: .../noise/sn0.6/... 
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    match = re.search(r'sn([0-9.]+)', path)
    if match:
        return match.group(1)
    
    match_inf = re.search(r'sn(inf)', path)
    if match_inf:
        return "inf"
        
    return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="Latent Space XGBoost Classifier")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing the saved .npy files")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for results")
    args = parser.parse_args()

    target_dir = args.target_dir
    print(f"Loading data from: {target_dir}")

    # Load data
    z_mean_train_path = os.path.join(target_dir, "z_mean_train.npy")
    z_mean_test_path = os.path.join(target_dir, "z_mean_test.npy")
    if not os.path.exists(z_mean_train_path) and args.out_dir and os.path.exists(os.path.join(args.out_dir, "z_mean_train.npy")):
        z_mean_train_path = os.path.join(args.out_dir, "z_mean_train.npy")
        z_mean_test_path = os.path.join(args.out_dir, "z_mean_test.npy")
        
    z_train = np.load(z_mean_train_path)
    z_test = np.load(z_mean_test_path)

    y_train = np.load(os.path.join(target_dir, "y_train.npy"))
    y_test = np.load(os.path.join(target_dir, "y_test.npy"))
    pdbids_test = np.load(os.path.join(target_dir, "pdbids_test.npy"))
    kills_test = np.load(os.path.join(target_dir, "kills_test.npy"))
    
    # Load angles
    x_angle_test = np.load(os.path.join(target_dir, "x_angle_test.npy"))
    
    # Ensure binary labels (0 or 1)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print(f"Train data shape: {z_train.shape}, y_train shape: {y_train.shape}")
    print(f"Test data shape: {z_test.shape}, y_test shape: {y_test.shape}")

    # Output directory for classifier results
    out_dir = args.out_dir if args.out_dir else os.path.join(target_dir, "classifier_xgboost_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")

    # Train XGBoost Model
    print("Training XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(z_train, y_train)

    # Plot Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax, max_num_features=20, height=0.5, color='teal')
    ax.set_title("XGBoost Feature Importance (Latent Dimensions)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=300)
    plt.close(fig)
    print("Saved Feature Importance Plot.")

    def evaluate_and_plot(pattern_name, mask):
        print(f"\n--- Evaluating Pattern: {pattern_name} ---")
        z_eval = z_test[mask]
        y_eval = y_test[mask]
        pdb_eval = pdbids_test[mask]
        kills_eval = kills_test[mask]
        angles_eval = x_angle_test[mask]

        if len(z_eval) == 0:
            print(f"No data for pattern {pattern_name}")
            return

        # Predict
        y_pred = model.predict(z_eval)

        acc = accuracy_score(y_eval, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_eval, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["State 0", "State 1"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"XGB Confusion Matrix ({pattern_name})\nAcc: {acc:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{pattern_name}.png"), dpi=300)
        plt.close(fig)

        # Error Analysis
        errors_mask = (y_eval != y_pred)
        if not np.any(errors_mask):
            print("No errors! Perfect classification.")
            return
            
        error_kills = kills_eval[errors_mask]
        error_angles = angles_eval[errors_mask]

        # Parse Noise Levels
        noise_levels = [parse_noise_level(k) for k in error_kills]
        noise_counts = Counter(noise_levels)
        
        # Plot Noise Levels Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        labels, counts = zip(*noise_counts.most_common())
        ax.bar(labels, counts, color='coral', edgecolor='black')
        ax.set_title(f"XGB Error Analysis: Noise Levels ({pattern_name})", fontsize=14)
        ax.set_xlabel("Noise Level (sn)", fontsize=12)
        ax.set_ylabel("Number of Misclassified Samples", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"error_noise_{pattern_name}.png"), dpi=300)
        plt.close(fig)

        # Plot Angle Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(error_angles, bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f"XGB Error Analysis: Angles ({pattern_name})", fontsize=14)
        ax.set_xlabel("Angle", fontsize=12)
        ax.set_ylabel("Number of Misclassified Samples", fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"error_angle_{pattern_name}.png"), dpi=300)
        plt.close(fig)

    # Pattern 1: without 1qvi
    mask_p1 = (pdbids_test != "1qvi")
    evaluate_and_plot("without_1qvi", mask_p1)

    # Pattern 2: all pdbs
    mask_p2 = np.ones(len(y_test), dtype=bool)
    evaluate_and_plot("all_pdbs", mask_p2)
    
    print(f"\nAll results saved to {out_dir}")

if __name__ == "__main__":
    main()
