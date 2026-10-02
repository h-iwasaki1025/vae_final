import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import re
from collections import Counter

def parse_noise_level(path):
    # 例: .../noise/sn0.6/... 
    # 文字列の場合はそのまま、バイナリ文字列ならデコード
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    match = re.search(r'sn([0-9.]+)', path)
    if match:
        return match.group(1)
    
    # sninf のようなパターンもあるかもしれない
    match_inf = re.search(r'sn(inf)', path)
    if match_inf:
        return "inf"
        
    return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="Latent Space MLP Classifier")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing the saved .npy files")
    args = parser.parse_args()

    target_dir = args.target_dir
    print(f"Loading data from: {target_dir}")

    # Load data
    z_mean_train = np.load(os.path.join(target_dir, "z_mean_train.npy"))
    y_train = np.load(os.path.join(target_dir, "y_train.npy"))
    z_mean_test = np.load(os.path.join(target_dir, "z_mean_test.npy"))
    y_test = np.load(os.path.join(target_dir, "y_test.npy"))
    pdbids_test = np.load(os.path.join(target_dir, "pdbids_test.npy"))
    kills_test = np.load(os.path.join(target_dir, "kills_test.npy"))
    
    # Load angles (using x_angle as representative rotation angle if available, or z_angle depending on the project. Let's load all and use x_angle as primary)
    x_angle_test = np.load(os.path.join(target_dir, "x_angle_test.npy"))
    
    # Ensure binary labels (0 or 1)
    # y_train and y_test might be floats, convert to int
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print(f"Train data shape: {z_mean_train.shape}, y_train shape: {y_train.shape}")
    print(f"Test data shape: {z_mean_test.shape}, y_test shape: {y_test.shape}")

    # Build MLP Model
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(z_mean_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model
    print("Training MLP Model...")
    history = model.fit(z_mean_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

    # Output directory for classifier results
    out_dir = os.path.join(target_dir, "classifier_results")
    os.makedirs(out_dir, exist_ok=True)

    def evaluate_and_plot(pattern_name, mask):
        print(f"\n--- Evaluating Pattern: {pattern_name} ---")
        z_eval = z_mean_test[mask]
        y_eval = y_test[mask]
        pdb_eval = pdbids_test[mask]
        kills_eval = kills_test[mask]
        angles_eval = x_angle_test[mask]

        if len(z_eval) == 0:
            print(f"No data for pattern {pattern_name}")
            return

        # Predict
        y_pred_prob = model.predict(z_eval)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        acc = accuracy_score(y_eval, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_eval, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["State 0", "State 1"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({pattern_name})\nAcc: {acc:.4f}")
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
        ax.set_title(f"Error Analysis: Noise Levels ({pattern_name})", fontsize=14)
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
        ax.set_title(f"Error Analysis: Angles ({pattern_name})", fontsize=14)
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
    
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
