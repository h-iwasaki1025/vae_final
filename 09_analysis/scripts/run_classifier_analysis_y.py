import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', str(path))
    if match:
        return float(match.group(1))
    return 0.0

def parse_angles(fname):
    parts = fname.split('.')[0].split('_')
    try:
        if parts[0] in ('0', '1', '2'):
            y_ang = int(parts[3])
            z_ang = int(parts[4])
        else:
            y_ang = int(parts[2])
            z_ang = int(parts[3])
        return y_ang, z_ang
    except Exception:
        return 0, 0

def fix_paths(paths):
    fixed = []
    for p in paths:
        p_str = str(p)
        if p_str.startswith('/Volumes/hikaru/'):
            fixed.append(p_str.replace('/Volumes/hikaru/', '/Users/hikaru/', 1))
        elif p_str.startswith('/Users/hikarui./'):
            fixed.append(p_str.replace('/Users/hikarui./', '/Users/hikaru/', 1))
        else:
            fixed.append(p_str)
    return np.array(fixed)

def main():
    model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep'
    out_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep/classifier_results'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data...")
    z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(model_dir, 'y_train.npy'), allow_pickle=True)
    pdbids_train = np.load(os.path.join(model_dir, 'pdbids_train.npy'), allow_pickle=True)
    
    z_mean_test = np.load(os.path.join(model_dir, 'z_mean_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(model_dir, 'y_test.npy'), allow_pickle=True)
    pdbids_test = np.load(os.path.join(model_dir, 'pdbids_test.npy'), allow_pickle=True)
    kills_test = fix_paths(np.load(os.path.join(model_dir, 'kills_test.npy'), allow_pickle=True))
    
    # Extract Noise and Angles for Test data
    noise_test = np.array([parse_noise(p) for p in kills_test])
    
    angle_y_test = []
    angle_z_test = []
    for p in kills_test:
        fname = os.path.basename(p)
        y_ang, z_ang = parse_angles(fname)
        angle_y_test.append(y_ang)
        angle_z_test.append(z_ang)
    angle_y_test = np.array(angle_y_test)
    angle_z_test = np.array(angle_z_test)
    
    # --- Filter out 1qvi ---
    print("Filtering out 1qvi...")
    train_mask = (pdbids_train != '1qvi')
    z_train_filt = z_mean_train[train_mask]
    y_train_filt = y_train[train_mask]
    
    test_mask = (pdbids_test != '1qvi')
    z_test_filt = z_mean_test[test_mask]
    y_test_filt = y_test[test_mask]
    pdbids_test_filt = pdbids_test[test_mask]
    noise_test_filt = noise_test[test_mask]
    angle_y_test_filt = angle_y_test[test_mask]
    angle_z_test_filt = angle_z_test[test_mask]
    
    print(f"Train data shape: {z_train_filt.shape}, Test data shape: {z_test_filt.shape}")
    
    # --- Train MLP Classifier ---
    print("Training MLP classifier...")
    # Fix random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = keras.Sequential([
        layers.InputLayer(input_shape=(z_train_filt.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(z_train_filt, y_train_filt, epochs=50, batch_size=64, validation_split=0.1, verbose=0)
    
    # Save the model
    model_save_path = os.path.join(out_dir, 'mlp_model_without_1qvi.keras')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # --- Accuracy vs Threshold ---
    print("Evaluating thresholds...")
    y_pred_prob = model.predict(z_test_filt).flatten()
    
    thresholds = np.linspace(0.01, 0.99, 99)
    accuracies = []
    
    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        acc = accuracy_score(y_test_filt, y_pred)
        accuracies.append(acc)
        
    best_idx = np.argmax(accuracies)
    best_thresh = thresholds[best_idx]
    best_acc = accuracies[best_idx]
    
    print(f"Best Threshold: {best_thresh:.2f}, Best Accuracy: {best_acc:.4f}")
    
    # Plot Accuracy vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, marker='.', color='royalblue', linewidth=2)
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold ({best_thresh:.2f})')
    plt.title('Test Accuracy vs Classification Threshold (without 1qvi)', fontsize=14)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_vs_threshold.png'), dpi=150)
    plt.close()
    
    # --- Confusion Matrix ---
    y_pred_best = (y_pred_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_test_filt, y_pred_best)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['1l2o (0)', '2ec6 (1)'], yticklabels=['1l2o (0)', '2ec6 (1)'])
    plt.title(f'Confusion Matrix at Threshold {best_thresh:.2f}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cm_optimal_thresh.png'), dpi=150)
    plt.close()
    
    # --- Error Analysis ---
    print("Performing error analysis...")
    errors_mask = (y_test_filt != y_pred_best)
    
    # 1. Error by Noise
    unique_noises = sorted(np.unique(noise_test_filt))
    error_rates = []
    for n in unique_noises:
        idx = (noise_test_filt == n)
        if np.sum(idx) > 0:
            err_rate = np.mean(errors_mask[idx])
            error_rates.append(err_rate)
        else:
            error_rates.append(0)
            
    plt.figure(figsize=(8, 5))
    plt.bar([str(n) for n in unique_noises], error_rates, color='salmon', alpha=0.8)
    plt.title('Error Rate by Noise Level', fontsize=14)
    plt.xlabel('Noise Level (sn)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_noise.png'), dpi=150)
    plt.close()
    
    # 2. Error by Angle Y and Angle Z
    error_angles_y = angle_y_test_filt[errors_mask]
    error_angles_z = angle_z_test_filt[errors_mask]
    
    plt.figure(figsize=(8, 5))
    plt.hist(error_angles_y, bins=12, range=(0, 360), color='mediumpurple', alpha=0.8, edgecolor='black')
    plt.title('Error Count by Angle Y', fontsize=14)
    plt.xlabel('Angle Y (deg)', fontsize=12)
    plt.ylabel('Error Count', fontsize=12)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_hist_AngleY.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.hist(error_angles_z, bins=12, range=(0, 360), color='mediumseagreen', alpha=0.8, edgecolor='black')
    plt.title('Error Count by Angle Z', fontsize=14)
    plt.xlabel('Angle Z (deg)', fontsize=12)
    plt.ylabel('Error Count', fontsize=12)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_hist_AngleZ.png'), dpi=150)
    plt.close()
    
    # 3. Heatmap Angle Y vs Angle Z
    # Calculate error rate per 30-deg bin
    bins = np.arange(0, 361, 30)
    total_counts, xedges, yedges = np.histogram2d(angle_y_test_filt, angle_z_test_filt, bins=[bins, bins])
    error_counts, _, _ = np.histogram2d(error_angles_y, error_angles_z, bins=[bins, bins])
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        error_heatmap = np.nan_to_num(error_counts / total_counts)
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(error_heatmap.T, cmap='Reds', annot=False, 
                xticklabels=[f"{int(e)}" for e in xedges[:-1]], 
                yticklabels=[f"{int(e)}" for e in yedges[:-1]])
    plt.gca().invert_yaxis()
    plt.title('Error Rate Heatmap (Angle Y vs Angle Z)', fontsize=14)
    plt.xlabel('Angle Y (deg)', fontsize=12)
    plt.ylabel('Angle Z (deg)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_heatmap_AngleY_vs_AngleZ.png'), dpi=150)
    plt.close()
    
    print("All classifier analysis plots generated successfully.")

if __name__ == '__main__':
    main()
