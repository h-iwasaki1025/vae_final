import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

MODEL_DIR = sys.argv[1]
OUTPUT_DIR = os.path.join(sys.argv[2], "classifier_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MLP architecture: 3 layers, nodes: 200 -> 100 -> 50
HIDDEN_LAYER_SIZES = (200, 100, 50)
MAX_ITER = 300
RANDOM_STATE = 42

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', str(path))
    if match: return float(match.group(1))
    return 0.0

def parse_angles(fname):
    parts = fname.split('.')[0].split('_')
    try:
        if parts[0] in ('0', '1', '2'): return int(parts[2]), int(parts[4])
        else: return int(parts[1]), int(parts[3])
    except Exception:
        return 0, 0

def main():
    print("Running Classifier Evaluation...")
    z_train = np.load(os.path.join(MODEL_DIR, "z_mean_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"), allow_pickle=True)
    kills_train = np.load(os.path.join(MODEL_DIR, "kills_train.npy"), allow_pickle=True)
    z_test = np.load(os.path.join(MODEL_DIR, "z_mean_test.npy"), allow_pickle=True)
    pdbids_test = np.load(os.path.join(MODEL_DIR, "pdbids_test.npy"), allow_pickle=True)
    kills_test = np.load(os.path.join(MODEL_DIR, "kills_test.npy"), allow_pickle=True)
    
    # Use all train data (state 0 and 1)
    z_train_f = z_train
    y_train_f = y_train
    
    mask_test = (pdbids_test != '1qvi')
    z_test_f = z_test[mask_test]
    pdbids_test_f = pdbids_test[mask_test]
    kills_test_f = kills_test[mask_test]
    y_test_f = np.where(pdbids_test_f == '1l2o', 0, 1) # 1l2o=0, 2ec6=1
    
    noise_test_f = np.array([parse_noise(p) for p in kills_test_f])
    angles_test_f = np.array([parse_angles(os.path.basename(p)) for p in kills_test_f])
    angleX_test_f = angles_test_f[:, 0]
    angleZ_test_f = angles_test_f[:, 1]
    
    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train_f)
    z_test_scaled = scaler.transform(z_test_f)
    
    mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    mlp.fit(z_train_scaled, y_train_f)
    
    y_pred_proba = mlp.predict_proba(z_test_scaled)[:, 1]
    
    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    best_acc = 0; best_thresh = 0.5
    for t in thresholds:
        pred = (y_pred_proba >= t).astype(int)
        acc = accuracy_score(y_test_f, pred)
        accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc; best_thresh = t
            
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy', color='blue', linewidth=2)
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold: {best_thresh:.2f}\\nAcc: {best_acc:.4f}')
    plt.title('Accuracy vs Classification Threshold (Without 1qvi)')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_threshold_without_1qvi.png'))
    plt.close()
    
    y_pred_best = (y_pred_proba >= best_thresh).astype(int)
    cm = confusion_matrix(y_test_f, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['1l2o(0)', '2ec6(1)'], yticklabels=['1l2o(0)', '2ec6(1)'])
    plt.title(f'Confusion Matrix at Best Threshold ({best_thresh:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(OUTPUT_DIR, 'cm_best_threshold_without_1qvi.png'))
    plt.close()
    
    errors = (y_test_f != y_pred_best)
    unique_noises = np.unique(noise_test_f)
    error_rates = {}
    for noise in unique_noises:
        mask = (noise_test_f == noise)
        if np.sum(mask) > 0:
            error_rates[noise] = np.mean(errors[mask])
            
    plt.figure(figsize=(10, 6))
    plt.bar([str(n) for n in error_rates.keys()], list(error_rates.values()), color='salmon')
    plt.title('Error Rate by Noise Level (Best Threshold)')
    plt.xlabel('Noise Level (sn)')
    plt.ylabel('Error Rate')
    plt.grid(axis='y')
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_noise_optimal_thresh.png'))
    plt.close()
    
    # Heatmap
    y_bins = np.arange(0, 390, 30); z_bins = np.arange(0, 390, 30)
    heatmap_data = np.zeros((len(y_bins)-1, len(z_bins)-1))
    
    for i in range(len(y_bins)-1):
        for j in range(len(z_bins)-1):
            mask = (angleX_test_f >= y_bins[i]) & (angleX_test_f < y_bins[i+1]) & \
                   (angleZ_test_f >= z_bins[j]) & (angleZ_test_f < z_bins[j+1])
            if np.sum(mask) > 0:
                heatmap_data[i, j] = np.mean(errors[mask])
            else:
                heatmap_data[i, j] = np.nan
                
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap='Reds', annot=False, vmin=0, vmax=np.nanmax(heatmap_data))
    plt.title('Error Rate Heatmap (Angle X vs Angle Z)')
    plt.xlabel('Angle Z (bins of 30 deg)')
    plt.ylabel('Angle X (bins of 30 deg)')
    plt.xticks(np.arange(len(z_bins)-1)+0.5, [f"{z_bins[j]}-{z_bins[j+1]}" for j in range(len(z_bins)-1)], rotation=45)
    plt.yticks(np.arange(len(y_bins)-1)+0.5, [f"{y_bins[i]}-{y_bins[i+1]}" for i in range(len(y_bins)-1)], rotation=0)
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_heatmap_AngleX_vs_AngleZ_improved.png'))
    plt.close()
    
    with open(os.path.join(OUTPUT_DIR, 'misclassified_summary.txt'), 'w') as f:
        f.write("Misclassified Data Summary\n")
        f.write("===========================\n")
        f.write(f"Total test instances: {len(y_test_f)}\n")
        f.write(f"Total misclassified: {np.sum(errors)}\n\n")
        
        f.write("--- Misclassified by Noise ---\n")
        unique_n, counts_n = np.unique(noise_test_f[errors], return_counts=True)
        for n, c in zip(unique_n, counts_n):
            f.write(f"Noise {n}: {c} errors\n")
            
        f.write("\n--- Misclassified by AngleX ---\n")
        unique_ax, counts_ax = np.unique(angleX_test_f[errors], return_counts=True)
        for ax, c in zip(unique_ax, counts_ax):
            f.write(f"AngleX {ax}: {c} errors\n")
            
        f.write("\n--- Misclassified by AngleZ ---\n")
        unique_az, counts_az = np.unique(angleZ_test_f[errors], return_counts=True)
        for az, c in zip(unique_az, counts_az):
            f.write(f"AngleZ {az}: {c} errors\n")
            
    print("Done Classifier Evaluation.")

if __name__ == '__main__':
    main()
