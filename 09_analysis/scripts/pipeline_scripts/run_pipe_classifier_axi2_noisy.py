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
MAX_ITER = 100
RANDOM_STATE = 42

def parse_noise(path):
    match = re.search(r'sn([0-9.]+)', str(path))
    if match: return float(match.group(1))
    return 0.0

def parse_angles(fname):
    parts = fname.split('.')[0].split('_')
    try:
        if parts[0] in ('0', '1', '2'): return int(parts[2]), int(parts[3]), int(parts[4])
        else: return int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        return 0, 0, 0

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
    angleY_test_f = angles_test_f[:, 1]
    angleZ_test_f = angles_test_f[:, 2]
    
    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train_f)
    z_test_scaled = scaler.transform(z_test_f)
    
    mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER, early_stopping=True, n_iter_no_change=20, random_state=RANDOM_STATE)
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
    
    def plot_heatmap(angle1, angle2, name1, name2, filename):
        bins1 = np.arange(0, 390, 30)
        bins2 = np.arange(0, 390, 30)
        heatmap_data = np.zeros((len(bins1)-1, len(bins2)-1))
        annot_data = np.empty((len(bins1)-1, len(bins2)-1), dtype=object)
        
        for i in range(len(bins1)-1):
            for j in range(len(bins2)-1):
                mask = (angle1 >= bins1[i]) & (angle1 < bins1[i+1]) & \
                       (angle2 >= bins2[j]) & (angle2 < bins2[j+1])
                total = np.sum(mask)
                if total > 0:
                    err_count = np.sum(errors[mask])
                    heatmap_data[i, j] = err_count / total
                    annot_data[i, j] = f"{err_count}/{total}"
                else:
                    heatmap_data[i, j] = np.nan
                    annot_data[i, j] = ""
                    
        plt.figure(figsize=(14, 12))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=annot_data, fmt='', vmin=0.0, vmax=1.0, linewidths=0.5,
                    cbar_kws={'label': 'Error Rate (0.0 to 1.0)'})
        plt.title(f'Error Rate Heatmap ({name1} vs {name2})\\nThreshold={best_thresh:.2f}, Acc={best_acc:.4f}')
        plt.xlabel(f'{name2}')
        plt.ylabel(f'{name1}')
        
        xticks_labels = [f"{bins2[j]}~{bins2[j+1]}°" for j in range(len(bins2)-1)]
        yticks_labels = [f"{bins1[i]}~{bins1[i+1]}°" for i in range(len(bins1)-1)]
        
        plt.xticks(np.arange(len(bins2)-1)+0.5, xticks_labels, rotation=45)
        plt.yticks(np.arange(len(bins1)-1)+0.5, yticks_labels, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()

    plot_heatmap(angleX_test_f, angleY_test_f, 'Angle X', 'Angle Y', 'error_heatmap_X_vs_Y.png')
    plot_heatmap(angleX_test_f, angleZ_test_f, 'Angle X', 'Angle Z', 'error_heatmap_X_vs_Z.png')
    plot_heatmap(angleY_test_f, angleZ_test_f, 'Angle Y', 'Angle Z', 'error_heatmap_Y_vs_Z.png')
    
    def plot_error_rate_bar(angle_arr, name, filename):
        bins = np.arange(0, 390, 10)
        error_rates_arr = []
        labels = []
        for i in range(len(bins)-1):
            mask = (angle_arr >= bins[i]) & (angle_arr < bins[i+1])
            if np.sum(mask) > 0:
                error_rates_arr.append(np.mean(errors[mask]))
                labels.append(f"{bins[i]}")
            else:
                error_rates_arr.append(0)
                labels.append(f"{bins[i]}")
        plt.figure(figsize=(15, 6))
        plt.bar(labels, error_rates_arr, color='orange')
        plt.title(f'Error Rate by {name}')
        plt.xlabel(f'{name} (deg)')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()

    plot_error_rate_bar(angleX_test_f, 'Angle X', 'error_rate_AngleX.png')
    plot_error_rate_bar(angleY_test_f, 'Angle Y', 'error_rate_AngleY.png')
    plot_error_rate_bar(angleZ_test_f, 'Angle Z', 'error_rate_AngleZ.png')
    
    with open(os.path.join(OUTPUT_DIR, 'misclassified_summary.txt'), 'w') as f:
        f.write("Misclassified Data Summary\\n")
        f.write("===========================\\n")
        f.write(f"Total test instances: {len(y_test_f)}\\n")
        f.write(f"Total misclassified: {np.sum(errors)}\\n\\n")
        
        f.write("--- Misclassified by Noise ---\\n")
        unique_n, counts_n = np.unique(noise_test_f[errors], return_counts=True)
        for n, c in zip(unique_n, counts_n):
            f.write(f"Noise {n}: {c} errors\\n")
            
        f.write("\\n--- Misclassified by AngleX ---\\n")
        unique_ax, counts_ax = np.unique(angleX_test_f[errors], return_counts=True)
        for ax, c in zip(unique_ax, counts_ax):
            f.write(f"AngleX {ax}: {c} errors\\n")
            
        f.write("\\n--- Misclassified by AngleY ---\\n")
        unique_ay, counts_ay = np.unique(angleY_test_f[errors], return_counts=True)
        for ay, c in zip(unique_ay, counts_ay):
            f.write(f"AngleY {ay}: {c} errors\\n")

        f.write("\\n--- Misclassified by AngleZ ---\\n")
        unique_az, counts_az = np.unique(angleZ_test_f[errors], return_counts=True)
        for az, c in zip(unique_az, counts_az):
            f.write(f"AngleZ {az}: {c} errors\\n")
            
    print("Done Classifier Evaluation.")

if __name__ == '__main__':
    main()
