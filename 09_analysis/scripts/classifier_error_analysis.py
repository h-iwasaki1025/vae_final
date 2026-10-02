import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers, models

def parse_noise_level(path):
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
    print("Starting Error Analysis based on optimal threshold...")
    
    target_dir = "/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep"
    model_dir = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep"
    out_dir = os.path.join(target_dir, "classifier_results")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    z_mean_train = np.load(os.path.join(model_dir, "z_mean_train.npy"))
    y_train = np.load(os.path.join(model_dir, "y_train.npy")).astype(int)
    z_mean_test = np.load(os.path.join(model_dir, "z_mean_test.npy"))
    y_test = np.load(os.path.join(model_dir, "y_test.npy")).astype(int)
    pdbids_test = np.load(os.path.join(model_dir, "pdbids_test.npy"), allow_pickle=True)
    kills_test = np.load(os.path.join(model_dir, "kills_test.npy"), allow_pickle=True)

    # 2. Extract Angles and Noise
    print("Extracting angles and noise from filenames...")
    angle_x_all = []
    angle_z_all = []
    noise_all = []
    
    for p in kills_test:
        noise_level = parse_noise_level(str(p))
        noise_all.append(noise_level)
        
        fname = os.path.basename(str(p))
        try:
            parts = fname.split('.')[0].split('_')
            if parts[0] in ('0', '1', '2'):
                x_ang = int(parts[2])
                z_ang = int(parts[4])
            else:
                x_ang = int(parts[1])
                z_ang = int(parts[3])
            angle_x_all.append(x_ang)
            angle_z_all.append(z_ang)
        except Exception:
            angle_x_all.append(0)
            angle_z_all.append(0)
    
    angle_x_all = np.array(angle_x_all)
    angle_z_all = np.array(angle_z_all)
    noise_all = np.array(noise_all)

    # 3. Apply Mask (Exclude 1qvi)
    mask = (pdbids_test != "1qvi")
    z_eval = z_mean_test[mask]
    y_eval = y_test[mask]
    angle_x_eval = angle_x_all[mask]
    angle_z_eval = angle_z_all[mask]
    noise_eval = noise_all[mask]

    # 4. Load or Train Model Predictions
    pred_path = os.path.join(out_dir, "y_pred_prob_without_1qvi.npy")
    if os.path.exists(pred_path):
        print("Loading cached predictions to avoid model retraining...")
        y_pred_prob = np.load(pred_path)
    else:
        print("Training MLP model and saving predictions...")
        tf.random.set_seed(42)
        np.random.seed(42)
        
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(z_mean_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(z_mean_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)

        print("Predicting on test data...")
        y_pred_prob = model.predict(z_eval, verbose=0).flatten()
        np.save(pred_path, y_pred_prob)

    # Find optimal threshold using cached predictions
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_acc = 0.0
    acc_list = []
    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        acc = accuracy_score(y_eval, y_pred)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"Optimal Threshold Found: {best_thresh:.2f} (Accuracy: {best_acc:.4f})")
    
    # Plot Accuracy vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, acc_list, color='indigo', linewidth=2)
    plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best Thresh: {best_thresh:.2f}')
    plt.axhline(y=best_acc, color='orange', linestyle=':', label=f'Best Acc: {best_acc:.4f}')
    plt.title('Accuracy vs Threshold (Fixed Predictions)', fontsize=14)
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_vs_threshold_fixed.png'), dpi=300)
    plt.close()
    
    y_pred_best = (y_pred_prob >= best_thresh).astype(int)
    
    # 5. Confusion Matrix (Optimal Threshold)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_eval, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['State 0', 'State 1'],
                yticklabels=['State 0', 'State 1'])
    plt.title(f"Confusion Matrix (Thresh={best_thresh:.2f})\nAccuracy: {best_acc:.4f}", fontsize=14)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cm_optimal_thresh_fixed.png'), dpi=300)
    plt.close()
    
    # 5.b Confusion Matrix (Threshold = 0.50)
    y_pred_50 = (y_pred_prob >= 0.50).astype(int)
    acc_50 = accuracy_score(y_eval, y_pred_50)
    cm_50 = confusion_matrix(y_eval, y_pred_50)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_50, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['State 0', 'State 1'],
                yticklabels=['State 0', 'State 1'])
    plt.title(f"Confusion Matrix (Thresh=0.50)\nAccuracy: {acc_50:.4f}", fontsize=14)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cm_thresh_0.50_fixed.png'), dpi=300)
    plt.close()
    
    # 6. Error Analysis
    print("Performing error analysis...")
    errors_mask = (y_pred_best != y_eval)
    
    error_x = angle_x_eval[errors_mask]
    error_z = angle_z_eval[errors_mask]
    error_noise = noise_eval[errors_mask]

    error_pdb = pdbids_test[mask][errors_mask]

    # Plot 1: Histogram of Error Angles (X) by PDB_ID
    df_x = pd.DataFrame({'AngleX': error_x, 'PDB_ID': error_pdb})
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_x, x='AngleX', hue='PDB_ID', multiple='dodge', bins=36, binrange=(0, 360), palette='Set2', edgecolor='black')
    plt.title(f"Errors by Angle X and PDB_ID (Threshold={best_thresh:.2f})", fontsize=14)
    plt.xlabel("Angle X (degrees)", fontsize=12)
    plt.ylabel("Number of Errors", fontsize=12)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_hist_AngleX_by_pdb_optimal_thresh.png'), dpi=300)
    plt.close()

    # Plot 2: Histogram of Error Angles (Z) by PDB_ID
    df_z = pd.DataFrame({'AngleZ': error_z, 'PDB_ID': error_pdb})
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_z, x='AngleZ', hue='PDB_ID', multiple='dodge', bins=36, binrange=(0, 360), palette='Set2', edgecolor='black')
    plt.title(f"Errors by Angle Z and PDB_ID (Threshold={best_thresh:.2f})", fontsize=14)
    plt.xlabel("Angle Z (degrees)", fontsize=12)
    plt.ylabel("Number of Errors", fontsize=12)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_hist_AngleZ_by_pdb_optimal_thresh.png'), dpi=300)
    plt.close()

    # Plot 3: Noise Levels Error Bar Chart by PDB_ID
    error_pdb = pdbids_test[mask][errors_mask]
    df_noise = pd.DataFrame({
        'NoiseLevel': [f"SN{n}" if n != "inf" else "SN_inf" for n in error_noise],
        'PDB_ID': error_pdb
    })
    
    plt.figure(figsize=(10, 6))
    noise_order = sorted(df_noise['NoiseLevel'].unique())
    sns.countplot(data=df_noise, x='NoiseLevel', hue='PDB_ID', order=noise_order, palette='Set2', edgecolor='gray')
    plt.title(f"Misclassified Count by Noise Level and PDB_ID (Threshold={best_thresh:.2f})", fontsize=14)
    plt.xlabel("Noise Level", fontsize=12)
    plt.ylabel("Error Count", fontsize=12)
    plt.legend(title='PDB_ID')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_noise_by_pdb_optimal_thresh.png'), dpi=300)
    plt.close()

    # Dataframe for Heatmaps
    df = pd.DataFrame({
        'AngleX': angle_x_eval,
        'AngleZ': angle_z_eval,
        'IsError': errors_mask
    })

    # Plot 4: 2D Heatmap (Fine grained / original style)
    summary_fine = df.groupby(['AngleZ', 'AngleX'])['IsError'].agg(['sum', 'count']).reset_index()
    summary_fine['ErrorRate'] = summary_fine['sum'] / summary_fine['count']
    pivot_fine = summary_fine.pivot(index='AngleZ', columns='AngleX', values='ErrorRate').fillna(0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_fine, cmap='rocket_r', vmin=0, vmax=1, cbar_kws={'label': 'Error Rate'})
    plt.title(f"Error Rate Heatmap (Fine)\nThreshold={best_thresh:.2f}, Acc={best_acc:.4f}", fontsize=14)
    plt.xlabel("Angle X", fontsize=12)
    plt.ylabel("Angle Z", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_heatmap_AngleX_vs_AngleZ_fine.png'), dpi=300)
    plt.close()

    # Plot 5: 2D Heatmap (Binned improved style)
    bin_size = 30
    df['Bin_AngleX'] = (df['AngleX'] // bin_size) * bin_size
    df['Bin_AngleZ'] = (df['AngleZ'] // bin_size) * bin_size
    summary_binned = df.groupby(['Bin_AngleZ', 'Bin_AngleX'])['IsError'].agg(['sum', 'count']).reset_index()
    summary_binned['ErrorRate'] = summary_binned['sum'] / summary_binned['count']
    pivot_binned = summary_binned.pivot(index='Bin_AngleZ', columns='Bin_AngleX', values='ErrorRate').fillna(0)
    
    pivot_binned.index = [f"{int(z)}~{int(z)+bin_size}°" for z in pivot_binned.index]
    pivot_binned.columns = [f"{int(x)}~{int(x)+bin_size}°" for x in pivot_binned.columns]

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_binned, cmap='YlOrRd', vmin=0, vmax=1, 
                annot=True, fmt=".2f", annot_kws={"size": 10},
                linewidths=0.5, linecolor='lightgray',
                cbar_kws={'label': 'Error Rate (0.0 to 1.0)'})
    plt.title(f"Error Rate Heatmap (Binned by {bin_size}°)\nThreshold={best_thresh:.2f}, Acc={best_acc:.4f}", fontsize=16)
    plt.xlabel("Angle X", fontsize=14)
    plt.ylabel("Angle Z", fontsize=14)
    plt.gca().invert_yaxis() 
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_heatmap_AngleX_vs_AngleZ_binned.png'), dpi=300)
    plt.close()

    print(f"All error analysis plots saved to {out_dir}")

if __name__ == "__main__":
    main()
