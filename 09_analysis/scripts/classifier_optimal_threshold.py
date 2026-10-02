import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    print("Starting optimal threshold search for classifier...")
    
    # ユーザーが指定した結果出力先
    target_dir = "/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609x_z_ep"
    # データが保存されているモデルディレクトリ
    model_dir = "/Users/hikaru/Code/2026_Project/04_Savemodel/20260609x_z_ep"
    
    out_dir = os.path.join(target_dir, "classifier_results")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    z_mean_train = np.load(os.path.join(model_dir, "z_mean_train.npy"))
    y_train = np.load(os.path.join(model_dir, "y_train.npy")).astype(int)
    z_mean_test = np.load(os.path.join(model_dir, "z_mean_test.npy"))
    y_test = np.load(os.path.join(model_dir, "y_test.npy")).astype(int)

    # State が 0, 1 以外の値(例えば 2 など)を含む可能性があるので、
    # 過去の classifier.py に倣って二値分類 (0 と 1) が前提のモデルを使いますが、
    # もし y_train や y_test に 2 が含まれている場合はエラーになるか動作がおかしくなる可能性があります。
    # 過去の `cm` のラベルが ["State 0", "State 1"] なので、基本的には 0, 1 であると仮定します。
    
    # 2. Build & Train MLP
    print("Training MLP model...")
    # Seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(z_mean_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # 軽く20エポックで学習（元のclassifier.pyと同じ）
    model.fit(z_mean_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)

    # 3. Predict on Test
    print("Predicting on test data (excluding 1qvi)...")
    pdbids_test = np.load(os.path.join(model_dir, "pdbids_test.npy"), allow_pickle=True)
    mask = (pdbids_test != "1qvi")
    z_eval = z_mean_test[mask]
    y_eval = y_test[mask]
    
    y_pred_prob = model.predict(z_eval, verbose=0).flatten()

    # 4. Search Threshold
    print("Searching for optimal threshold...")
    thresholds = np.linspace(0.01, 0.99, 99)
    accuracies = []
    best_thresh = 0.5
    best_acc = 0.0

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        acc = accuracy_score(y_eval, y_pred)
        accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"Best Threshold: {best_thresh:.2f} with Accuracy: {best_acc:.4f}")

    # 5. Plot Accuracy vs Threshold
    print("Plotting Accuracy vs Threshold...")
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, marker='o', markersize=3, color='b', linewidth=1)
    plt.axvline(x=best_thresh, color='r', linestyle='--', 
                label=f'Best Threshold = {best_thresh:.2f}\nAccuracy = {best_acc:.4f}')
    plt.title('Classification Accuracy vs. Threshold (without 1qvi)', fontsize=14)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_vs_threshold_without_1qvi.png'), dpi=300)
    plt.close()

    # 6. Confusion Matrix with Best Threshold
    print("Generating Confusion Matrix for best threshold...")
    y_pred_best = (y_pred_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_eval, y_pred_best)
    
    unique_labels = np.unique(y_eval)
    display_labels = [f"State {val}" for val in unique_labels]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix (Best Thresh={best_thresh:.2f})\nAccuracy: {best_acc:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cm_best_threshold_without_1qvi.png'), dpi=300)
    plt.close(fig)

    # 7. Confusion Matrix with Threshold = 0.5
    print("Generating Confusion Matrix for threshold = 0.50...")
    thresh_05 = 0.50
    y_pred_05 = (y_pred_prob >= thresh_05).astype(int)
    cm_05 = confusion_matrix(y_eval, y_pred_05)
    
    disp_05 = ConfusionMatrixDisplay(confusion_matrix=cm_05, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_05.plot(ax=ax, cmap="Blues")
    acc_05 = accuracy_score(y_eval, y_pred_05)
    ax.set_title(f"Confusion Matrix (Thresh={thresh_05:.2f})\nAccuracy: {acc_05:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cm_threshold_0.50_without_1qvi.png'), dpi=300)
    plt.close(fig)

    print(f"All done! Results saved in {out_dir}")

if __name__ == "__main__":
    main()
