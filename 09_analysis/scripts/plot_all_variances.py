import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_variances_for_model(model_name):
    print(f"Generating variance plots for {model_name}...")
    
    model_dir = f'/Users/hikaru/Code/2026_Project/04_Savemodel/{model_name}'
    base_output_dir = f'/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_{model_name}'
    
    # Check if necessary files exist
    if not os.path.exists(os.path.join(model_dir, 'z_mean_test.npy')):
        if not os.path.exists(os.path.join(base_output_dir, 'z_mean_test.npy')):
            print(f"Error: Data not found for {model_name}.")
            return
        else:
            data_dir = base_output_dir
    else:
        data_dir = model_dir

    # Load Data
    z_test = np.load(os.path.join(data_dir, 'z_mean_test.npy'))

    # 1. PCA Explained Variance
    pca = PCA(n_components=10)
    import random
    random.seed(720)
    np.random.seed(720)
    n_vis = min(len(z_test), 20000)
    idx_fit = np.random.choice(len(z_test), n_vis, replace=False)
    pca.fit(z_test[idx_fit])

    evr = pca.explained_variance_ratio_ * 100  # to percentage
    cumulative_evr = np.cumsum(evr)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x_labels = [f"PC {i+1}" for i in range(10)]
    x_pos = np.arange(len(x_labels))
    
    # Bar plot for individual variance
    bars = ax1.bar(x_pos, evr, alpha=0.7, color='steelblue', label='Individual Variance')
    ax1.set_ylabel('Explained Variance Ratio (%)', fontsize=12)
    ax1.set_xlabel('Principal Components', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)

    # Line plot for cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(x_pos, cumulative_evr, marker='o', color='firebrick', linewidth=2, label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax2.set_ylim([0, 105])

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.title(f'PCA Explained Variance Ratio - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, 'pca_explained_variance.png'), dpi=150)
    plt.close(fig)

    # 2. Latent Dimension (Z) Variances
    z_vars = np.var(z_test, axis=0)
    
    # Sort by variance descending
    sorted_idx = np.argsort(z_vars)[::-1]
    sorted_vars = z_vars[sorted_idx]
    sorted_labels = [f"Dim {idx}" for idx in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos_z = np.arange(len(sorted_labels))
    
    bars_z = ax.bar(x_pos_z, sorted_vars, alpha=0.8, color='seagreen')
    
    # Add values on top of bars
    for bar in bars_z:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Variance', fontsize=12)
    ax.set_xlabel('Latent Dimensions (Sorted)', fontsize=12)
    ax.set_title(f'Latent Dimension (Z) Variances - {model_name}', fontsize=14)
    ax.set_xticks(x_pos_z)
    ax.set_xticklabels(sorted_labels)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, 'z_latent_variance.png'), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    models = ["20260609x_z_ep", "20260609x_y_ep", "20260609y_z_ep"]
    for model in models:
        plot_variances_for_model(model)
    print("All variance plots generated successfully.")
