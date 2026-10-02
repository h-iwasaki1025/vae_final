import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker

def main():
    model_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/20260609y_z_ep'
    output_dir = '/Users/hikaru/Code/2026_Project/07_result/20260609_z_noise10_20260609y_z_ep'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Train data for PCA...")
    z_mean_train = np.load(os.path.join(model_dir, 'z_mean_train.npy'), allow_pickle=True)
    
    print('Calculating PCA on full training data...')
    pca = PCA(n_components=z_mean_train.shape[1])
    pca.fit(z_mean_train)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("Explained variance ratio per PC:")
    for i, ev in enumerate(explained_variance):
        print(f"PC{i+1}: {ev:.4f} ({ev*100:.1f}%)")
        
    print(f"Cumulative variance for 10 PCs: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.1f}%)")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(1, len(explained_variance) + 1)
    
    # Bar plot for individual variance
    bars = ax1.bar(x, explained_variance, alpha=0.7, color='steelblue', label='Individual Explained Variance')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_ylim(0, max(explained_variance) * 1.1)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax1.set_xticks(x)
    
    # Line plot for cumulative variance
    ax2 = ax1.twinx()
    line = ax2.plot(x, cumulative_variance, marker='o', color='darkorange', linewidth=2, label='Cumulative Explained Variance')
    ax2.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9)
                 
    # Add values on the line points
    for i, val in enumerate(cumulative_variance):
        ax2.text(x[i] - 0.1, val + 0.02, f'{val*100:.1f}%', ha='right', va='bottom', fontsize=9, fontweight='bold', color='darkorange')

    plt.title('PCA Explained Variance - Model 20260609y_z_ep (Train Data)', fontsize=14, fontweight='bold', pad=15)
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'pca_explained_variance_y.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"Saved explained variance plot to {save_path}")

if __name__ == '__main__':
    main()
