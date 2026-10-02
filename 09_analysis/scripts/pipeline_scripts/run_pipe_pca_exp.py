import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

MODEL_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running PCA Explained Variance...")
z_test = np.load(os.path.join(MODEL_DIR, "z_mean_test.npy"), allow_pickle=True)
np.random.seed(720)
sample_indices = np.random.choice(len(z_test), min(20000, len(z_test)), replace=False)
pca = PCA(n_components=10)
pca.fit(z_test[sample_indices])

explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), explained_variance_ratio, alpha=0.7, color='blue', label='Individual explained variance')
plt.step(range(1, 11), np.cumsum(explained_variance_ratio), where='mid', color='red', label='Cumulative explained variance')
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, 11))
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

out_path = os.path.join(OUTPUT_DIR, 'pca_explained_variance.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {out_path}")
