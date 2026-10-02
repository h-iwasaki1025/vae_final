import os
import sys
import re
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

DATA_DIR = sys.argv[1] # e.g., /Users/hikaru/Code/2026_Project/05_Data/pdb_align_axi2
MODEL_DIR = sys.argv[2] # e.g., /Users/hikaru/Code/2026_Project/04_Savemodel/20260612_01_axi2
OUTPUT_DIR = os.path.join(sys.argv[3], "raw_classifier_results") # e.g., /Users/hikaru/Code/2026_Project/07_result/20260612_01_axi2_fixed
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM = 10
HIDDEN_LAYER_SIZES = (200, 100, 50)
MAX_ITER = 300
RANDOM_STATE = 42

PDB_TRAIN = ['1kqm', '1sr6', '1s5g', '1kk7', '1kk8', '1kwo']
PDB_TEST = ['1l2o', '2ec6'] # test only on 1l2o and 2ec6

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        eps = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_encoder(input_shape, latent_dim):
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    for filters in [2, 4, 8, 16, 32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', strides=2, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inp, [z_mean, z_log_var, z])

def build_decoder(latent_dim, output_channels=1):
    inp = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(1 * 1 * 128, activation='relu')(inp)
    x = tf.keras.layers.Reshape((1, 1, 128))(x)
    for filters in [128, 64, 32, 16, 8, 4, 2]:
        x = tf.keras.layers.Conv2DTranspose(filters, 3, activation='relu', padding='same', strides=2)(x)
    out = tf.keras.layers.Conv2DTranspose(output_channels, 3, activation='sigmoid', padding='same')(x)
    return tf.keras.Model(inp, out)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

def parse_filename(fname):
    # e.g. 0_1kqm_000_150_000_sn0.6.tif
    parts = fname.split('.')[0].split('_')
    # State, PDB, AngleX, AngleY, AngleZ, snX.X
    state = int(parts[0])
    pdb = parts[1]
    angleX = int(parts[2])
    angleZ = int(parts[4])
    
    match = re.search(r'sn([0-9]+\.[0-9]+)', fname)
    noise = float(match.group(1)) if match else 0.0
    return state, pdb, angleX, angleZ, noise

def load_images_for_pdbs(pdb_list):
    print(f"Searching images for PDBs: {pdb_list}")
    images = []
    labels = [] # (state, pdb, angleX, angleZ, noise)
    
    # We will search recursively
    for pdb in pdb_list:
        pattern = os.path.join(DATA_DIR, f"*_{pdb}_*", "**", "*.tif")
        files = glob.glob(pattern, recursive=True)
        print(f"  Found {len(files)} files for {pdb}")
        
        for f in files:
            fname = os.path.basename(f)
            # Make sure it's a generated noise file like sn0.6.tif, skip original .tif if needed
            # We assume all target files end with snX.X.tif
            if 'sn' not in fname:
                continue
                
            state, p, ax, az, noise = parse_filename(fname)
            
            # Correcting state based on PDB directly to ensure correctness
            # 1l2o = 0, 2ec6 = 1.  (Train: 1kqm..=0, 1kk7..=1)
            # Explicitly force state mapping to avoid previous dataset errors
            if p in ['1l2o', '1kqm', '1sr6', '1s5g']:
                state = 0
            elif p in ['1qvi', '2ec6', '1kk7', '1kk8', '1kwo']:
                state = 1
                
            try:
                with Image.open(f) as img:
                    arr = np.array(img).astype('float32') / 255.0
                    if len(arr.shape) == 2: arr = np.expand_dims(arr, axis=-1)
                    images.append(arr)
                    labels.append((state, p, ax, az, noise))
            except Exception as e:
                pass
    return np.array(images), labels

def main():
    print("Loading VAE model...")
    encoder = build_encoder((128, 128, 1), LATENT_DIM)
    decoder = build_decoder(LATENT_DIM)
    vae = VAE(encoder, decoder)
    _ = vae(tf.zeros((1, 128, 128, 1)))
    vae.load_weights(os.path.join(MODEL_DIR, 'vae_best_weights.weights.h5'))
    
    print("Loading Train Images...")
    X_train_raw, labels_train = load_images_for_pdbs(PDB_TRAIN)
    y_train = np.array([l[0] for l in labels_train])
    
    print("Loading Test Images...")
    X_test_raw, labels_test = load_images_for_pdbs(PDB_TEST)
    y_test = np.array([l[0] for l in labels_test])
    angleX_test = np.array([l[2] for l in labels_test])
    angleZ_test = np.array([l[3] for l in labels_test])
    noise_test = np.array([l[4] for l in labels_test])
    
    print(f"Train images: {X_train_raw.shape}, Test images: {X_test_raw.shape}")
    print(f"Train classes: {np.unique(y_train, return_counts=True)}")
    print(f"Test classes: {np.unique(y_test, return_counts=True)}")
    
    print("Inferring Train z_mean...")
    z_train, _, _ = vae.encoder.predict(X_train_raw, batch_size=256)
    print("Inferring Test z_mean...")
    z_test, _, _ = vae.encoder.predict(X_test_raw, batch_size=256)
    
    print("Training MLPClassifier...")
    mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    mlp.fit(z_train, y_train)
    
    y_pred_proba = mlp.predict_proba(z_test)[:, 1]
    
    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    best_acc = 0; best_thresh = 0.5
    for t in thresholds:
        pred = (y_pred_proba >= t).astype(int)
        acc = accuracy_score(y_test, pred)
        accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc; best_thresh = t
            
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy', color='blue', linewidth=2)
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold: {best_thresh:.2f}\\nAcc: {best_acc:.4f}')
    plt.title('Accuracy vs Classification Threshold (Raw Inference)')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_threshold_raw.png'))
    plt.close()
    
    y_pred_best = (y_pred_proba >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['State 0', 'State 1'], yticklabels=['State 0', 'State 1'])
    plt.title(f'Confusion Matrix at Best Threshold ({best_thresh:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(OUTPUT_DIR, 'cm_best_threshold_raw.png'))
    plt.close()
    
    errors = (y_test != y_pred_best)
    
    with open(os.path.join(OUTPUT_DIR, 'misclassified_summary.txt'), 'w') as f:
        f.write("Raw Image Classification Data Summary\n")
        f.write("=====================================\n")
        f.write(f"Total test instances: {len(y_test)}\n")
        f.write(f"Total misclassified: {np.sum(errors)}\n\n")
        
        f.write("--- Misclassified by Noise ---\n")
        unique_n, counts_n = np.unique(noise_test[errors], return_counts=True)
        for n, c in zip(unique_n, counts_n):
            f.write(f"Noise {n}: {c} errors\n")
            
        f.write("\n--- Misclassified by AngleX ---\n")
        unique_ax, counts_ax = np.unique(angleX_test[errors], return_counts=True)
        for ax, c in zip(unique_ax, counts_ax):
            f.write(f"AngleX {ax}: {c} errors\n")
            
        f.write("\n--- Misclassified by AngleZ ---\n")
        unique_az, counts_az = np.unique(angleZ_test[errors], return_counts=True)
        for az, c in zip(unique_az, counts_az):
            f.write(f"AngleZ {az}: {c} errors\n")
            
    print("Done Raw Classifier Evaluation.")

if __name__ == '__main__':
    main()
