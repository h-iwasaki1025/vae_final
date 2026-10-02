import os
import glob
from PIL import Image

BASE_DIR = "/Users/hikaru/Code/2026_Project/03_Output"
DIRS = ["20260612_01_axi2", "20260612_01pdb_axi_inplane", "20260612pdb_axi_inplane"]

def make_gif(pattern, output_path, duration=200):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No images found for {pattern}")
        return
        
    print(f"Creating {output_path} from {len(files)} images...")
    images = [Image.open(f) for f in files]
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print(f"Saved {output_path}")

def main():
    for d in DIRS:
        target_dir = os.path.join(BASE_DIR, d)
        print(f"--- Processing {d} ---")
        
        # 1. Scatter Matrix
        pattern_scatter = os.path.join(target_dir, "vae_scatter_matrix_ep*.png")
        out_scatter = os.path.join(target_dir, "anim_scatter_matrix.gif")
        make_gif(pattern_scatter, out_scatter)
        
        # 2. PCA Wide
        pattern_pca = os.path.join(target_dir, "vae_pca_wide_ep*.png")
        out_pca = os.path.join(target_dir, "anim_pca_wide.gif")
        make_gif(pattern_pca, out_pca)
        
        # 3. Recon
        pattern_recon = os.path.join(target_dir, "recon", "vae_recondiff_ep*.png")
        out_recon = os.path.join(target_dir, "anim_recon.gif")
        make_gif(pattern_recon, out_recon)

if __name__ == '__main__':
    main()
