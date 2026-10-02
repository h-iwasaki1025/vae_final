import os
import glob
from PIL import Image

target_dir = "/Users/hikaru/Code/2026_Project/03_Output/20260612pdb_align_inplane"

def make_gif(pattern, output_path, duration=200):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No images found for {pattern}")
        return
        
    print(f"Creating {output_path} from {len(files)} images...")
    images = [Image.open(f) for f in files]
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print(f"Saved {output_path}")

print(f"--- Processing {os.path.basename(target_dir)} ---")
make_gif(os.path.join(target_dir, "vae_scatter_matrix_ep*.png"), os.path.join(target_dir, "anim_scatter_matrix.gif"))
make_gif(os.path.join(target_dir, "vae_pca_wide_ep*.png"), os.path.join(target_dir, "anim_pca_wide.gif"))
make_gif(os.path.join(target_dir, "recon", "vae_recondiff_ep*.png"), os.path.join(target_dir, "anim_recon.gif"))
