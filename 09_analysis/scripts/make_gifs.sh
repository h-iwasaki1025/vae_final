#!/bin/bash
BASE_DIR="/Users/hikaru/Code/2026_Project/03_Output"
DIRS=("20260612_01_axi2" "20260612_01pdb_axi_inplane" "20260612pdb_axi_inplane")

for DIR in "${DIRS[@]}"; do
    TARGET_DIR="$BASE_DIR/$DIR"
    echo "Processing $TARGET_DIR..."
    
    # 1. 10D Scatter Matrices
    if ls $TARGET_DIR/vae_scatter_matrix_ep*.png 1> /dev/null 2>&1; then
        echo "Creating anim_scatter_matrix.gif..."
        convert -delay 20 -loop 0 $TARGET_DIR/vae_scatter_matrix_ep*.png $TARGET_DIR/anim_scatter_matrix.gif
    else
        echo "No scatter matrix images found in $DIR"
    fi
    
    # 2. 2D PCA (Wide)
    if ls $TARGET_DIR/vae_pca_wide_ep*.png 1> /dev/null 2>&1; then
        echo "Creating anim_pca_wide.gif..."
        convert -delay 20 -loop 0 $TARGET_DIR/vae_pca_wide_ep*.png $TARGET_DIR/anim_pca_wide.gif
    else
        echo "No pca wide images found in $DIR"
    fi
    
    # 3. Recon diff
    if ls $TARGET_DIR/recon/vae_recondiff_ep*.png 1> /dev/null 2>&1; then
        echo "Creating anim_recon.gif..."
        convert -delay 20 -loop 0 $TARGET_DIR/recon/vae_recondiff_ep*.png $TARGET_DIR/anim_recon.gif
    else
        echo "No recon images found in $DIR/recon"
    fi
    
    echo "Done with $DIR."
done
echo "All GIF generation complete."
