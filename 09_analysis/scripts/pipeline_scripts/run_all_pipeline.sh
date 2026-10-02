#!/bin/bash
MODEL_DIR=${1:-"/Users/hikaru/Code/2026_Project/04_Savemodel/20260612pdb_axi2normal"}
OUTPUT_DIR=${2:-"/Users/hikaru/Code/2026_Project/07_result/20260612pdb_axi2normal_fixed"}
PYTHON_BIN="/Users/hikaru/Code/miniforge3/envs/tf_vae/bin/python3"
SCRIPT_DIR="/Users/hikaru/Code/2026_Project/06_Analysis/pipeline_scripts"

echo "=== Starting Full Pipeline (Fixed Angles) ==="
echo "Model: $MODEL_DIR"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

$PYTHON_BIN $SCRIPT_DIR/run_pipe_infer_additional.py "$MODEL_DIR"
$PYTHON_BIN $SCRIPT_DIR/run_pipe_pca_exp.py "$MODEL_DIR" "$OUTPUT_DIR"
$PYTHON_BIN $SCRIPT_DIR/run_pipe_recon.py "$MODEL_DIR" "$OUTPUT_DIR"
$PYTHON_BIN $SCRIPT_DIR/run_pipe_classifier.py "$MODEL_DIR" "$OUTPUT_DIR"
$PYTHON_BIN $SCRIPT_DIR/run_pipe_scatter.py "$MODEL_DIR" "$OUTPUT_DIR"
$PYTHON_BIN $SCRIPT_DIR/run_pipe_individual.py "$MODEL_DIR" "$OUTPUT_DIR"
$PYTHON_BIN $SCRIPT_DIR/run_pipe_angle_maps.py "$MODEL_DIR" "$OUTPUT_DIR"

echo "=== Pipeline Finished ==="
