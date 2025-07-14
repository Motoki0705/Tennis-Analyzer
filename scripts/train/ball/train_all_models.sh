#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines return the exit status of the last command to exit with a non-zero status.
set -o pipefail

# --- Configuration ---

# Find the project root directory dynamically
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../" &> /dev/null && pwd)

# Define key directories
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/ball"
HYDRA_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/train_ball"

# Array of model configurations to train
# Add new model config names here to include them in the training cycle.
MODELS_TO_TRAIN=(
    "lite_tracknet_generic"
    "conv3d_tsm_fpn"
    "video_swin_generic"
)

# --- Functions ---

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO] $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS] $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m[WARNING] $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m[ERROR] $1\033[0m" >&2
}

# Function to train a single model and archive its checkpoint
train_and_archive_model() {
    local config_name=$1
    print_info "Starting training for model: ${config_name}"

    # Set a deterministic output path for this run
    local run_output_dir="${HYDRA_OUTPUT_ROOT}/${config_name}"
    
    # Clean up previous run directory to ensure a fresh start
    if [ -d "${run_output_dir}" ]; then
        print_info "Removing previous output directory: ${run_output_dir}"
        rm -rf "${run_output_dir}"
    fi

    # Run the training script using python -m for robust path handling
    if ! python -m src.ball.api.train \
        --config-path="${PROJECT_ROOT}/configs/train/ball" \
        --config-name="${config_name}" \
        hydra.run.dir="${run_output_dir}"; then
        print_error "Training failed for ${config_name}. Aborting."
        exit 1
    fi

    print_success "Training completed for ${config_name}."

    # --- Checkpoint Archiving ---
    local source_ckpt_dir="${run_output_dir}/checkpoints"
    
    # Find the best checkpoint file. Using -print -quit to get only the first match.
    local best_ckpt
    best_ckpt=$(find "${source_ckpt_dir}" -name "*.ckpt" -print -quit)

    if [ -z "${best_ckpt}" ]; then
        print_warning "No checkpoint file found for ${config_name} in ${source_ckpt_dir}. Skipping archival."
        return
    }

    local dest_dir="${CHECKPOINT_ROOT}/${config_name}"
    local dest_path="${dest_dir}/best_model.ckpt"

    print_info "Archiving checkpoint: ${best_ckpt} -> ${dest_path}"
    
    # Create destination directory and move the checkpoint
    mkdir -p "${dest_dir}"
    mv "${best_ckpt}" "${dest_path}"

    print_success "Checkpoint for ${config_name} successfully archived."
}

# --- Main Execution ---

cd "${PROJECT_ROOT}" # Ensure script runs from the project root

print_info "Starting batch training for all ball models..."

for model_config in "${MODELS_TO_TRAIN[@]}"; do
    echo
    print_info "======================================================================"
    print_info "Processing model: ${model_config}"
    print_info "======================================================================"
    train_and_archive_model "${model_config}"
    
    # Optional: Add a small delay between trainings if needed
    # sleep 5 
done

print_info "======================================================================"
print_success "All training jobs completed successfully!"
print_info "======================================================================"
