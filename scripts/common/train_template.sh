#!/bin/bash

# Standardized Training Script Template
# Usage: train_template.sh <module_name> <models_to_train>

set -e
set -u
set -o pipefail

# Parse arguments
MODULE_NAME="${1:-ball}"
shift
MODELS_TO_TRAIN=("$@")

# Configuration
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../" &> /dev/null && pwd)

CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${MODULE_NAME}"
HYDRA_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/train_${MODULE_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs/training"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG_FILE="${LOG_DIR}/batch_training_${MODULE_NAME}_${TIMESTAMP}.log"

# Functions
print_info() {
    local msg="[INFO] $1"
    echo -e "\033[1;34m${msg}\033[0m"
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${msg}" >> "${BATCH_LOG_FILE}"
}

print_success() {
    local msg="[SUCCESS] $1"
    echo -e "\033[1;32m${msg}\033[0m"
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${msg}" >> "${BATCH_LOG_FILE}"
}

print_warning() {
    local msg="[WARNING] $1"
    echo -e "\033[1;33m${msg}\033[0m"
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${msg}" >> "${BATCH_LOG_FILE}"
}

print_error() {
    local msg="[ERROR] $1"
    echo -e "\033[1;31m${msg}\033[0m" >&2
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${msg}" >> "${BATCH_LOG_FILE}"
}

find_best_checkpoint() {
    local source_ckpt_dir=$1
    local config_name=$2
    
    print_info "Searching for best checkpoint in ${source_ckpt_dir}" >&2
    
    local best_ckpt
    best_ckpt=$(find "${source_ckpt_dir}" -name "*best*.ckpt" -type f | head -n 1)
    
    if [ -n "${best_ckpt}" ]; then
        print_info "Found best checkpoint: ${best_ckpt}" >&2
        echo "${best_ckpt}"
        return 0
    fi
    
    best_ckpt=$(find "${source_ckpt_dir}" -name "*last*.ckpt" -type f | head -n 1)
    
    if [ -n "${best_ckpt}" ]; then
        print_info "Found last checkpoint: ${best_ckpt}" >&2
        echo "${best_ckpt}"
        return 0
    fi
    
    best_ckpt=$(find "${source_ckpt_dir}" -name "*.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
    
    if [ -n "${best_ckpt}" ]; then
        print_info "Found most recent checkpoint: ${best_ckpt}" >&2
        echo "${best_ckpt}"
        return 0
    fi
    
    print_warning "No checkpoint found in ${source_ckpt_dir}" >&2
    return 1
}

train_and_archive_model() {
    local config_name=$1
    print_info "Starting training for model: ${config_name}"
    
    local start_time=$(date +%s)
    local run_output_dir="${HYDRA_OUTPUT_ROOT}/${config_name}/${TIMESTAMP}"
    
    if [ -d "${run_output_dir}" ]; then
        print_info "Removing previous output directory: ${run_output_dir}"
        rm -rf "${run_output_dir}"
    fi
    
    mkdir -p "${run_output_dir}/checkpoints"
    mkdir -p "${run_output_dir}/logs"
    
    local training_log="${run_output_dir}/training.log"
    
    if ! python -m src.${MODULE_NAME}.api.train \
            --config-path="${PROJECT_ROOT}/configs/train/${MODULE_NAME}" \
            --config-name="${config_name}" \
            hydra.run.dir="${run_output_dir}" \
            trainer.default_root_dir="${run_output_dir}" \
            callbacks.checkpoint.dirpath="${run_output_dir}/checkpoints" \
            hydra.job.chdir=false \
            > "${training_log}" 2>&1; then
        
        print_error "Training failed for ${config_name}. Check log: ${training_log}"
        tail -n 10 "${training_log}" >&2 || true
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_success "Training completed for ${config_name} in ${duration} seconds"
    
    # Archive checkpoint
    local source_ckpt_dir="${run_output_dir}/checkpoints"
    local best_ckpt
    
    if ! best_ckpt=$(find_best_checkpoint "${source_ckpt_dir}" "${config_name}"); then
        print_warning "No checkpoint file found for ${config_name}. Skipping archival."
        return 0
    fi
    
    local dest_dir="${CHECKPOINT_ROOT}/${config_name}"
    local dest_path="${dest_dir}/best_model.ckpt"
    local backup_path="${dest_dir}/best_model_backup_${TIMESTAMP}.ckpt"
    
    print_info "Archiving checkpoint: ${best_ckpt} -> ${dest_path}"
    
    mkdir -p "${dest_dir}"
    
    if [ -f "${dest_path}" ]; then
        print_info "Backing up existing checkpoint to ${backup_path}"
        mv "${dest_path}" "${backup_path}"
    fi
    
    cp "${best_ckpt}" "${dest_path}"
    
    local symlink_path="${dest_dir}/experiment_${TIMESTAMP}"
    ln -sf "${run_output_dir}" "${symlink_path}"
    
    print_success "Checkpoint for ${config_name} successfully archived."
    
    local ckpt_info="${dest_dir}/checkpoint_info.txt"
    {
        echo "Checkpoint: ${config_name}"
        echo "Source: ${best_ckpt}"
        echo "Archived: $(date)"
        echo "Training duration: ${duration} seconds"
        echo "Experiment directory: ${run_output_dir}"
    } > "${ckpt_info}"
}

# Main execution
mkdir -p "${CHECKPOINT_ROOT}"
mkdir -p "${HYDRA_OUTPUT_ROOT}"
mkdir -p "${LOG_DIR}"

echo "Batch training started at $(date)" > "${BATCH_LOG_FILE}"

cd "${PROJECT_ROOT}"

print_info "Starting batch training for ${MODULE_NAME} models..."
print_info "Batch log file: ${BATCH_LOG_FILE}"

total_models=${#MODELS_TO_TRAIN[@]}
completed_models=0
failed_models=0
overall_start_time=$(date +%s)

for model_config in "${MODELS_TO_TRAIN[@]}"; do
    echo
    print_info "======================================================================"
    print_info "Processing model: ${model_config} ($(($completed_models + 1))/${total_models})"
    print_info "======================================================================"
    
    if train_and_archive_model "${model_config}"; then
        ((completed_models++))
    else
        ((failed_models++))
        print_error "Failed to train model: ${model_config}"
    fi
done

overall_end_time=$(date +%s)
overall_duration=$((overall_end_time - overall_start_time))

print_info "======================================================================"
print_success "Batch training completed!"
print_info "Total models: ${total_models}"
print_info "Completed: ${completed_models}"
print_info "Failed: ${failed_models}"
print_info "Total duration: ${overall_duration} seconds"
print_info "Batch log file: ${BATCH_LOG_FILE}"
print_info "======================================================================"

if [ ${failed_models} -gt 0 ]; then
    exit 1
fi