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
    local source_ckpt_dir="${run_output_dir}/checkpoints"
    
    # Clean up previous run directory to ensure a fresh start
    if [ -d "${run_output_dir}" ]; then
        print_info "Removing previous output directory: ${run_output_dir}"
        rm -rf "${run_output_dir}"
    fi

    # Run the training script using python -m for robust path handling
    if ! python -m src.ball.api.train \
        --config-path="${PROJECT_ROOT}/configs/train/ball" \
        --config-name="${config_name}" \
        hydra.run.dir="${run_output_dir}" \
        trainer.default_root_dir="${run_output_dir}" \
        callbacks.checkpoint.dirpath="${source_ckpt_dir}"; then
        print_error "Training failed for ${config_name}. Aborting."
        exit 1
    fi

    print_success "Training completed for ${config_name}."

    # --- Checkpoint Archiving ---    
    # Find the best checkpoint file. Using -print -quit to get only the first match.
    local best_ckpt
    best_ckpt=$(find "${source_ckpt_dir}" -name "*.ckpt" -print -quit)

    if [ -z "${best_ckpt}" ]; then
        print_warning "No checkpoint file found for ${config_name} in ${source_ckpt_dir}. Skipping archival."
        return
    fi 

    local dest_dir="${CHECKPOINT_ROOT}/${config_name}"
    local dest_path="${dest_dir}/best_model.ckpt"

    print_info "Archiving checkpoint: ${best_ckpt} -> ${dest_path}"
    
    # Create destination directory and move the checkpoint
    mkdir -p "${dest_dir}"#!/bin/bash

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
LOG_DIR="${PROJECT_ROOT}/logs/training"

# Array of model configurations to train
# Add new model config names here to include them in the training cycle.
MODELS_TO_TRAIN=(
    "lite_tracknet_generic"
    "conv3d_tsm_fpn"
    "video_swin_generic"
)

# Training configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG_FILE="${LOG_DIR}/batch_training_${TIMESTAMP}.log"

# --- Functions ---

# Function to print colored output
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

# Function to log environment information
log_environment() {
    local run_output_dir=$1
    local config_name=$2
    
    print_info "Logging environment information for ${config_name}"
    
    # Create experiment info file
    local experiment_info="${run_output_dir}/experiment_info.txt"
    {
        echo "Experiment: ${config_name}"
        echo "Start time: $(date)"
        echo "Host: $(hostname)"
        echo "User: $(whoami)"
        echo "Working directory: $(pwd)"
        echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'Not a git repository')"
        echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
        echo "Python version: $(python --version)"
        echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
    } > "${experiment_info}"
    
    # Save requirements
    if command -v pip &> /dev/null; then
        pip freeze > "${run_output_dir}/requirements.txt" 2>/dev/null || true
    fi
    
    # Copy config file for reproducibility
    local config_file="${PROJECT_ROOT}/configs/train/ball/${config_name}.yaml"
    if [ -f "${config_file}" ]; then
        cp "${config_file}" "${run_output_dir}/"
    fi
}

# Function to validate checkpoint
validate_checkpoint() {
    local ckpt_path=$1
    local config_name=$2
    
    print_info "Validating checkpoint: ${ckpt_path}"
    
    # Check if checkpoint file exists and is not empty
    if [ ! -f "${ckpt_path}" ]; then
        print_error "Checkpoint file does not exist: ${ckpt_path}"
        return 1
    fi
    
    if [ ! -s "${ckpt_path}" ]; then
        print_error "Checkpoint file is empty: ${ckpt_path}"
        return 1
    fi
    
    # Optional: Add more sophisticated validation using Python
    # python -c "import torch; torch.load('${ckpt_path}', map_location='cpu'); print('Checkpoint is valid')" 2>/dev/null || {
    #     print_error "Checkpoint file is corrupted: ${ckpt_path}"
    #     return 1
    # }
    
    print_success "Checkpoint validation passed for ${config_name}"
    return 0
}

# Function to find best checkpoint with improved logic
find_best_checkpoint() {
    local source_ckpt_dir=$1
    local config_name=$2
    
    print_info "Searching for best checkpoint in ${source_ckpt_dir}"
    
    # First, try to find a checkpoint with 'best' in the name
    local best_ckpt
    best_ckpt=$(find "${source_ckpt_dir}" -name "*best*.ckpt" -type f | head -n 1)
    
    if [ -n "${best_ckpt}" ]; then
        print_info "Found best checkpoint: ${best_ckpt}"
        echo "${best_ckpt}"
        return 0
    fi
    
    # If no 'best' checkpoint, try to find 'last' checkpoint
    best_ckpt=$(find "${source_ckpt_dir}" -name "*last*.ckpt" -type f | head -n 1)
    
    if [ -n "${best_ckpt}" ]; then
        print_info "Found last checkpoint: ${best_ckpt}"
        echo "${best_ckpt}"
        return 0
    fi
    
    # If neither 'best' nor 'last', get the most recent checkpoint
    best_ckpt=$(find "${source_ckpt_dir}" -name "*.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
    
    if [ -n "${best_ckpt}" ]; then
        print_info "Found most recent checkpoint: ${best_ckpt}"
        echo "${best_ckpt}"
        return 0
    fi
    
    print_warning "No checkpoint found in ${source_ckpt_dir}"
    return 1
}

# Function to train a single model and archive its checkpoint
train_and_archive_model() {
    local config_name=$1
    print_info "Starting training for model: ${config_name}"
    
    local start_time=$(date +%s)
    
    # Set a deterministic output path for this run with timestamp
    local run_output_dir="${HYDRA_OUTPUT_ROOT}/${config_name}/${TIMESTAMP}"
    
    # Clean up previous run directory to ensure a fresh start
    if [ -d "${run_output_dir}" ]; then
        print_info "Removing previous output directory: ${run_output_dir}"
        rm -rf "${run_output_dir}"
    fi
    
    # Create necessary directories
    mkdir -p "${run_output_dir}/checkpoints"
    mkdir -p "${run_output_dir}/logs"
    
    # Log environment information
    log_environment "${run_output_dir}" "${config_name}"
    
    # Run the training script using python -m for robust path handling
    print_info "Executing training command for ${config_name}"
    local training_log="${run_output_dir}/training.log"
    
    if ! python -m src.ball.api.train \
        --config-path="${PROJECT_ROOT}/configs/train/ball" \
        --config-name="${config_name}" \
        hydra.run.dir="${run_output_dir}" \
        trainer.default_root_dir="${run_output_dir}" \
        callbacks.checkpoint.dirpath="${run_output_dir}/checkpoints" \
        hydra.job.chdir=false \
        > "${training_log}" 2>&1; then
        print_error "Training failed for ${config_name}. Check log: ${training_log}"
        
        # Show last few lines of the log for debugging
        print_error "Last 10 lines of training log:"
        tail -n 10 "${training_log}" >&2 || true
        
        exit 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_success "Training completed for ${config_name} in ${duration} seconds"
    
    # --- Checkpoint Archiving ---
    local source_ckpt_dir="${run_output_dir}/checkpoints"
    
    # Find the best checkpoint file
    local best_ckpt
    if ! best_ckpt=$(find_best_checkpoint "${source_ckpt_dir}" "${config_name}"); then
        print_warning "No checkpoint file found for ${config_name} in ${source_ckpt_dir}. Skipping archival."
        return 0
    fi
    
    # Validate checkpoint before archiving
    if ! validate_checkpoint "${best_ckpt}" "${config_name}"; then
        print_error "Checkpoint validation failed for ${config_name}. Skipping archival."
        return 1
    fi
    
    local dest_dir="${CHECKPOINT_ROOT}/${config_name}"
    local dest_path="${dest_dir}/best_model.ckpt"
    local backup_path="${dest_dir}/best_model_backup_${TIMESTAMP}.ckpt"
    
    print_info "Archiving checkpoint: ${best_ckpt} -> ${dest_path}"
    
    # Create destination directory
    mkdir -p "${dest_dir}"
    
    # Backup existing checkpoint if it exists
    if [ -f "${dest_path}" ]; then
        print_info "Backing up existing checkpoint to ${backup_path}"
        mv "${dest_path}" "${backup_path}"
    fi
    
    # Copy (don't move) the checkpoint to preserve the original
    cp "${best_ckpt}" "${dest_path}"
    
    # Create a symlink to the original experiment
    local symlink_path="${dest_dir}/experiment_${TIMESTAMP}"
    ln -sf "${run_output_dir}" "${symlink_path}"
    
    print_success "Checkpoint for ${config_name} successfully archived."
    
    # Log checkpoint info
    local ckpt_info="${dest_dir}/checkpoint_info.txt"
    {
        echo "Checkpoint: ${config_name}"
        echo "Source: ${best_ckpt}"
        echo "Archived: $(date)"
        echo "Training duration: ${duration} seconds"
        echo "Experiment directory: ${run_output_dir}"
    } > "${ckpt_info}"
}

# Function to cleanup old outputs (optional)
cleanup_old_outputs() {
    local days_to_keep=${1:-30}
    print_info "Cleaning up outputs older than ${days_to_keep} days"
    
    find "${HYDRA_OUTPUT_ROOT}" -type d -name "20*" -mtime +${days_to_keep} -exec rm -rf {} + 2>/dev/null || true
    find "${LOG_DIR}" -name "*.log" -mtime +${days_to_keep} -delete 2>/dev/null || true
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not available in PATH"
        exit 1
    fi
    
    # Check if the training module exists
    if ! python -c "import src.ball.api.train" 2>/dev/null; then
        print_error "Training module src.ball.api.train not found"
        exit 1
    fi
    
    # Check if config directory exists
    if [ ! -d "${PROJECT_ROOT}/configs/train/ball" ]; then
        print_error "Config directory not found: ${PROJECT_ROOT}/configs/train/ball"
        exit 1
    fi
    
    # Check if all config files exist
    for config_name in "${MODELS_TO_TRAIN[@]}"; do
        local config_file="${PROJECT_ROOT}/configs/train/ball/${config_name}.yaml"
        if [ ! -f "${config_file}" ]; then
            print_error "Config file not found: ${config_file}"
            exit 1
        fi
    done
    
    print_success "All prerequisites check passed"
}

# Function to handle script interruption
cleanup_on_exit() {
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        print_error "Script interrupted with exit code ${exit_code}"
    fi
    print_info "Cleaning up..."
    # Add any cleanup tasks here
    exit ${exit_code}
}

# --- Main Execution ---

# Set trap for cleanup on exit
trap cleanup_on_exit EXIT

# Create necessary directories
mkdir -p "${CHECKPOINT_ROOT}"
mkdir -p "${HYDRA_OUTPUT_ROOT}"
mkdir -p "${LOG_DIR}"

# Initialize log file
echo "Batch training started at $(date)" > "${BATCH_LOG_FILE}"

# Change to project root
cd "${PROJECT_ROOT}"

# Check prerequisites
check_prerequisites

# Optional: Cleanup old outputs
# cleanup_old_outputs 30

print_info "Starting batch training for all ball models..."
print_info "Batch log file: ${BATCH_LOG_FILE}"

# Track overall statistics
local total_models=${#MODELS_TO_TRAIN[@]}
local completed_models=0
local failed_models=0
local overall_start_time=$(date +%s)

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
        # Continue with next model instead of exiting
    fi
    
    # Optional: Add a small delay between trainings if needed
    # sleep 5 
done

local overall_end_time=$(date +%s)
local overall_duration=$((overall_end_time - overall_start_time))

print_info "======================================================================"
print_success "Batch training completed!"
print_info "Total models: ${total_models}"
print_info "Completed: ${completed_models}"
print_info "Failed: ${failed_models}"
print_info "Total duration: ${overall_duration} seconds"
print_info "Batch log file: ${BATCH_LOG_FILE}"
print_info "======================================================================"

# Exit with error if any model failed
if [ ${failed_models} -gt 0 ]; then
    exit 1
fi
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
