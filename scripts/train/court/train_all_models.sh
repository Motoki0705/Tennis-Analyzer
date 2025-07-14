#!/bin/bash

# Court Detection Model Training Script
# Trains all court detection models with generic LitModule architecture

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/court"
LOG_DIR="${PROJECT_ROOT}/logs/court"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_environment() {
    print_status "Checking environment prerequisites..."
    
    # Check if we're in the correct directory
    if [[ ! -f "${PROJECT_ROOT}/src/court/api/train.py" ]]; then
        print_error "Court training script not found. Please run from project root."
        exit 1
    fi
    
    # Check if Python environment has required packages
    python -c "import pytorch_lightning, hydra" 2>/dev/null || {
        print_error "Required Python packages not found. Please install requirements.txt"
        exit 1
    }
    
    # Create directories if they don't exist
    mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
    
    print_success "Environment check completed"
}

# Function to train a single model
train_model() {
    local config_name="$1"
    local model_name="$2"
    
    print_status "Starting training for ${model_name} model..."
    print_status "Configuration: ${config_name}"
    
    local log_file="${LOG_DIR}/${model_name}_$(date +%Y%m%d_%H%M%S).log"
    local start_time=$(date +%s)
    
    # Change to project root for training
    cd "${PROJECT_ROOT}"
    
    # Run training with error handling
    if python -m src.court.api.train --config-name="${config_name}" \
        trainer.default_root_dir="${CHECKPOINT_DIR}/${model_name}" \
        hydra.job.chdir=True \
        2>&1 | tee "${log_file}"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        print_success "${model_name} training completed successfully"
        print_status "Training time: ${hours}h ${minutes}m ${seconds}s"
        print_status "Log file: ${log_file}"
        
        # Archive the checkpoint
        local checkpoint_archive="${CHECKPOINT_DIR}/${model_name}_$(date +%Y%m%d_%H%M%S)"
        if [[ -d "${CHECKPOINT_DIR}/${model_name}" ]]; then
            mv "${CHECKPOINT_DIR}/${model_name}" "${checkpoint_archive}"
            print_status "Checkpoint archived to: ${checkpoint_archive}"
        fi
        
        return 0
    else
        print_error "${model_name} training failed"
        print_error "Check log file: ${log_file}"
        return 1
    fi
}

# Main training function
main() {
    print_status "Court Detection Model Training Pipeline"
    print_status "========================================"
    
    # Check environment
    check_environment
    
    # Array of models to train [config_name, display_name]
    declare -a models=(
        "lite_tracknet_generic:LiteTrackNet"
        "swin_unet_generic:SwinUNet"
        "vit_unet_generic:VitUNet"
        "fpn_generic:FPN"
    )
    
    local total_models=${#models[@]}
    local successful_models=0
    local failed_models=0
    
    print_status "Training ${total_models} court detection models..."
    print_status ""
    
    # Train each model
    for model_config in "${models[@]}"; do
        IFS=':' read -r config_name model_name <<< "$model_config"
        
        print_status "Model $(($successful_models + $failed_models + 1))/${total_models}: ${model_name}"
        print_status "----------------------------------------"
        
        if train_model "${config_name}" "${model_name}"; then
            ((successful_models++))
        else
            ((failed_models++))
            
            # Ask if user wants to continue
            if [[ $failed_models -lt $total_models ]]; then
                read -p "Continue with remaining models? (y/n): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_warning "Training pipeline stopped by user"
                    break
                fi
            fi
        fi
        
        print_status ""
    done
    
    # Summary
    print_status "Training Pipeline Summary"
    print_status "========================"
    print_success "Successfully trained: ${successful_models}/${total_models} models"
    
    if [[ $failed_models -gt 0 ]]; then
        print_error "Failed models: ${failed_models}/${total_models}"
    fi
    
    print_status "Checkpoints directory: ${CHECKPOINT_DIR}"
    print_status "Logs directory: ${LOG_DIR}"
    
    if [[ $successful_models -eq $total_models ]]; then
        print_success "All court detection models trained successfully!"
        exit 0
    else
        print_error "Some models failed to train. Check logs for details."
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Court Detection Model Training Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --check       Check environment only"
        echo ""
        echo "Models trained:"
        echo "  - LiteTrackNet (lite_tracknet_generic)"
        echo "  - SwinUNet (swin_unet_generic)"
        echo "  - VitUNet (vit_unet_generic)"
        echo "  - FPN (fpn_generic)"
        echo ""
        echo "Output:"
        echo "  Checkpoints: checkpoints/court/"
        echo "  Logs: logs/court/"
        exit 0
        ;;
    --check)
        check_environment
        print_success "Environment check passed"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        print_error "Use --help for usage information"
        exit 1
        ;;
esac