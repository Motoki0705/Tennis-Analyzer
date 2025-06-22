#!/bin/bash

# Local Ball Classifier Training Script
# ローカル分類器学習スクリプト

set -e  # Exit on any error

echo "🚀 Starting Local Ball Classifier Training"
echo "============================================"

# Default parameters
ANNOTATION_FILE="datasets/ball/coco_annotations_ball_pose_court.json"
IMAGES_DIR="datasets/ball/images"
OUTPUT_DIR="./local_classifier_checkpoints"
MODEL_TYPE="standard"
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=0.001
PATCH_SIZE=16
POSITION_NOISE=2
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --annotation_file)
            ANNOTATION_FILE="$2"
            shift 2
            ;;
        --images_dir)
            IMAGES_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --position_noise)
            POSITION_NOISE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --annotation_file   Path to annotation file (default: $ANNOTATION_FILE)"
            echo "  --images_dir        Path to images directory (default: $IMAGES_DIR)"
            echo "  --output_dir        Output directory (default: $OUTPUT_DIR)"
            echo "  --model_type        Model type: standard|efficient (default: $MODEL_TYPE)"
            echo "  --epochs            Number of epochs (default: $EPOCHS)"
            echo "  --batch_size        Batch size (default: $BATCH_SIZE)"
            echo "  --learning_rate     Learning rate (default: $LEARNING_RATE)"
echo "  --position_noise    Position noise ±pixels (default: $POSITION_NOISE)"
echo "  --device            Device: cuda|cpu (default: $DEVICE)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model_type efficient --epochs 30 --batch_size 32"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate input files
if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "❌ Error: Annotation file not found: $ANNOTATION_FILE"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

# Print configuration
echo "📋 Configuration:"
echo "   Annotation file: $ANNOTATION_FILE"
echo "   Images directory: $IMAGES_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo "   Model type: $MODEL_TYPE"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LEARNING_RATE"
echo "   Position noise: ±${POSITION_NOISE}px"
echo "   Device: $DEVICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "🏋️ Starting training..."
python -m src.ball.local_classifier.train \
    --annotation_file "$ANNOTATION_FILE" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --patch_size "$PATCH_SIZE" \
    --position_noise "$POSITION_NOISE" \
    --device "$DEVICE"

echo ""
echo "✅ Training completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo "🏆 Best model: $OUTPUT_DIR/best_model.pth"
echo "" 