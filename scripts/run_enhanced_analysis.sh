#!/bin/bash

# Enhanced Ball Analysis Script
# 3段階フィルタリング統合分析スクリプト

set -e  # Exit on any error

echo "🎬 Starting Enhanced Ball Analysis"
echo "=================================="

# Default parameters
VIDEO_PATH=""
BALL_TRACKER_CONFIG="third_party/WASB-SBDT/src/configs/model/wasb.yaml"
BALL_TRACKER_WEIGHTS=""
LOCAL_CLASSIFIER=""
OUTPUT_DIR=""
STAGE1_THRESHOLD=0.5
STAGE2_THRESHOLD=0.5
STAGE3_MAX_DISTANCE=50.0
DEVICE="cuda"
NO_VISUALIZE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --ball_tracker_config)
            BALL_TRACKER_CONFIG="$2"
            shift 2
            ;;
        --ball_tracker_weights)
            BALL_TRACKER_WEIGHTS="$2"
            shift 2
            ;;
        --local_classifier)
            LOCAL_CLASSIFIER="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --stage1_threshold)
            STAGE1_THRESHOLD="$2"
            shift 2
            ;;
        --stage2_threshold)
            STAGE2_THRESHOLD="$2"
            shift 2
            ;;
        --stage3_max_distance)
            STAGE3_MAX_DISTANCE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no_visualize)
            NO_VISUALIZE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --video VIDEO_PATH --local_classifier MODEL_PATH [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --video                 Path to input video file"
            echo "  --local_classifier      Path to trained local classifier model"
            echo ""
            echo "Optional arguments:"
            echo "  --ball_tracker_config   ball_tracker config file (default: $BALL_TRACKER_CONFIG)"
            echo "  --ball_tracker_weights  ball_tracker weights file (required if using ball_tracker)"
            echo "  --output_dir            Output directory (default: auto-generated)"
            echo "  --stage1_threshold      Stage 1 confidence threshold (default: $STAGE1_THRESHOLD)"
            echo "  --stage2_threshold      Stage 2 confidence threshold (default: $STAGE2_THRESHOLD)"
            echo "  --stage3_max_distance   Stage 3 max distance (default: $STAGE3_MAX_DISTANCE)"
            echo "  --device                Device: cuda|cpu (default: $DEVICE)"
            echo "  --no_visualize          Disable video visualization"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic analysis with local classifier only"
            echo "  $0 --video video.mp4 --local_classifier checkpoints/best_model.pth"
            echo ""
            echo "  # Full 3-stage analysis"
            echo "  $0 --video video.mp4 \\"
            echo "     --ball_tracker_weights ball_tracker.pth \\"
            echo "     --local_classifier local_classifier.pth \\"
            echo "     --output_dir ./analysis_results"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$VIDEO_PATH" ]; then
    echo "❌ Error: Video path is required (--video)"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$LOCAL_CLASSIFIER" ]; then
    echo "❌ Error: Local classifier model is required (--local_classifier)"
    echo "Use --help for usage information"
    exit 1
fi

# Validate input files
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

if [ ! -f "$LOCAL_CLASSIFIER" ]; then
    echo "❌ Error: Local classifier model not found: $LOCAL_CLASSIFIER"
    exit 1
fi

if [ ! -f "$BALL_TRACKER_CONFIG" ]; then
    echo "❌ Error: ball_tracker config not found: $BALL_TRACKER_CONFIG"
    exit 1
fi

if [ -n "$BALL_TRACKER_WEIGHTS" ] && [ ! -f "$BALL_TRACKER_WEIGHTS" ]; then
    echo "❌ Error: ball_tracker weights not found: $BALL_TRACKER_WEIGHTS"
    exit 1
fi

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
    OUTPUT_DIR="./analysis_results/${VIDEO_NAME}_enhanced"
fi

# Print configuration
echo "📋 Configuration:"
echo "   Video: $VIDEO_PATH"
echo "   ball_tracker config: $BALL_TRACKER_CONFIG"
echo "   ball_tracker weights: ${BALL_TRACKER_WEIGHTS:-'Not provided (Stage 1 disabled)'}"
echo "   Local classifier: $LOCAL_CLASSIFIER"
echo "   Output directory: $OUTPUT_DIR"
echo "   Stage 1 threshold: $STAGE1_THRESHOLD"
echo "   Stage 2 threshold: $STAGE2_THRESHOLD"
echo "   Stage 3 max distance: $STAGE3_MAX_DISTANCE"
echo "   Device: $DEVICE"
echo "   Visualization: $([ "$NO_VISUALIZE" = true ] && echo "Disabled" || echo "Enabled")"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python -m src.ball.enhanced_analysis_tool"
CMD="$CMD --video \"$VIDEO_PATH\""
CMD="$CMD --ball_tracker_config \"$BALL_TRACKER_CONFIG\""
CMD="$CMD --local_classifier \"$LOCAL_CLASSIFIER\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --stage1_threshold $STAGE1_THRESHOLD"
CMD="$CMD --stage2_threshold $STAGE2_THRESHOLD"
CMD="$CMD --stage3_max_distance $STAGE3_MAX_DISTANCE"
CMD="$CMD --device $DEVICE"

if [ -n "$BALL_TRACKER_WEIGHTS" ]; then
    CMD="$CMD --ball_tracker_weights \"$BALL_TRACKER_WEIGHTS\""
fi

if [ "$NO_VISUALIZE" = true ]; then
    CMD="$CMD --no_visualize"
fi

# Run analysis
echo "🎯 Starting enhanced analysis..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "✅ Analysis completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo "📊 Summary report: $OUTPUT_DIR/summary_report.md"
echo "🎬 Analysis video: $OUTPUT_DIR/${VIDEO_NAME}_enhanced_analysis.mp4"
echo "" 