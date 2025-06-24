#!/bin/bash

# =============================================================================
# Tennis Ball Detection Inference Script
# =============================================================================
# 🎾 テニスボール検出推論実行スクリプト
#
# このスクリプトは、テニス動画からのボール検出を簡単に実行するための
# コマンドラインインターフェースです。基本実行、高性能実行、バッチ処理など
# 様々な用途に対応した実行パターンを提供します。
#
# Features:
# - 🚀 ワンコマンド実行
# - ⚙️ 用途別実行パターン
# - 📊 統計レポート出力
# - 🛠️ カスタム設定対応
#
# Usage:
#   bash scripts/infer/ball_detection.sh [mode] [options...]
#
# Modes:
#   basic         - 基本実行（デフォルト設定）
#   performance   - 高性能実行（GPU最適化）
#   batch         - バッチ処理（複数動画）
#   custom        - カスタム実行（詳細設定）
#
# Examples:
#   # 基本実行
#   bash scripts/infer/ball_detection.sh basic input.mp4 output.mp4 model.ckpt
#
#   # 高性能実行
#   bash scripts/infer/ball_detection.sh performance input.mp4 output.mp4 model.pth
#
#   # バッチ処理
#   bash scripts/infer/ball_detection.sh batch videos/ results/ model.ckpt
#
# =============================================================================

set -e  # エラー時に停止

# スクリプトディレクトリの取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# デフォルト設定
DEFAULT_MODEL_TYPE="auto"
DEFAULT_DEVICE="auto"
DEFAULT_LOG_LEVEL="INFO"

# カラー出力設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ロゴ表示
print_logo() {
    echo -e "${GREEN}"
    echo "🎾 ======================================"
    echo "   Tennis Ball Detection System"
    echo "   高性能テニスボール検出・解析システム"
    echo "======================================${NC}"
    echo
}

# ヘルプ表示
print_help() {
    cat << EOF
Usage: bash scripts/infer/ball_detection.sh [mode] [options...]

MODES:
  basic         基本実行（デフォルト設定）
  performance   高性能実行（GPU最適化、並列処理）
  batch         バッチ処理（複数動画の一括処理）
  custom        カスタム実行（詳細設定指定）

BASIC MODE:
  bash scripts/infer/ball_detection.sh basic <video> <output> <model> [options]
  
  Arguments:
    video         入力動画ファイルパス
    output        出力動画ファイルパス
    model         モデルファイルパス (.ckpt または .pth)
  
  Options:
    --model-type  モデルタイプ (lite_tracknet|wasb_sbdt|auto)
    --device      実行デバイス (auto|cpu|cuda)
    --log-level   ログレベル (DEBUG|INFO|WARNING|ERROR)

PERFORMANCE MODE:
  bash scripts/infer/ball_detection.sh performance <video> <output> <model> [options]
  
  Options (+ basic options):
    --batch-size    バッチサイズ (default: 16)
    --num-workers   ワーカー数 (default: 8)
    --async         非同期処理モード

BATCH MODE:
  bash scripts/infer/ball_detection.sh batch <input_dir> <output_dir> <model> [options]
  
  Arguments:
    input_dir     入力動画ディレクトリ
    output_dir    出力ディレクトリ
    model         モデルファイルパス
  
  Options (+ basic options):
    --parallel-jobs   並列ジョブ数 (default: 2)
    --report-path     レポート出力パス
    --continue        エラー時続行
    --overwrite       既存ファイル上書き

CUSTOM MODE:
  bash scripts/infer/ball_detection.sh custom [all options]
  
  すべてのオプションを直接指定可能

EXAMPLES:
  # 基本実行（LiteTrackNet）
  bash scripts/infer/ball_detection.sh basic \\
    tennis_match.mp4 result.mp4 checkpoints/lite_tracknet.ckpt

  # 高性能実行（WASB-SBDT）
  bash scripts/infer/ball_detection.sh performance \\
    long_match.mp4 result.mp4 models/wasb_sbdt.pth \\
    --batch-size 16 --async

  # バッチ処理
  bash scripts/infer/ball_detection.sh batch \\
    tennis_videos/ results/ model.ckpt \\
    --parallel-jobs 4 --report-path report.json

  # カスタム実行
  bash scripts/infer/ball_detection.sh custom \\
    --video input.mp4 --output output.mp4 --model-path model.ckpt \\
    --config high_performance --ball-radius 12 --enable-prediction

VISUALIZATION OPTIONS:
  --ball-radius         ボール描画半径 (default: 8)
  --trajectory-length   軌跡表示フレーム数 (default: 20)
  --enable-smoothing    位置スムージング有効化
  --enable-prediction   位置予測表示有効化
  --confidence-threshold 信頼度閾値 (default: 0.5)

SYSTEM OPTIONS:
  --help, -h            このヘルプを表示
  --version             バージョン情報表示
  --check-gpu           GPU利用可能性チェック

EOF
}

# エラーメッセージ表示
print_error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
}

# 成功メッセージ表示
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 情報メッセージ表示
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 警告メッセージ表示
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# GPU利用可能性チェック
check_gpu() {
    print_info "GPU利用可能性をチェック中..."
    
    cd "${PROJECT_ROOT}"
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('GPU not available - will use CPU')
"
}

# 依存関係チェック
check_dependencies() {
    print_info "依存関係をチェック中..."
    
    cd "${PROJECT_ROOT}"
    
    # Python環境チェック
    if ! command -v python3 &> /dev/null; then
        print_error "Python3が見つかりません"
        exit 1
    fi
    
    # 必要なモジュールチェック
    python3 -c "
import sys
missing_modules = []
required_modules = ['torch', 'cv2', 'numpy', 'tqdm']

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print(f'Missing modules: {missing_modules}')
    print('Please install requirements: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('All required modules are available')
" || exit 1
}

# 基本実行
run_basic() {
    if [ $# -lt 3 ]; then
        print_error "基本実行には video, output, model の3つの引数が必要です"
        echo "Usage: bash scripts/infer/ball_detection.sh basic <video> <output> <model> [options]"
        exit 1
    fi
    
    local video="$1"
    local output="$2"
    local model="$3"
    shift 3
    
    print_info "基本実行モード"
    print_info "入力動画: ${video}"
    print_info "出力動画: ${output}"
    print_info "モデル: ${model}"
    
    cd "${PROJECT_ROOT}"
    python3 -m src.predictor.api.inference \
        --video "${video}" \
        --output "${output}" \
        --model_path "${model}" \
        --model_type "${DEFAULT_MODEL_TYPE}" \
        --device "${DEFAULT_DEVICE}" \
        --config memory_efficient \
        --log_level "${DEFAULT_LOG_LEVEL}" \
        "$@"
}

# 高性能実行
run_performance() {
    if [ $# -lt 3 ]; then
        print_error "高性能実行には video, output, model の3つの引数が必要です"
        echo "Usage: bash scripts/infer/ball_detection.sh performance <video> <output> <model> [options]"
        exit 1
    fi
    
    local video="$1"
    local output="$2"
    local model="$3"
    shift 3
    
    # デフォルト高性能設定
    local batch_size=16
    local num_workers=8
    local async_flag=""
    
    # オプション解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            --batch-size)
                batch_size="$2"
                shift 2
                ;;
            --num-workers)
                num_workers="$2"
                shift 2
                ;;
            --async)
                async_flag="--async_processing"
                shift
                ;;
            *)
                # その他のオプションはそのまま通す
                break
                ;;
        esac
    done
    
    print_info "高性能実行モード"
    print_info "入力動画: ${video}"
    print_info "出力動画: ${output}"
    print_info "モデル: ${model}"
    print_info "バッチサイズ: ${batch_size}"
    print_info "ワーカー数: ${num_workers}"
    
    cd "${PROJECT_ROOT}"
    python3 -m src.predictor.api.inference \
        --video "${video}" \
        --output "${output}" \
        --model_path "${model}" \
        --model_type "${DEFAULT_MODEL_TYPE}" \
        --device "${DEFAULT_DEVICE}" \
        --config high_performance \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        ${async_flag} \
        --log_level "${DEFAULT_LOG_LEVEL}" \
        "$@"
}

# バッチ処理実行
run_batch() {
    if [ $# -lt 3 ]; then
        print_error "バッチ処理には input_dir, output_dir, model の3つの引数が必要です"
        echo "Usage: bash scripts/infer/ball_detection.sh batch <input_dir> <output_dir> <model> [options]"
        exit 1
    fi
    
    local input_dir="$1"
    local output_dir="$2"
    local model="$3"
    shift 3
    
    # デフォルトバッチ設定
    local parallel_jobs=2
    local report_path=""
    local continue_flag=""
    local overwrite_flag=""
    
    # オプション解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel-jobs)
                parallel_jobs="$2"
                shift 2
                ;;
            --report-path)
                report_path="$2"
                shift 2
                ;;
            --continue)
                continue_flag="--continue_on_error"
                shift
                ;;
            --overwrite)
                overwrite_flag="--overwrite"
                shift
                ;;
            *)
                # その他のオプションはそのまま通す
                break
                ;;
        esac
    done
    
    print_info "バッチ処理モード"
    print_info "入力ディレクトリ: ${input_dir}"
    print_info "出力ディレクトリ: ${output_dir}"
    print_info "モデル: ${model}"
    print_info "並列ジョブ数: ${parallel_jobs}"
    
    cd "${PROJECT_ROOT}"
    
    local cmd="python3 -m src.predictor.api.batch_process \
        --input_dir \"${input_dir}\" \
        --output_dir \"${output_dir}\" \
        --model_path \"${model}\" \
        --model_type \"${DEFAULT_MODEL_TYPE}\" \
        --device \"${DEFAULT_DEVICE}\" \
        --parallel_jobs \"${parallel_jobs}\" \
        ${continue_flag} \
        ${overwrite_flag} \
        --log_level \"${DEFAULT_LOG_LEVEL}\""
    
    if [ -n "${report_path}" ]; then
        cmd="${cmd} --report_path \"${report_path}\""
    fi
    
    cmd="${cmd} $*"
    
    eval "${cmd}"
}

# カスタム実行
run_custom() {
    print_info "カスタム実行モード"
    print_info "全オプションを直接指定して実行します"
    
    cd "${PROJECT_ROOT}"
    python3 -m src.predictor.api.inference "$@"
}

# メイン処理
main() {
    # 引数が空の場合はヘルプ表示
    if [ $# -eq 0 ]; then
        print_logo
        print_help
        exit 0
    fi
    
    # オプション処理
    case "$1" in
        --help|-h)
            print_logo
            print_help
            exit 0
            ;;
        --version)
            echo "Tennis Ball Detection System v1.0.0"
            echo "Powered by PyTorch, OpenCV, and WASB-SBDT"
            exit 0
            ;;
        --check-gpu)
            print_logo
            check_gpu
            exit 0
            ;;
    esac
    
    # ロゴ表示
    print_logo
    
    # 依存関係チェック
    check_dependencies
    
    # モード判定と実行
    local mode="$1"
    shift
    
    case "${mode}" in
        basic)
            run_basic "$@"
            ;;
        performance)
            run_performance "$@"
            ;;
        batch)
            run_batch "$@"
            ;;
        custom)
            run_custom "$@"
            ;;
        *)
            print_error "Unknown mode: ${mode}"
            echo "Available modes: basic, performance, batch, custom"
            echo "Use --help for detailed usage information"
            exit 1
            ;;
    esac
    
    print_success "処理が完了しました！"
}

# スクリプト実行
main "$@" 