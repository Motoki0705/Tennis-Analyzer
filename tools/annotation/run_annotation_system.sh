#!/bin/bash

# Tennis Event Annotation System - 統合実行スクリプト
# このスクリプトでアノテーションシステム全体のワークフローを実行できます

set -e  # エラー時に停止

# 色付きログ出力用の関数
log_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# 設定
WORKSPACE_DIR="./datasets/annotation_workspace"
INPUT_VIDEO="./datasets/inputs/game1.mp4"
BALL_CKPT="checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
EVENT_CKPT="checkpoints/event/transformer_v2/epoch=18-step=532.ckpt"
EVENT_THRESHOLD=0.3
CLIP_DURATION=4.0
OUTPUT_DATASET="./datasets/tennis_events_dataset.json"
BACKEND_PORT=8000
FRONTEND_PORT=3000

# ヘルプメッセージ
show_help() {
    cat << EOF
Tennis Event Annotation System - 統合実行スクリプト

使用方法:
    $0 [OPTIONS] COMMAND

COMMANDS:
    setup           依存関係をインストールし、環境をセットアップ
    prepare         手動配置クリップから空アノテーション生成
    server          Webアノテーションサーバーを起動
    merge           アノテーションをCOCO形式にマージ
    full            完全なワークフローを実行（prepare → server → merge）
    clean           一時ファイルと中間ファイルをクリーンアップ

OPTIONS:
    -i, --input VIDEO      元動画のパス（記録用、オプション）
    -w, --workspace DIR    ワークスペースディレクトリ（default: $WORKSPACE_DIR）
    -o, --output FILE      出力データセットファイル（default: $OUTPUT_DATASET）
    --backend-port PORT    バックエンドポート（default: $BACKEND_PORT）
    --frontend-port PORT   フロントエンドポート（default: $FRONTEND_PORT）
    -h, --help             このヘルプを表示

MANUAL CLIP WORKFLOW:
    1. 手動でクリップを抽出・配置: $WORKSPACE_DIR/clips/*.mp4
    2. 空アノテーション生成: $0 prepare
    3. Webツールでアノテーション: $0 server
    4. データセット生成: $0 merge

例:
    # 環境セットアップ
    $0 setup

    # クリップを手動配置後、空アノテーション生成
    $0 prepare

    # アノテーションサーバー起動
    $0 server

    # COCO形式にマージ
    $0 merge

    # 完全ワークフロー実行（クリップ配置後）
    $0 full

EOF
}

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_VIDEO="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DATASET="$2"
            shift 2
            ;;
        --backend-port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --frontend-port)
            FRONTEND_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        setup|prepare|server|merge|full|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# コマンドチェック
if [ -z "$COMMAND" ]; then
    log_error "コマンドが指定されていません"
    show_help
    exit 1
fi

# ワークスペースディレクトリの設定
CLIPS_DIR="$WORKSPACE_DIR/clips"
ANNOTATIONS_DIR="$WORKSPACE_DIR/annotations"

# 共通チェック関数
check_file() {
    if [ ! -f "$1" ]; then
        log_error "ファイルが見つかりません: $1"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        log_error "ディレクトリが見つかりません: $1"
        exit 1
    fi
}

# 環境セットアップ
setup_environment() {
    log_info "環境をセットアップ中..."
    
    # Python依存関係のインストール
    log_info "Python依存関係をインストール中..."
    pip install -r requirements.txt
    
    # フロントエンド依存関係のインストール
    if [ -d "web_app/frontend" ]; then
        log_info "フロントエンド依存関係をインストール中..."
        cd web_app/frontend
        npm install
        cd - > /dev/null
    fi
    
    # ワークスペースディレクトリの作成
    log_info "ワークスペースディレクトリを作成中..."
    mkdir -p "$WORKSPACE_DIR"
    mkdir -p "$CLIPS_DIR"
    mkdir -p "$ANNOTATIONS_DIR"
    
    # データセット出力ディレクトリの作成
    OUTPUT_DIR=$(dirname "$OUTPUT_DATASET")
    mkdir -p "$OUTPUT_DIR"
    
    log_info "環境セットアップが完了しました"
}

# 手動配置クリップから空アノテーション生成
run_prepare() {
    log_info "手動配置クリップから空のアノテーション生成中..."
    
    # ワークスペースディレクトリ作成
    mkdir -p "$CLIPS_DIR"
    mkdir -p "$ANNOTATIONS_DIR"
    
    # クリップディレクトリの存在確認
    if [ ! -d "$CLIPS_DIR" ]; then
        log_error "クリップディレクトリが見つかりません: $CLIPS_DIR"
        echo ""
        echo "📋 手動クリップ配置の手順:"
        echo "1. ディレクトリを作成: mkdir -p $CLIPS_DIR"
        echo "2. 動画クリップを配置: $CLIPS_DIR/*.mp4"
        echo "3. 再度このコマンドを実行してください"
        exit 1
    fi
    
    # クリップファイルの存在確認
    CLIP_COUNT=$(find "$CLIPS_DIR" -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.flv" -o -name "*.wmv" 2>/dev/null | wc -l)
    
    if [ "$CLIP_COUNT" -eq 0 ]; then
        log_error "クリップファイルが見つかりません"
        echo ""
        echo "📋 手動クリップ配置の手順:"
        echo "1. 動画編集ツールまたはFFmpegを使用して、元動画からクリップを抽出"
        echo "2. 抽出したクリップを $CLIPS_DIR/ に配置"
        echo "   対応形式: .mp4, .avi, .mov, .mkv, .flv, .wmv"
        echo "3. 再度このコマンドを実行してください"
        echo ""
        echo "FFmpegの例:"
        echo "  ffmpeg -i input.mp4 -ss 00:01:30 -t 00:00:04 -c copy $CLIPS_DIR/clip_001.mp4"
        exit 1
    fi
    
    log_info "発見されたクリップファイル数: $CLIP_COUNT"
    log_info "クリップディレクトリ: $CLIPS_DIR"
    log_info "アノテーション出力ディレクトリ: $ANNOTATIONS_DIR"
    
    # 空のアノテーションJSON生成
    log_info "空のアノテーションJSONを生成中..."
    python generate_empty_annotations.py \
        --clips_dir "$CLIPS_DIR" \
        --annotations_dir "$ANNOTATIONS_DIR" \
        --source_video "${INPUT_VIDEO:-手動配置}" \
        --validate \
        --verbose
    
    if [ $? -eq 0 ]; then
        # 生成されたファイル数を報告
        ANNOTATION_COUNT=$(find "$ANNOTATIONS_DIR" -name "*.json" | wc -l)
        
        log_info "空のアノテーションJSON生成が完了しました"
        log_info "生成されたクリップ数: $CLIP_COUNT"
        log_info "生成されたアノテーション数: $ANNOTATION_COUNT"
        echo ""
        echo "📋 次のステップ:"
        echo "1. Webアノテーションツールを起動: $0 server"
        echo "2. ブラウザでアノテーション作業を実行"
        echo "3. 完了後、データセットを生成: $0 merge"
    else
        log_error "空のアノテーションJSON生成に失敗しました"
        exit 1
    fi
}

# Webサーバー起動
run_server() {
    log_info "Webアノテーションサーバーを起動中..."
    
    # ワークスペースディレクトリチェック
    check_dir "$WORKSPACE_DIR"
    
    # バックグラウンドでバックエンド起動
    log_info "バックエンドサーバーを起動中（ポート: $BACKEND_PORT）..."
    python web_app/app.py \
        --host 127.0.0.1 \
        --port "$BACKEND_PORT" \
        --data_dir "$WORKSPACE_DIR" &
    BACKEND_PID=$!
    
    # バックエンドの起動を少し待つ
    sleep 3
    
    # フロントエンド起動（あれば）
    if [ -d "web_app/frontend" ]; then
        log_info "フロントエンドサーバーを起動中（ポート: $FRONTEND_PORT）..."
        cd web_app/frontend
        BROWSER=none PORT="$FRONTEND_PORT" npm start &
        FRONTEND_PID=$!
        cd - > /dev/null
        
        log_info "サーバーが起動しました:"
        log_info "  - バックエンド: http://127.0.0.1:$BACKEND_PORT"
        log_info "  - フロントエンド: http://127.0.0.1:$FRONTEND_PORT"
        log_info ""
        log_info "アノテーション作業を開始してください。"
        log_info "終了するには Ctrl+C を押してください。"
        
        # シグナルハンドラでプロセスを終了
        trap 'log_info "サーバーを終了中..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT TERM
        wait
    else
        log_info "バックエンドサーバーが起動しました:"
        log_info "  - URL: http://127.0.0.1:$BACKEND_PORT"
        log_info ""
        log_info "終了するには Ctrl+C を押してください。"
        
        trap 'log_info "サーバーを終了中..."; kill $BACKEND_PID 2>/dev/null; exit' INT TERM
        wait $BACKEND_PID
    fi
}

# COCO形式マージ
run_merge() {
    log_info "アノテーションをCOCO形式にマージ中..."
    
    # アノテーションディレクトリチェック
    check_dir "$ANNOTATIONS_DIR"
    
    # アノテーションファイル数チェック
    ANNOTATION_COUNT=$(find "$ANNOTATIONS_DIR" -name "*.json" | wc -l)
    if [ "$ANNOTATION_COUNT" -eq 0 ]; then
        log_error "アノテーションファイルが見つかりません: $ANNOTATIONS_DIR"
        exit 1
    fi
    
    log_info "マージ対象のアノテーション数: $ANNOTATION_COUNT"
    
    # 出力ディレクトリの作成
    OUTPUT_DIR=$(dirname "$OUTPUT_DATASET")
    mkdir -p "$OUTPUT_DIR"
    
    # 統計ファイルのパス
    STATS_FILE="${OUTPUT_DATASET%.*}_statistics.json"
    
    # マージ実行
    python merge_to_coco.py \
        --input_dir "$ANNOTATIONS_DIR" \
        --output_file "$OUTPUT_DATASET" \
        --stats_file "$STATS_FILE" \
        --cleanup \
        --verbose
    
    log_info "マージが完了しました"
    log_info "出力ファイル: $OUTPUT_DATASET"
    log_info "統計ファイル: $STATS_FILE"
}

# クリーンアップ
run_clean() {
    log_info "一時ファイルをクリーンアップ中..."
    
    # 一時ディレクトリの削除
    if [ -d "temp_frames" ]; then
        rm -rf temp_frames
        log_info "一時フレームファイルを削除しました"
    fi
    
    # バックアップファイルの削除
    find "$WORKSPACE_DIR" -name "*.bak" -delete 2>/dev/null || true
    log_info "バックアップファイルを削除しました"
    
    # ログファイルの削除（あれば）
    find . -name "*.log" -delete 2>/dev/null || true
    
    log_info "クリーンアップが完了しました"
}

# 完全ワークフロー実行
run_full() {
    log_info "完全ワークフローを開始します..."
    
    # 1. 環境セットアップ（必要に応じて）
    if [ ! -f "$WORKSPACE_DIR/.setup_done" ]; then
        setup_environment
        touch "$WORKSPACE_DIR/.setup_done"
    fi
    
    # 2. 空アノテーション生成
    run_prepare
    
    # 3. サーバー起動の案内
    log_info ""
    log_info "=========================================="
    log_info "空アノテーション生成が完了しました。"
    log_info "次にWebアノテーションツールを起動します。"
    log_info ""
    log_info "アノテーション作業を完了したら、"
    log_info "サーバーを停止（Ctrl+C）してください。"
    log_info "=========================================="
    log_info ""
    
    read -p "Webサーバーを起動しますか？ (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_server
    fi
    
    # 4. マージ
    log_info ""
    log_info "=========================================="
    log_info "アノテーション作業が完了しました。"
    log_info "COCO形式データセットにマージします。"
    log_info "=========================================="
    log_info ""
    
    run_merge
    
    # 5. クリーンアップ
    read -p "一時ファイルをクリーンアップしますか？ (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_clean
    fi
    
    log_info ""
    log_info "=========================================="
    log_info "完全ワークフローが完了しました！"
    log_info "データセット: $OUTPUT_DATASET"
    log_info "=========================================="
}

# メイン処理
case $COMMAND in
    setup)
        setup_environment
        ;;
    prepare)
        run_prepare
        ;;
    server)
        run_server
        ;;
    merge)
        run_merge
        ;;
    clean)
        run_clean
        ;;
    full)
        run_full
        ;;
    *)
        log_error "不明なコマンド: $COMMAND"
        show_help
        exit 1
        ;;
esac 