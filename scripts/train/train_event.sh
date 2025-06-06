#!/bin/bash
# イベント検出モデルのトレーニングスクリプト (Bash/Shell)

# デフォルト値
MODEL="transformer"
BALANCED_DATASET=""
METHOD="hybrid"
CREATE_BALANCED=false

# ヘルプ表示
function show_help {
    echo "使用方法: $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  -m, --model MODEL          使用するモデル (bilstm または transformer)"
    echo "  -b, --balanced-dataset PATH バランス調整されたデータセットのパス"
    echo "  -t, --method METHOD        バランス調整手法 (under_sampling, over_sampling, hybrid)"
    echo "  -c, --create-balanced      新しくバランス調整されたデータセットを作成する"
    echo "  -h, --help                 このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0 --model transformer --create-balanced --method hybrid"
    echo "  $0 --model bilstm --balanced-dataset outputs/analysis/balanced_dataset_hybrid.json"
    exit 1
}

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -b|--balanced-dataset)
            BALANCED_DATASET="$2"
            shift 2
            ;;
        -t|--method)
            METHOD="$2"
            shift 2
            ;;
        -c|--create-balanced)
            CREATE_BALANCED=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "不明なオプション: $1"
            show_help
            ;;
    esac
done

# 開始メッセージ
echo -e "\e[32mイベント検出モデルのトレーニングを開始します...\e[0m"

# パラメータ表示
echo -e "\e[36mパラメータ:\e[0m"
echo -e "\e[36m  モデル: $MODEL\e[0m"
echo -e "\e[36m  バランス調整データセット: $BALANCED_DATASET\e[0m"
echo -e "\e[36m  バランス調整手法: $METHOD\e[0m"
echo -e "\e[36m  データセット作成: $CREATE_BALANCED\e[0m"

# バランス調整されたデータセットを作成する場合
if $CREATE_BALANCED; then
    echo -e "\e[33mバランス調整されたデータセットを作成します...\e[0m"
    
    # ディレクトリが存在しない場合は作成
    mkdir -p outputs/analysis
    
    # バランス調整スクリプトの実行
    python scripts/analyze_event_status_ratio.py --create-balanced --method $METHOD
    
    # 作成されたファイルを指定
    BALANCED_DATASET="outputs/analysis/balanced_dataset_${METHOD}.json"
fi

# コマンド構築
CMD="python scripts/train/train_event_detector.py"

# モデルの指定
CMD="${CMD} model=${MODEL}"

# バランス調整されたデータセットの指定
if [ ! -z "$BALANCED_DATASET" ]; then
    CMD="${CMD} +balanced_dataset_path=${BALANCED_DATASET}"
fi

# コマンド表示
echo -e "\e[33m実行コマンド: $CMD\e[0m"

# コマンド実行
eval $CMD 