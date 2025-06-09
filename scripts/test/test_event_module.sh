#!/bin/bash
# イベント検出モジュールのテストスクリプト

# 色の設定
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===================================================${NC}"
echo -e "${YELLOW}  イベント検出モジュールのテスト実行            ${NC}"
echo -e "${YELLOW}===================================================${NC}"

# プロジェクトルートに移動
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(pwd)
    while [ ! -d "$PROJECT_ROOT/src" ] && [ "$PROJECT_ROOT" != "/" ]; do
        PROJECT_ROOT=$(dirname "$PROJECT_ROOT")
    done
fi

cd "$PROJECT_ROOT" || exit 1

# テスト対象ファイル
DATASET_TEST="tests/data/test_event_dataset_io.py"
MODEL_TEST="tests/train_model_instantiate/test_event_models.py"
TRANSFORMER_TEST="tests/train_model_instantiate/test_transformer_model.py"

# ログファイルの設定
DATASET_LOG="tests/logs/event_dataset_test.log"
MODEL_LOG="tests/logs/event_model_test.log"
TRANSFORMER_LOG="tests/logs/transformer_model_test.log"

# ログディレクトリがなければ作成
mkdir -p tests/logs

echo -e "${YELLOW}1. データセットのテスト実行中...${NC}"
python -m pytest "$DATASET_TEST" -v | tee "$DATASET_LOG"
DATASET_RESULT=$?

echo -e "${YELLOW}2. RNNモデルのテスト実行中...${NC}"
python -m pytest "$MODEL_TEST" -v | tee "$MODEL_LOG"
MODEL_RESULT=$?

echo -e "${YELLOW}3. Transformerモデルのテスト実行中...${NC}"
python -m pytest "$TRANSFORMER_TEST" -v | tee "$TRANSFORMER_LOG"
TRANSFORMER_RESULT=$?

# 結果の表示
echo -e "\n${YELLOW}===================================================${NC}"
echo -e "${YELLOW}  テスト結果                                    ${NC}"
echo -e "${YELLOW}===================================================${NC}"

if [ $DATASET_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ データセットテスト: 成功${NC}"
else
    echo -e "${RED}✗ データセットテスト: 失敗${NC}"
    echo -e "${RED}  詳細はログファイルを確認: ${DATASET_LOG}${NC}"
    echo -e "${RED}  主なエラー: ${NC}"
    grep -A 3 "FAILED" "$DATASET_LOG" | head -n 10
fi

if [ $MODEL_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ RNNモデルテスト: 成功${NC}"
else
    echo -e "${RED}✗ RNNモデルテスト: 失敗${NC}"
    echo -e "${RED}  詳細はログファイルを確認: ${MODEL_LOG}${NC}"
    echo -e "${RED}  主なエラー: ${NC}"
    grep -A 3 "FAILED" "$MODEL_LOG" | head -n 10
fi

if [ $TRANSFORMER_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Transformerモデルテスト: 成功${NC}"
else
    echo -e "${RED}✗ Transformerモデルテスト: 失敗${NC}"
    echo -e "${RED}  詳細はログファイルを確認: ${TRANSFORMER_LOG}${NC}"
    echo -e "${RED}  主なエラー: ${NC}"
    grep -A 3 "FAILED" "$TRANSFORMER_LOG" | head -n 10
fi

# 最終結果
if [ $DATASET_RESULT -eq 0 ] && [ $MODEL_RESULT -eq 0 ] && [ $TRANSFORMER_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}全てのテストが成功しました！${NC}"
    exit 0
else
    echo -e "\n${RED}一部のテストが失敗しました。詳細はログファイルを確認してください。${NC}"
    echo -e "${RED}データセットテストログ: ${DATASET_LOG}${NC}"
    echo -e "${RED}RNNモデルテストログ: ${MODEL_LOG}${NC}"
    echo -e "${RED}Transformerモデルテストログ: ${TRANSFORMER_LOG}${NC}"
    exit 1
fi 