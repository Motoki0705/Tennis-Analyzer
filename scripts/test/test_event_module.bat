@echo off
REM イベント検出モジュールのテストスクリプト (バッチファイル版)

setlocal

REM 色の設定
set GREEN=92
set RED=91
set YELLOW=93

echo [%YELLOW%m====================================================[0m
echo [%YELLOW%m  イベント検出モジュールのテスト実行            [0m
echo [%YELLOW%m====================================================[0m

REM プロジェクトルートに移動
set "ProjectRoot=%~dp0..\..\"
cd /d "%ProjectRoot%"

REM テスト対象ファイル
set DATASET_TEST=tests\data\test_event_dataset_io.py
set MODEL_TEST=tests\train_model_instantiate\test_event_models.py
set TRANSFORMER_TEST=tests\train_model_instantiate\test_transformer_model.py

REM ログファイルの設定
set DATASET_LOG=tests\logs\event_dataset_test.log
set MODEL_LOG=tests\logs\event_model_test.log
set TRANSFORMER_LOG=tests\logs\transformer_model_test.log

REM ログディレクトリがなければ作成
if not exist tests\logs mkdir tests\logs

echo [%YELLOW%m1. データセットのテスト実行中...[0m
python -m pytest %DATASET_TEST% -v > %DATASET_LOG% 2>&1
set DATASET_RESULT=%ERRORLEVEL%

echo [%YELLOW%m2. RNNモデルのテスト実行中...[0m
python -m pytest %MODEL_TEST% -v > %MODEL_LOG% 2>&1
set MODEL_RESULT=%ERRORLEVEL%

echo [%YELLOW%m3. Transformerモデルのテスト実行中...[0m
python -m pytest %TRANSFORMER_TEST% -v > %TRANSFORMER_LOG% 2>&1
set TRANSFORMER_RESULT=%ERRORLEVEL%

REM 結果の表示
echo.
echo [%YELLOW%m====================================================[0m
echo [%YELLOW%m  テスト結果                                    [0m
echo [%YELLOW%m====================================================[0m

if %DATASET_RESULT% EQU 0 (
    echo [%GREEN%m✓ データセットテスト: 成功[0m
) else (
    echo [%RED%m✗ データセットテスト: 失敗[0m
    echo [%RED%m  詳細はログファイルを確認: %DATASET_LOG%[0m
    echo [%RED%m  主なエラー: [0m
    findstr /C:"FAILED" /C:"Error" /C:"Exception" %DATASET_LOG%
)

if %MODEL_RESULT% EQU 0 (
    echo [%GREEN%m✓ RNNモデルテスト: 成功[0m
) else (
    echo [%RED%m✗ RNNモデルテスト: 失敗[0m
    echo [%RED%m  詳細はログファイルを確認: %MODEL_LOG%[0m
    echo [%RED%m  主なエラー: [0m
    findstr /C:"FAILED" /C:"Error" /C:"Exception" %MODEL_LOG%
)

if %TRANSFORMER_RESULT% EQU 0 (
    echo [%GREEN%m✓ Transformerモデルテスト: 成功[0m
) else (
    echo [%RED%m✗ Transformerモデルテスト: 失敗[0m
    echo [%RED%m  詳細はログファイルを確認: %TRANSFORMER_LOG%[0m
    echo [%RED%m  主なエラー: [0m
    findstr /C:"FAILED" /C:"Error" /C:"Exception" %TRANSFORMER_LOG%
)

REM 最終結果
if %DATASET_RESULT% EQU 0 if %MODEL_RESULT% EQU 0 if %TRANSFORMER_RESULT% EQU 0 (
    echo.
    echo [%GREEN%m全てのテストが成功しました！[0m
    exit /b 0
) else (
    echo.
    echo [%RED%m一部のテストが失敗しました。詳細はログファイルを確認してください。[0m
    echo [%RED%mデータセットテストログ: %DATASET_LOG%[0m
    echo [%RED%mRNNモデルテストログ: %MODEL_LOG%[0m
    echo [%RED%mTransformerモデルテストログ: %TRANSFORMER_LOG%[0m
    exit /b 1
)

endlocal 