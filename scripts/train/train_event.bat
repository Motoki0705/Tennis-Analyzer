@echo off
REM イベント検出モデルのトレーニングスクリプト (Windows Batch)

setlocal EnableDelayedExpansion

REM デフォルト値
set MODEL=transformer
set BALANCED_DATASET=
set METHOD=hybrid
set CREATE_BALANCED=false

REM コマンドライン引数の解析
:parse_args
if "%1"=="" goto :done_parsing
if /i "%1"=="--model" (
    set MODEL=%2
    shift /1
    shift /1
    goto :parse_args
)
if /i "%1"=="-m" (
    set MODEL=%2
    shift /1
    shift /1
    goto :parse_args
)
if /i "%1"=="--balanced-dataset" (
    set BALANCED_DATASET=%2
    shift /1
    shift /1
    goto :parse_args
)
if /i "%1"=="-b" (
    set BALANCED_DATASET=%2
    shift /1
    shift /1
    goto :parse_args
)
if /i "%1"=="--method" (
    set METHOD=%2
    shift /1
    shift /1
    goto :parse_args
)
if /i "%1"=="-t" (
    set METHOD=%2
    shift /1
    shift /1
    goto :parse_args
)
if /i "%1"=="--create-balanced" (
    set CREATE_BALANCED=true
    shift /1
    goto :parse_args
)
if /i "%1"=="-c" (
    set CREATE_BALANCED=true
    shift /1
    goto :parse_args
)
if /i "%1"=="--help" (
    goto :show_help
)
if /i "%1"=="-h" (
    goto :show_help
)
echo 不明なオプション: %1
goto :show_help

:show_help
echo 使用方法: %0 [オプション]
echo.
echo オプション:
echo   -m, --model MODEL          使用するモデル (bilstm または transformer)
echo   -b, --balanced-dataset PATH バランス調整されたデータセットのパス
echo   -t, --method METHOD        バランス調整手法 (under_sampling, over_sampling, hybrid)
echo   -c, --create-balanced      新しくバランス調整されたデータセットを作成する
echo   -h, --help                 このヘルプを表示
echo.
echo 例:
echo   %0 --model transformer --create-balanced --method hybrid
echo   %0 --model bilstm --balanced-dataset outputs/analysis/balanced_dataset_hybrid.json
exit /b 1

:done_parsing

REM 開始メッセージ
echo [92mイベント検出モデルのトレーニングを開始します...[0m

REM パラメータ表示
echo [96mパラメータ:[0m
echo [96m  モデル: %MODEL%[0m
echo [96m  バランス調整データセット: %BALANCED_DATASET%[0m
echo [96m  バランス調整手法: %METHOD%[0m
echo [96m  データセット作成: %CREATE_BALANCED%[0m

REM バランス調整されたデータセットを作成する場合
if "%CREATE_BALANCED%"=="true" (
    echo [93mバランス調整されたデータセットを作成します...[0m
    
    REM ディレクトリが存在しない場合は作成
    if not exist outputs\analysis mkdir outputs\analysis
    
    REM バランス調整スクリプトの実行
    python scripts/analyze_event_status_ratio.py --create-balanced --method %METHOD%
    
    REM 作成されたファイルを指定
    set BALANCED_DATASET=outputs/analysis/balanced_dataset_%METHOD%.json
)

REM コマンド構築
set CMD=python scripts/train/train_event_detector.py

REM モデルの指定
set CMD=%CMD% model=%MODEL%

REM バランス調整されたデータセットの指定
if not "%BALANCED_DATASET%"=="" (
    set CMD=%CMD% +balanced_dataset_path=%BALANCED_DATASET%
)

REM コマンド表示
echo [93m実行コマンド: %CMD%[0m

REM コマンド実行
%CMD%

endlocal 