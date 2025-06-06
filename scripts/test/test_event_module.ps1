# イベント検出モジュールのテストスクリプト (PowerShell版)
# UTF-8 with BOM

# 色の設定
$GREEN = [ConsoleColor]::Green
$RED = [ConsoleColor]::Red
$YELLOW = [ConsoleColor]::Yellow

function Write-ColorOutput {
    param (
        [string]$Message,
        [ConsoleColor]$ForegroundColor = [ConsoleColor]::White
    )
    $oldColor = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $oldColor
}

Write-ColorOutput "===================================================" $YELLOW
Write-ColorOutput "  Event Detection Module Test Execution            " $YELLOW
Write-ColorOutput "===================================================" $YELLOW

# プロジェクトルートに移動
$ProjectRoot = $PSScriptRoot
while (-not (Test-Path "$ProjectRoot\src") -and $ProjectRoot -ne "") {
    $ProjectRoot = Split-Path -Parent $ProjectRoot
}

if (-not (Test-Path "$ProjectRoot\src")) {
    Write-ColorOutput "Project root not found." $RED
    exit 1
}

Set-Location $ProjectRoot

# テスト対象ファイル
$DATASET_TEST = "tests\data\test_event_dataset_io.py"
$MODEL_TEST = "tests\train_model_instantiate\test_event_models.py"
$TRANSFORMER_TEST = "tests\train_model_instantiate\test_transformer_model.py"

# ログファイルの設定
$DATASET_LOG = "tests\logs\event_dataset_test.log"
$MODEL_LOG = "tests\logs\event_model_test.log"
$TRANSFORMER_LOG = "tests\logs\transformer_model_test.log"

# ログディレクトリがなければ作成
if (-not (Test-Path "tests\logs")) {
    New-Item -ItemType Directory -Path "tests\logs" -Force | Out-Null
}

Write-ColorOutput "1. Dataset test running..." $YELLOW
$datasetOutput = & python -m pytest $DATASET_TEST -v
$DATASET_RESULT = $LASTEXITCODE
$datasetOutput | Out-File -FilePath $DATASET_LOG -Encoding utf8

Write-ColorOutput "2. RNN model test running..." $YELLOW
$modelOutput = & python -m pytest $MODEL_TEST -v
$MODEL_RESULT = $LASTEXITCODE
$modelOutput | Out-File -FilePath $MODEL_LOG -Encoding utf8

Write-ColorOutput "3. Transformer model test running..." $YELLOW
$transformerOutput = & python -m pytest $TRANSFORMER_TEST -v
$TRANSFORMER_RESULT = $LASTEXITCODE
$transformerOutput | Out-File -FilePath $TRANSFORMER_LOG -Encoding utf8

# 結果の表示
Write-ColorOutput "`n===================================================" $YELLOW
Write-ColorOutput "  Test Results                                    " $YELLOW
Write-ColorOutput "===================================================" $YELLOW

if ($DATASET_RESULT -eq 0) {
    Write-ColorOutput "✓ Dataset test: Success" $GREEN
} else {
    Write-ColorOutput "✗ Dataset test: Failed" $RED
    Write-ColorOutput "  Check log file for details: $DATASET_LOG" $RED
    Write-ColorOutput "  Main errors: " $RED
    $datasetErrors = $datasetOutput | Select-String -Pattern "FAILED" -Context 0,3
    if ($datasetErrors) {
        foreach ($error in $datasetErrors[0..2]) {
            Write-Output $error
        }
    }
}

if ($MODEL_RESULT -eq 0) {
    Write-ColorOutput "✓ RNN model test: Success" $GREEN
} else {
    Write-ColorOutput "✗ RNN model test: Failed" $RED
    Write-ColorOutput "  Check log file for details: $MODEL_LOG" $RED
    Write-ColorOutput "  Main errors: " $RED
    $modelErrors = $modelOutput | Select-String -Pattern "FAILED" -Context 0,3
    if ($modelErrors) {
        foreach ($error in $modelErrors[0..2]) {
            Write-Output $error
        }
    }
}

if ($TRANSFORMER_RESULT -eq 0) {
    Write-ColorOutput "✓ Transformer model test: Success" $GREEN
} else {
    Write-ColorOutput "✗ Transformer model test: Failed" $RED
    Write-ColorOutput "  Check log file for details: $TRANSFORMER_LOG" $RED
    Write-ColorOutput "  Main errors: " $RED
    $transformerErrors = $transformerOutput | Select-String -Pattern "FAILED" -Context 0,3
    if ($transformerErrors) {
        foreach ($error in $transformerErrors[0..2]) {
            Write-Output $error
        }
    }
}

# 最終結果
if (($DATASET_RESULT -eq 0) -and ($MODEL_RESULT -eq 0) -and ($TRANSFORMER_RESULT -eq 0)) {
    Write-ColorOutput "`nAll tests passed successfully!" $GREEN
    exit 0
} else {
    Write-ColorOutput "`nSome tests failed. Please check the log files for details." $RED
    Write-ColorOutput "Dataset test log: $DATASET_LOG" $RED
    Write-ColorOutput "RNN model test log: $MODEL_LOG" $RED
    Write-ColorOutput "Transformer model test log: $TRANSFORMER_LOG" $RED
    exit 1
} 