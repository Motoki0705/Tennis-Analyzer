#!/usr/bin/env pwsh
<#
.SYNOPSIS
    イベント検出モデルのトレーニングスクリプト (PowerShell)

.DESCRIPTION
    バランス調整されたデータセットを使用してイベント検出モデルをトレーニングします。

.PARAMETER Model
    使用するモデルタイプ ("bilstm" または "transformer")

.PARAMETER BalancedDataset
    バランス調整されたデータセットのパス

.PARAMETER Method
    バランス調整手法 ("under_sampling", "over_sampling", "hybrid")

.PARAMETER CreateBalanced
    新しくバランス調整されたデータセットを作成するかどうか

.EXAMPLE
    ./scripts/train/train_event.ps1 -Model transformer -CreateBalanced -Method hybrid

.EXAMPLE
    ./scripts/train/train_event.ps1 -Model bilstm -BalancedDataset "outputs/analysis/balanced_dataset_hybrid.json"
#>

param (
    [string]$Model = "transformer",
    [string]$BalancedDataset = "",
    [ValidateSet("under_sampling", "over_sampling", "hybrid")]
    [string]$Method = "hybrid",
    [switch]$CreateBalanced
)

# 開始メッセージ
Write-Host "イベント検出モデルのトレーニングを開始します..." -ForegroundColor Green

# パラメータ表示
Write-Host "パラメータ:" -ForegroundColor Cyan
Write-Host "  モデル: $Model" -ForegroundColor Cyan
Write-Host "  バランス調整データセット: $BalancedDataset" -ForegroundColor Cyan
Write-Host "  バランス調整手法: $Method" -ForegroundColor Cyan
Write-Host "  データセット作成: $CreateBalanced" -ForegroundColor Cyan

# バランス調整されたデータセットを作成する場合
if ($CreateBalanced) {
    Write-Host "バランス調整されたデータセットを作成します..." -ForegroundColor Yellow
    
    # ディレクトリが存在しない場合は作成
    $outputDir = "outputs/analysis"
    if (-not (Test-Path $outputDir)) {
        New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
    }
    
    # バランス調整スクリプトの実行
    python scripts/analyze_event_status_ratio.py --create-balanced --method $Method
    
    # 作成されたファイルを指定
    $BalancedDataset = "outputs/analysis/balanced_dataset_$Method.json"
}

# コマンド構築
$cmd = "python scripts/train/train_event_detector.py"

# モデルの指定
$cmd += " model=$Model"

# バランス調整されたデータセットの指定
if ($BalancedDataset -ne "") {
    $cmd += " +balanced_dataset_path=$BalancedDataset"
}

# コマンド表示
Write-Host "実行コマンド: $cmd" -ForegroundColor Yellow

# コマンド実行
Invoke-Expression $cmd 