# AI Revenue Intelligence Inference CLI Runner for Windows PowerShell
# Usage: .\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv

param (
    [string]$DataDir = "./data",
    [string]$ModelPath = "./pickle/model.pkl",
    [string]$OutputPath = "./output/predictions.csv"
)

Write-Host "=== AI Revenue Intelligence: Inference CLI (PowerShell) ===" -ForegroundColor Cyan
Write-Host "Data Directory: $DataDir"
Write-Host "Model Path:     $ModelPath"
Write-Host "Output Path:    $OutputPath"
Write-Host "============================================================" -ForegroundColor Cyan

python src/predict.py $DataDir $ModelPath $OutputPath
