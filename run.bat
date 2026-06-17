@echo off
REM AI Revenue Intelligence Inference CLI Runner for Windows CMD
REM Usage: run.bat [data_dir] [model_path] [output_path]

set DATA_DIR=%1
set MODEL_PATH=%2
set OUTPUT_PATH=%3

if "%DATA_DIR%"=="" set DATA_DIR=./data
if "%MODEL_PATH%"=="" set MODEL_PATH=./pickle/model.pkl
if "%OUTPUT_PATH%"=="" set OUTPUT_PATH=./output/predictions.csv

echo === AI Revenue Intelligence: Inference CLI (Windows CMD) ===
echo Data Directory: %DATA_DIR%
echo Model Path:     %MODEL_PATH%
echo Output Path:    %OUTPUT_PATH%
echo ========================================================

python src/predict.py "%DATA_DIR%" "%MODEL_PATH%" "%OUTPUT_PATH%"
