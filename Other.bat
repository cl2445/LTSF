@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    LTSF-Linear Experiment Script
echo ========================================
echo.

:: Create log directories
if not exist ".\logs" mkdir ".\logs"
if not exist ".\logs\LongForecasting" mkdir ".\logs\LongForecasting"

:: Set experiment parameters
set seq_len=336
set model_name=FLinear
set GPU_ID=0
set pred_lengths=96 192 336 720

:: Display experiment configuration
echo Experiment Configuration:
echo - Model: %model_name%
echo - Sequence Length: %seq_len%
echo - GPU ID: %GPU_ID%
echo - Prediction Lengths: %pred_lengths%
echo - Datasets: Exchange Rate, Electricity, Traffic
echo.

:: Calculate total experiments
set "total_experiments=0"
for %%p in (%pred_lengths%) do (
    set /a total_experiments+=3
)
echo Total experiments to run: %total_experiments%
echo.

:: Record start time
set "start_time=%date% %time%"
echo Experiment start time: %start_time%
echo ========================================
echo.

set "current_exp=0"

for %%p in (%pred_lengths%) do (
    echo.
    echo [Prediction Length: %%p] Starting...
    echo ----------------------------------------
    
    :: Exchange Rate experiment
    set /a current_exp+=1
    echo [!current_exp!/%total_experiments%] Running Exchange Rate experiment...
    echo Start time: %date% %time%
    
    python -u run_longExp.py ^
        --is_training 1 ^
        --root_path .\dataset\ ^
        --data_path exchange_rate.csv ^
        --model_id Exchange_%seq_len%_%%p ^
        --model %model_name% ^
        --data custom ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 8 ^
        --des Exp ^
        --itr 1 --batch_size 8 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Exchange_%seq_len%_%%p.log ^
        --use_gpu True ^
        --gpu %GPU_ID%
    
    if !errorlevel! equ 0 (
        echo [SUCCESS] Exchange Rate experiment completed
    ) else (
        echo [FAILED] Exchange Rate experiment failed
    )
    echo End time: %date% %time%
    echo.

    :: Electricity experiment
    set /a current_exp+=1
    echo [!current_exp!/%total_experiments%] Running Electricity experiment...
    echo Start time: %date% %time%
    
    python -u run_longExp.py ^
        --is_training 1 ^
        --root_path .\dataset\ ^
        --data_path electricity.csv ^
        --model_id Electricity_%seq_len%_%%p ^
        --model %model_name% ^
        --data custom ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 321 ^
        --des Exp ^
        --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Electricity_%seq_len%_%%p.log ^
        --use_gpu True ^
        --gpu %GPU_ID%
    
    if !errorlevel! equ 0 (
        echo [SUCCESS] Electricity experiment completed
    ) else (
        echo [FAILED] Electricity experiment failed
    )
    echo End time: %date% %time%
    echo.

    :: Traffic experiment
    set /a current_exp+=1
    echo [!current_exp!/%total_experiments%] Running Traffic experiment...
    echo Start time: %date% %time%
    
    python -u run_longExp.py ^
        --is_training 1 ^
        --root_path .\dataset\ ^
        --model_id traffic_%seq_len%_%%p ^
        --model %model_name% ^
        --data custom ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 862 ^
        --des Exp ^
        --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Traffic_%seq_len%_%%p.log ^
        --use_gpu True ^
        --gpu %GPU_ID%
    
    if !errorlevel! equ 0 (
        echo [SUCCESS] Traffic experiment completed
    ) else (
        echo [FAILED] Traffic experiment failed
    )
    echo End time: %date% %time%
    echo.
    
    echo [Prediction Length: %%p] All experiments completed
    echo ----------------------------------------
)

:: Record end time
set "end_time=%date% %time%"
echo.
echo ========================================
echo All experiments completed!
echo Start time: %start_time%
echo End time: %end_time%
echo Log files saved in: logs\LongForecasting\
echo ========================================

:: Display log file list
echo.
echo Generated log files:
dir /b logs\LongForecasting\%model_name%_*.log

echo.
pause
   

