@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo    LTSF-Linear 四数据集实验脚本 (改进版)
echo ========================================
echo.

:: 创建日志目录
if not exist ".\logs" mkdir ".\logs"
if not exist ".\logs\LongForecasting" mkdir ".\logs\LongForecasting"

:: 设置实验参数
set seq_len=336
set model_name=FLinear
set GPU_ID=0
set pred_lengths=96 192 336 720

:: 显示实验配置
echo 实验配置:
echo - 模型: %model_name%
echo - 序列长度: %seq_len%
echo - GPU ID: %GPU_ID%
echo - 预测长度: %pred_lengths%
echo - 数据集: Exchange Rate, Electricity, Traffic, Weather
echo.

:: 计算总实验数
set "total_experiments=0"
for %%p in (%pred_lengths%) do (
    set /a total_experiments+=4
)
echo 总实验数: %total_experiments%
echo.

:: 记录开始时间
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "start_date=%%a %%b %%c"
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "start_time=%%a:%%b"
set "start_datetime=%start_date% %start_time%"
echo 实验开始时间: %start_datetime%
echo ========================================
echo.

set "current_exp=0"

for %%p in (%pred_lengths%) do (
    echo.
    echo [预测长度: %%p] 开始...
    echo ----------------------------------------
    
    :: Exchange Rate 实验
    set /a current_exp+=1
    echo [!current_exp!/%total_experiments%] 运行 Exchange Rate 实验...
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_start_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_start_time=%%a:%%b"
    echo 开始时间: !exp_start_date! !exp_start_time!
    
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
        --itr 1 --batch_size 8 --learning_rate 0.0005 >logs\LongForecasting\%model_name%_Exchange_%seq_len%_%%p.log ^
        --use_gpu True ^
        --gpu %GPU_ID%
    
    if !errorlevel! equ 0 (
        echo [成功] Exchange Rate 实验完成
    ) else (
        echo [失败] Exchange Rate 实验失败
    )
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_end_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_end_time=%%a:%%b"
    echo 结束时间: !exp_end_date! !exp_end_time!
    echo.

    :: Electricity 实验
    set /a current_exp+=1
    echo [!current_exp!/%total_experiments%] 运行 Electricity 实验...
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_start_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_start_time=%%a:%%b"
    echo 开始时间: !exp_start_date! !exp_start_time!
    
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
        --itr 1 --batch_size 16 --learning_rate 0.001 >logs\LongForecasting\%model_name%_Electricity_%seq_len%_%%p.log ^
        --use_gpu True ^
        --gpu %GPU_ID%
    
    if !errorlevel! equ 0 (
        echo [成功] Electricity 实验完成
    ) else (
        echo [失败] Electricity 实验失败
    )
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_end_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_end_time=%%a:%%b"
    echo 结束时间: !exp_end_date! !exp_end_time!
    echo.

    :: Traffic 实验
    set /a current_exp+=1
    echo [!current_exp!/%total_experiments%] 运行 Traffic 实验...
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_start_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_start_time=%%a:%%b"
    echo 开始时间: !exp_start_date! !exp_start_time!
    
    python -u run_longExp.py ^
        --is_training 1 ^
        --root_path .\dataset\ ^
        --data_path traffic.csv ^
        --model_id traffic_%seq_len%_%%p ^
        --model %model_name% ^
        --data custom ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 862 ^
        --des Exp ^
        --itr 1 --batch_size 16 --learning_rate 0.05 >logs\LongForecasting\%model_name%_Traffic_%seq_len%_%%p.log ^
        --use_gpu True ^
        --gpu %GPU_ID%
    
    if !errorlevel! equ 0 (
        echo [成功] Traffic 实验完成
    ) else (
        echo [失败] Traffic 实验失败
    )
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_end_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_end_time=%%a:%%b"
    echo 结束时间: !exp_end_date! !exp_end_time!
    echo.

    :: Weather 实验
    set /a current_exp+=1
    echo [!current_exp!/%total_experiments%] 运行 Weather 实验...
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_start_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_start_time=%%a:%%b"
    echo 开始时间: !exp_start_date! !exp_start_time!
    
    python -u run_longExp.py ^
        --is_training 1 ^
        --root_path .\dataset\ ^
        --data_path weather.csv ^
        --model_id weather_%seq_len%_%%p ^
        --model %model_name% ^
        --data custom ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 21 ^
        --des Exp ^
        --itr 1 --batch_size 16 >logs\LongForecasting\%model_name%_Weather_%seq_len%_%%p.log ^
        --use_gpu True ^
        --gpu %GPU_ID%
    
    if !errorlevel! equ 0 (
        echo [成功] Weather 实验完成
    ) else (
        echo [失败] Weather 实验失败
    )
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "exp_end_date=%%a %%b %%c"
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "exp_end_time=%%a:%%b"
    echo 结束时间: !exp_end_date! !exp_end_time!
    echo.
    
    echo [预测长度: %%p] 所有实验完成
    echo ----------------------------------------
)

:: 记录结束时间
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "end_date=%%a %%b %%c"
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "end_time=%%a:%%b"
set "end_datetime=%end_date% %end_time%"
echo.
echo ========================================
echo 所有实验完成!
echo 开始时间: %start_datetime%
echo 结束时间: %end_datetime%
echo 日志文件保存在: logs\LongForecasting\
echo ========================================

:: 显示生成的日志文件列表
echo.
echo 生成的日志文件:
dir /b logs\LongForecasting\%model_name%_*.log

echo.
pause 