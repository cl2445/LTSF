@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo    Traffic 96预测长度 DLinear实验
echo ========================================
echo.

:: 创建日志目录
if not exist ".\logs" mkdir ".\logs"
if not exist ".\logs\LongForecasting" mkdir ".\logs\LongForecasting"

:: 设置实验参数
set seq_len=336
set model_name=FLinear
set pred_len=96

echo 实验配置:
echo - 模型: %model_name%
echo - 序列长度: %seq_len%
echo - 预测长度: %pred_len%
echo.

:: 记录开始时间
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "start_date=%%a %%b %%c"
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "start_time=%%a:%%b"
set "start_datetime=%start_date% %start_time%"
echo 实验开始时间: %start_datetime%
echo ========================================
echo.

echo 运行 Traffic DLinear 实验...
python -u run_longExp.py ^
    --is_training 1 ^
    --root_path .\dataset\ ^
    --data_path traffic.csv ^
    --model_id traffic_%seq_len%_%pred_len% ^
    --model %model_name% ^
    --data custom ^
    --features M ^
    --seq_len %seq_len% ^
    --pred_len %pred_len% ^
    --enc_in 862 ^
    --des Exp ^
    --itr 1 --batch_size 16 --learning_rate 0.05 --num_workers 0 >logs\LongForecasting\%model_name%_traffic_%seq_len%_%pred_len%.log

if !errorlevel! equ 0 (
    echo [成功] Traffic DLinear 实验完成
) else (
    echo [失败] Traffic DLinear 实验失败
)

:: 记录结束时间
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "end_date=%%a %%b %%c"
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "end_time=%%a:%%b"
set "end_datetime=%end_date% %end_time%"
echo.
echo ========================================
echo 实验完成!
echo 开始时间: %start_datetime%
echo 结束时间: %end_datetime%
echo 日志文件保存在: logs\LongForecasting\%model_name%_traffic_%seq_len%_%pred_len%.log
echo ========================================

echo.
pause 