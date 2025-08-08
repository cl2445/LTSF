@echo off
set GPU_ID=0
if not exist ".\logs" (
    mkdir .\logs
)

if not exist ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

set model_name=DLinear

:: ETTm1, univariate results, pred_len= 24 48 96 192 336 720



 call python -u run_longExp.py ^
   --is_training 1 ^
   --root_path .\dataset\ ^
   --data_path ETTm1.csv ^
   --model_id ETTm1_336_336 ^
   --model %model_name% ^
   --data ETTm1 ^
   --seq_len 336 ^
   --pred_len 336 ^
   --enc_in 1 ^
   --des "Exp" ^
   --itr 1 --batch_size 8 --feature S --learning_rate 0.0001 > logs\LongForecasting\%model_name%_fS_ETTm1_336_336.log ^
   --use_gpu True ^
   --gpu %GPU_ID%

  