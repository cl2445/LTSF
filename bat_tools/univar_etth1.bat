@echo off
set GPU_ID=0
if not exist ".\logs" (
    mkdir .\logs
)

if not exist ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

set model_name=DLinear

:: ETTh1, univariate results, pred_len= 96 192 336 720

 call python -u run_longExp.py ^
   --is_training 1 ^
   --root_path .\dataset\ ^
   --data_path ETTh1.csv ^
   --model_id ETTh1_336_96 ^
   --model %model_name% ^
   --data ETTh1 ^
   --seq_len 336 ^
   --pred_len 96 ^
   --enc_in 1 ^
   --des "Exp" ^
   --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh1_336_96.log ^
   --use_gpu True ^
   --gpu %GPU_ID%

 call python -u run_longExp.py ^
   --is_training 1 ^
   --root_path .\dataset\ ^
   --data_path ETTh1.csv ^
   --model_id ETTh1_336_192 ^
   --model %model_name% ^
   --data ETTh1 ^
   --seq_len 336 ^
   --pred_len 192 ^
   --enc_in 1 ^
   --des "Exp" ^
   --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh1_336_192.log ^
   --use_gpu True ^
   --gpu %GPU_ID%

 call python -u run_longExp.py ^
   --is_training 1 ^
   --root_path .\dataset\ ^
   --data_path ETTh1.csv ^
   --model_id ETTh1_336_336 ^
   --model %model_name% ^
   --data ETTh1 ^
   --seq_len 336 ^
   --pred_len 336 ^
   --enc_in 1 ^
   --des "Exp" ^
   --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh1_336_336.log ^
   --use_gpu True ^
   --gpu %GPU_ID%

 call python -u run_longExp.py ^
   --is_training 1 ^
   --root_path .\dataset\ ^
   --data_path ETTh1.csv ^
   --model_id ETTh1_336_720 ^
   --model %model_name% ^
   --data ETTh1 ^
   --seq_len 336 ^
   --pred_len 720 ^
   --enc_in 1 ^
   --des "Exp" ^
   --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh1_336_720.log ^
   --use_gpu True ^
   --gpu %GPU_ID%
  