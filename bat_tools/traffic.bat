@echo off

if not exist ".\logs" mkdir ".\logs"
if not exist ".\logs\LongForecasting" mkdir ".\logs\LongForecasting"

set seq_len=336
set model_name=Star
set GPU_ID=0

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path traffic.csv ^
  --model_id traffic_%seq_len%_192 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 192 ^
  --enc_in 862 ^
  --des Exp ^
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%model_name%_traffic_%seq_len%_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%
