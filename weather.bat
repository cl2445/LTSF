@echo off

if not exist ".\logs" mkdir ".\logs"
if not exist ".\logs\LongForecasting" mkdir ".\logs\LongForecasting"

set seq_len=336
set model_name=DLinear
set GPU_ID=0

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path weather.csv ^
  --model_id weather_%seq_len%_96 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --enc_in 21 ^
  --des Exp ^
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%model_name%_weather_%seq_len%_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path weather.csv ^
  --model_id weather_%seq_len%_192 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 192 ^
  --enc_in 21 ^
  --des Exp ^
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%model_name%_weather_%seq_len%_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path weather.csv ^
  --model_id weather_%seq_len%_336 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 336 ^
  --enc_in 21 ^
  --des Exp ^
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%model_name%_weather_%seq_len%_336.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path weather.csv ^
  --model_id weather_%seq_len%_720 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 720 ^
  --enc_in 21 ^
  --des Exp ^
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%model_name%_weather_%seq_len%_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%
