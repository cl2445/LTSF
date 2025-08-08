@echo off

if not exist ".\logs" mkdir ".\logs"
if not exist ".\logs\LongForecasting" mkdir ".\logs\LongForecasting"

set seq_len=336
set model_name=FLinear
set GPU_ID=0

echo Starting ETTm1 experiments with %model_name% model...
echo.

REM ETTm1 experiments
python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_%seq_len%_96 ^
  --model %model_name% ^
  --data ETTm1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 8 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm1_%seq_len%_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_%seq_len%_192 ^
  --model %model_name% ^
  --data ETTm1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 192 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 8 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm1_%seq_len%_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_%seq_len%_336 ^
  --model %model_name% ^
  --data ETTm1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 336 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 8 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm1_%seq_len%_336.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_%seq_len%_720 ^
  --model %model_name% ^
  --data ETTm1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 720 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 8 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm1_%seq_len%_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

echo ETTm1 experiments completed.
echo.
echo Starting ETTm2 experiments...
echo.

REM ETTm2 experiments
python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_%seq_len%_96 ^
  --model %model_name% ^
  --data ETTm2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm2_%seq_len%_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_%seq_len%_192 ^
  --model %model_name% ^
  --data ETTm2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 192 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm2_%seq_len%_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_%seq_len%_336 ^
  --model %model_name% ^
  --data ETTm2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 336 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm2_%seq_len%_336.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_%seq_len%_720 ^
  --model %model_name% ^
  --data ETTm2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 720 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Ettm2_%seq_len%_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

echo ETTm2 experiments completed.
echo.
echo Starting ETTh1 experiments with %model_name% model...
echo.

REM ETTh1 experiments
python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_%seq_len%_96 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth1_%seq_len%_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_%seq_len%_192 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 192 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth1_%seq_len%_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_%seq_len%_336 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 336 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth1_%seq_len%_336.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_%seq_len%_720 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 720 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth1_%seq_len%_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

echo ETTh1 experiments completed.
echo.
echo Starting ETTh2 experiments with %model_name% model...
echo.

REM ETTh2 experiments
python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_%seq_len%_96 ^
  --model %model_name% ^
  --data ETTh2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 96 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth2_%seq_len%_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_%seq_len%_192 ^
  --model %model_name% ^
  --data ETTh2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 192 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth2_%seq_len%_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_%seq_len%_336 ^
  --model %model_name% ^
  --data ETTh2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 336 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth2_%seq_len%_336.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_%seq_len%_720 ^
  --model %model_name% ^
  --data ETTh2 ^
  --features M ^
  --seq_len %seq_len% ^
  --pred_len 720 ^
  --enc_in 7 ^
  --des Exp ^
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%model_name%_Etth2_%seq_len%_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

echo ETTh2 experiments completed.
echo.
echo All experiments completed successfully!
pause 