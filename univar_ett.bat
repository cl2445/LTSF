@echo off
set GPU_ID=0
if not exist ".\logs" (
    mkdir .\logs
)

if not exist ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

set model_name=DLinear

echo Starting univariate ETT experiments with %model_name% model...
echo.

echo Starting ETTh1 univariate experiments...
echo.

REM ETTh1 univariate experiments
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

echo ETTh1 univariate experiments completed.
echo.

echo Starting ETTh2 univariate experiments...
echo.

REM ETTh2 univariate experiments
call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_336_96 ^
  --model %model_name% ^
  --data ETTh2 ^
  --seq_len 336 ^
  --pred_len 96 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh2_336_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_336_192 ^
  --model %model_name% ^
  --data ETTh2 ^
  --seq_len 336 ^
  --pred_len 192 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh2_336_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_336_336 ^
  --model %model_name% ^
  --data ETTh2 ^
  --seq_len 336 ^
  --pred_len 336 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh2_336_336.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTh2.csv ^
  --model_id ETTh2_336_720 ^
  --model %model_name% ^
  --data ETTh2 ^
  --seq_len 336 ^
  --pred_len 720 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 > logs\LongForecasting\%model_name%_fS_ETTh2_336_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

echo ETTh2 univariate experiments completed.
echo.

echo Starting ETTm1 univariate experiments...
echo.

REM ETTm1 univariate experiments
call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_336_96 ^
  --model %model_name% ^
  --data ETTm1 ^
  --seq_len 336 ^
  --pred_len 96 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 8 --feature S --learning_rate 0.0001 > logs\LongForecasting\%model_name%_fS_ETTm1_336_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_336_192 ^
  --model %model_name% ^
  --data ETTm1 ^
  --seq_len 336 ^
  --pred_len 192 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 8 --feature S --learning_rate 0.0001 > logs\LongForecasting\%model_name%_fS_ETTm1_336_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

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

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_336_720 ^
  --model %model_name% ^
  --data ETTm1 ^
  --seq_len 336 ^
  --pred_len 720 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 8 --feature S --learning_rate 0.0001 > logs\LongForecasting\%model_name%_fS_ETTm1_336_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

echo ETTm1 univariate experiments completed.
echo.

echo Starting ETTm2 univariate experiments...
echo.

REM ETTm2 univariate experiments
call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_336_96 ^
  --model %model_name% ^
  --data ETTm2 ^
  --seq_len 336 ^
  --pred_len 96 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.001 > logs\LongForecasting\%model_name%_fS_ETTm2_336_96.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_336_192 ^
  --model %model_name% ^
  --data ETTm2 ^
  --seq_len 336 ^
  --pred_len 192 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.001 > logs\LongForecasting\%model_name%_fS_ETTm2_336_192.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_336_336 ^
  --model %model_name% ^
  --data ETTm2 ^
  --seq_len 336 ^
  --pred_len 336 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.001 > logs\LongForecasting\%model_name%_fS_ETTm2_336_336.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

call python -u run_longExp.py ^
  --is_training 1 ^
  --root_path .\dataset\ ^
  --data_path ETTm2.csv ^
  --model_id ETTm2_336_720 ^
  --model %model_name% ^
  --data ETTm2 ^
  --seq_len 336 ^
  --pred_len 720 ^
  --enc_in 1 ^
  --des "Exp" ^
  --itr 1 --batch_size 32 --feature S --learning_rate 0.001 > logs\LongForecasting\%model_name%_fS_ETTm2_336_720.log ^
  --use_gpu True ^
  --gpu %GPU_ID%

echo ETTm2 univariate experiments completed.
echo.
echo All univariate ETT experiments completed successfully!
pause 