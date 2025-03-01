@echo off

:: 创建日志目录
if not exist ".\logs" (
    mkdir .\logs
)

if not exist ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

:: 定义模型和预测长度数组
set model_names=DLinear
set pred_lengths=96 192 336 720
set seq_len=336
set GPU_ID=0
:: 运行模型实验
for %%m in (%model_names%) do (
    for %%p in (%pred_lengths%) do (
        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path ETTh1.csv ^
            --model_id ETTh1_%seq_len%_%%p ^
            --model %%m ^
            --data ETTh1 ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 7 ^
            --des Exp ^
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%%m_ETTh1_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%

        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path ETTh2.csv ^
            --model_id ETTh2_%seq_len%_%%p ^
            --model %%m ^
            --data ETTh2 ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 7 ^
            --des Exp ^
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs\LongForecasting\%%m_ETTh2_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%

        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path ETTm1.csv ^
            --model_id ETTm1_%seq_len%_%%p ^
            --model %%m ^
            --data ETTm1 ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 7 ^
            --des Exp ^
            --itr 1 --batch_size 8 --learning 0.005 >logs\LongForecasting\%%m_ETTm1_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%


        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path ETTm2.csv ^
            --model_id ETTm2_%seq_len%_%%p ^
            --model %%m ^
            --data ETTm2 ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 7 ^
            --des Exp ^
            --itr 1 --batch_size 8 --learning_rate 0.005 >logs\LongForecasting\%%m_ETTm2_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%
    )
)

