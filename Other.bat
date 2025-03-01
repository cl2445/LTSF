@echo off

:: 创建日志目录
if not exist ".\logs" (
    mkdir .\logs
)

if not exist ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

:: 定义模型和预测长度数组
set model_names=DLinear_maxpool
set pred_lengths=96 192 336 720
set seq_len=336
set GPU_ID=0
:: 运行模型实验
for %%m in (%model_names%) do (
    for %%p in (%pred_lengths%) do (
        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path weather.csv ^
            --model_id weather_%seq_len%_%%p ^
            --model %%m ^
            --data custom ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 21 ^
            --des Exp ^
            --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%%m_Weather_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%

        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path exchange_rate.csv ^
            --model_id Exchange_%seq_len%_%%p ^
            --model %%m ^
            --data custom ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 8 ^
            --des Exp ^
            --itr 1 --batch_size 8 --learning_rate 0.005 >logs\LongForecasting\%%m_Exchange_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%

        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path electricity.csv ^
            --model_id Electricity_%seq_len%_%%p ^
            --model %%m ^
            --data custom ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 321 ^
            --des Exp ^
            --itr 1 --batch_size 16 --learning 0.005 >logs\LongForecasting\%%m_Electricity_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%


        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --model_id traffic_%seq_len%_%%p ^
            --model %%m ^
            --data custom ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --enc_in 862 ^
            --des Exp ^
            --itr 1 --batch_size 16 --learning_rate 0.005 >logs\LongForecasting\%%m_Traffic_%seq_len%_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%
    )
)

