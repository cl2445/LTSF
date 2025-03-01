@echo off

:: 创建日志目录
if not exist ".\logs" (
    mkdir .\logs
)

if not exist ".\logs\LongForecasting" (
    mkdir .\logs\LongForecasting
)

:: 定义模型和预测长度数组
set model_names=Autoformer Informer Transformer
set pred_lengths=96 192 336 720
set GPU_ID=0
:: 运行模型实验
for %%m in (%model_names%) do (
    for %%p in (%pred_lengths%) do (

        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path .\dataset\ ^
            --data_path exchange_rate.csv ^
            --model_id exchange_96_%%p ^
            --model %%m ^
            --data custom ^
            --features M ^
            --seq_len 96 ^
            --label_len 48 ^
            --pred_len %%p ^
            --e_layers 2 ^
            --d_layers 1 ^
            --factor 3 ^
            --enc_in 8 ^
            --dec_in 8 ^
            --c_out 8 ^
            --des 'Exp' ^
            --itr 1 ^
            --train_epochs 1 > logs\LongForecasting\%%m_weather_%%p.log ^
            --use_gpu True ^
            --gpu %GPU_ID%



    )
)

