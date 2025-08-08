# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import sys

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def load_original_data(dataset_name):
    """加载原始数据集"""
    dataset_file = f"dataset/exchange_rate.csv"
    if os.path.exists(dataset_file):
        return pd.read_csv(dataset_file)
    else:
        print(f"未找到数据集文件: {dataset_file}")
        return None

def load_predictions(dataset_name, model_name, seq_len=336, pred_len=96):
    """加载预测结果"""
    result_pattern = f"results/{dataset_name}_{seq_len}_{pred_len}_{model_name}_*"
    result_folders = glob.glob(result_pattern)
    
    if not result_folders:
        print(f"未找到结果文件夹: {result_pattern}")
        return None
    
    result_folder = result_folders[0]
    pred_file = os.path.join(result_folder, "pred.npy")
    
    if os.path.exists(pred_file):
        return np.load(pred_file)
    else:
        print(f"未找到预测文件: {pred_file}")
        return None

def analyze_predictions(dataset_name, model_name, seq_len=336, pred_len=96, variable_idx=0):
    """
    详细分析预测结果
    
    Args:
        dataset_name: 数据集名称
        model_name: 模型名称
        seq_len: 序列长度
        pred_len: 预测长度
        variable_idx: 要分析的变量索引
    """
    print(f"分析 {dataset_name} 数据集 {model_name} 模型的预测结果")
    print(f"序列长度: {seq_len}, 预测长度: {pred_len}, 变量索引: {variable_idx}")
    print("=" * 80)
    
    # 加载原始数据
    df = load_original_data(dataset_name)
    if df is None:
        return
    
    # 加载预测结果
    predictions = load_predictions(dataset_name, model_name, seq_len, pred_len)
    if predictions is None:
        return
    
    print(f"原始数据形状: {df.shape}")
    print(f"预测结果形状: {predictions.shape}")
    print()
    
    # 获取原始数据中的目标变量
    if variable_idx < df.shape[1]:
        original_values = df.iloc[:, variable_idx].values
        print(f"原始数据变量 {variable_idx} 统计信息:")
        print(f"  最小值: {original_values.min():.6f}")
        print(f"  最大值: {original_values.max():.6f}")
        print(f"  平均值: {original_values.mean():.6f}")
        print(f"  标准差: {original_values.std():.6f}")
        print()
        
        # 显示原始数据的最后336个值（输入序列）
        print("原始数据的最后336个值（输入序列）:")
        last_336 = original_values[-seq_len:]
        for i in range(0, len(last_336), 50):  # 每50个值显示一行
            end_idx = min(i + 50, len(last_336))
            print(f"  索引 {i}-{end_idx-1}: {last_336[i:end_idx]}")
        print()
        
        # 显示预测结果
        if variable_idx < predictions.shape[1]:
            pred_values = predictions[:, variable_idx]
            print(f"预测结果变量 {variable_idx} 统计信息:")
            print(f"  最小值: {pred_values.min():.6f}")
            print(f"  最大值: {pred_values.max():.6f}")
            print(f"  平均值: {pred_values.mean():.6f}")
            print(f"  标准差: {pred_values.std():.6f}")
            print()
            
            print("预测结果的所有值:")
            for i in range(len(pred_values)):
                print(f"  时间步 {i+1}: {pred_values[i]:.6f}")
            print()
            
            # 显示完整的序列（原始数据最后336个 + 预测的96个）
            full_sequence = np.concatenate([last_336, pred_values])
            print(f"完整序列（原始336个 + 预测{pred_len}个）:")
            print(f"  总长度: {len(full_sequence)}")
            print(f"  完整序列统计: 最小值={full_sequence.min():.6f}, 最大值={full_sequence.max():.6f}, 平均值={full_sequence.mean():.6f}")
            print()
            
            return {
                'original_last_336': last_336,
                'predictions': pred_values,
                'full_sequence': full_sequence,
                'original_data': original_values
            }
        else:
            print(f"变量索引 {variable_idx} 超出预测结果范围")
    else:
        print(f"变量索引 {variable_idx} 超出原始数据范围")

def compare_multiple_predictions(dataset_name, model_names, seq_len=336, pred_len=96, variable_idx=0):
    """比较多个模型的预测结果"""
    print(f"比较 {dataset_name} 数据集多个模型的预测结果")
    print(f"序列长度: {seq_len}, 预测长度: {pred_len}, 变量索引: {variable_idx}")
    print("=" * 80)
    
    results = {}
    
    for model_name in model_names:
        print(f"\n模型: {model_name}")
        print("-" * 40)
        
        result = analyze_predictions(dataset_name, model_name, seq_len, pred_len, variable_idx)
        if result:
            results[model_name] = result
    
    # 比较不同模型的预测结果
    if len(results) > 1:
        print("\n模型比较:")
        print("-" * 40)
        for model_name, result in results.items():
            pred_values = result['predictions']
            print(f"{model_name}: 范围=[{pred_values.min():.6f}, {pred_values.max():.6f}], 均值={pred_values.mean():.6f}")

def save_detailed_results(dataset_name, model_name, seq_len=336, pred_len=96, variable_idx=0, output_file=None):
    """保存详细的分析结果到CSV文件"""
    if output_file is None:
        output_file = f"detailed_analysis_{dataset_name}_{model_name}_{seq_len}_{pred_len}_var{variable_idx}.csv"
    
    result = analyze_predictions(dataset_name, model_name, seq_len, pred_len, variable_idx)
    if result is None:
        return
    
    # 创建详细的数据框
    df_original = pd.DataFrame({
        'time_step': range(1, len(result['original_last_336']) + 1),
        'original_value': result['original_last_336'],
        'type': 'input'
    })
    
    df_predictions = pd.DataFrame({
        'time_step': range(len(result['original_last_336']) + 1, len(result['original_last_336']) + len(result['predictions']) + 1),
        'predicted_value': result['predictions'],
        'type': 'prediction'
    })
    
    # 合并数据
    df_combined = pd.concat([df_original, df_predictions], ignore_index=True)
    
    # 保存到CSV
    df_combined.to_csv(output_file, index=False)
    print(f"详细结果已保存到: {output_file}")
    
    return df_combined

def main():
    print("详细预测结果分析工具")
    print("=" * 50)
    
    # 示例分析
    print("1. 分析Exchange数据集的FLinear预测结果:")
    analyze_predictions("Exchange", "FLinear", 336, 96, 0)

    print("\n" + "=" * 80)

    print("2. 比较Exchange数据集不同模型:")
    compare_multiple_predictions("Exchange", ["FLinear", "DLinear"], 336, 96, 0)

    print("\n" + "=" * 80)

    print("3. 保存详细结果到CSV:")
    save_detailed_results("Exchange", "FLinear", 336, 96, 0)
    
    print("\n" + "=" * 80)
    
    print("4. 使用示例:")
    print("查看其他数据集和预测长度:")
    print("  analyze_predictions('traffic', 'FLinear', 336, 192, 0)")
    print("  analyze_predictions('Electricity', 'FLinear', 336, 336, 0)")
    print("  analyze_predictions('weather', 'FLinear', 336, 720, 0)")

if __name__ == "__main__":
    main() 