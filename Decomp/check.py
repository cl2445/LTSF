import os
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# 指定文件路径
file_path = '../dataset/ETTh1.csv'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # 读取数据，设置date列为时间索引并解析为日期时间格式
    data = pd.read_csv(file_path, index_col='date', parse_dates=True)
    print(data.head())

    # 确保数据是按照时间顺序排序的
    data = data.sort_index()

    # 检查数据的时间间隔
    data.index = pd.to_datetime(data.index)
    print(data.index.to_series().diff().value_counts())

    # 检查缺失值
    print(data.isnull().sum())

    # 填补缺失值（如果有）
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # 选择一列进行季节性分解，假设是HUFL列
    column_to_decompose = 'HUFL'

    # 使用乘法模型进行季节性分解
    result_multiplicative = seasonal_decompose(data[column_to_decompose], model='multiplicative', period=168)
    result_multiplicative.plot()
    plt.show()

    # 使用加法模型进行季节性分解
    result_additive = seasonal_decompose(data[column_to_decompose], model='additive', period=168)
    result_additive.plot()
    plt.show()
