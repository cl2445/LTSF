# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose
#
# # 设置中文字体（如果需要）
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
#
# # 第一步：创建一个随机的时间序列并画一张图
# np.random.seed(0)  # 设置随机种子以保证结果的可重复性
# n_points = 104  # 将时间序列的点数增加到至少 104
# time_index = pd.date_range(start='2023-01-01', periods=n_points, freq='W')
# random_ts = pd.Series(np.random.randn(n_points).cumsum() + 50, index=time_index)
#
# plt.figure(figsize=(12, 6))
# plt.plot(random_ts, label='时间序列')
# plt.title('随机时间序列')
# plt.xlabel('时间')
# plt.ylabel('数值')
# plt.legend()
# # plt.grid(True)
# plt.show()
#
# # 第二步：获取时间序列的趋势项并画一张图 (不包含原始序列)
# # 使用 seasonal_decompose 进行分解，这里我们假设季节周期为一年（52周）
# decomposition = seasonal_decompose(random_ts, model='additive', period=52, extrapolate_trend='freq')
# trend = decomposition.trend
#
# plt.figure(figsize=(12, 6))
# plt.plot(trend, label='趋势项', color='red')
# plt.title('时间序列的趋势项')
# plt.xlabel('时间')
# plt.ylabel('数值')
# plt.legend()
# # plt.grid(True)
# plt.show()
#
# # 第三步：获取时间序列的季节项并画一张图
# seasonal = decomposition.seasonal
#
# plt.figure(figsize=(12, 6))
# plt.plot(seasonal, label='季节项', color='green')
# plt.title('时间序列的季节项')
# plt.xlabel('时间')
# plt.ylabel('季节性影响')
# plt.legend()
# # plt.grid(True)
# plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 第一步：创建一个原始的随机时间序列
np.random.seed(0)
n_original = 100
start_date = '2023-01-01'
original_time_index = pd.date_range(start=start_date, periods=n_original, freq='W')
original_ts = pd.Series(np.random.randn(n_original).cumsum() + 50, index=original_time_index)

# 第二步：创建一段“预测”结果
n_predicted = 50
start_predict_date = original_time_index[-1] + pd.Timedelta(weeks=1)
predicted_time_index = pd.date_range(start=start_predict_date, periods=n_predicted, freq='W')
predicted_values = np.random.randn(n_predicted).cumsum() + original_ts.iloc[-1]
predicted_ts = pd.Series(predicted_values, index=predicted_time_index)

# 第三步：创建一个包含原始数据和预测数据的 DataFrame
combined_df = pd.DataFrame({'value': pd.concat([original_ts, predicted_ts]),
                            'type': ['original'] * len(original_ts) + ['predicted'] * len(predicted_ts)})

# 第四步：绘制图形，按类型着色
plt.figure(figsize=(14, 7))
for type_name, group in combined_df.groupby('type'):
    color = 'blue' if type_name == 'original' else 'red'
    plt.plot(group.index, group['value'], label=type_name, color=color)

plt.title('原始随机序列与预测结果')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.grid(False)
plt.show()