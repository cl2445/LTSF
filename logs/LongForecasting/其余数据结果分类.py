import os
import shutil

# 定义文件夹路径
source_folder = 'D:\时间序列\LTSF-Linear\logs\LongForecasting'
weather_folder = os.path.join(source_folder, 'Weather')
exchange_folder = os.path.join(source_folder, 'Exchange')
electricity_folder = os.path.join(source_folder, 'Electricity')
traffic_folder = os.path.join(source_folder, 'Traffic')

# 创建子文件夹，如果它们不存在
os.makedirs(weather_folder, exist_ok=True)
os.makedirs(exchange_folder, exist_ok=True)
os.makedirs(electricity_folder, exist_ok=True)
os.makedirs(traffic_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    filepath = os.path.join(source_folder, filename)
    if os.path.isfile(filepath):  # 确保只处理文件
        if 'weather' in filename.lower():
            shutil.move(filepath, os.path.join(weather_folder, filename))
        elif 'exchange' in filename.lower():
            shutil.move(filepath, os.path.join(exchange_folder, filename))
        elif 'electricity' in filename.lower():
            shutil.move(filepath, os.path.join(electricity_folder, filename))
        elif 'traffic' in filename.lower():
            shutil.move(filepath, os.path.join(traffic_folder, filename))

print("文件分类完成。")
