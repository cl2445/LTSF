import os
import shutil

# 定义文件夹路径
source_folder = 'D:\时间序列\LTSF-Linear\logs\LongForecasting\Ett'
etth1_folder = os.path.join(source_folder, 'Etth1')
etth2_folder = os.path.join(source_folder, 'Etth2')
ettm1_folder = os.path.join(source_folder, 'Ettm1')
ettm2_folder = os.path.join(source_folder, 'Ettm2')

# 创建子文件夹，如果它们不存在
os.makedirs(etth1_folder, exist_ok=True)
os.makedirs(etth2_folder, exist_ok=True)
os.makedirs(ettm1_folder, exist_ok=True)
os.makedirs(ettm2_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    filepath = os.path.join(source_folder, filename)
    if os.path.isfile(filepath):  # 确保只处理文件
        if 'etth1' in filename.lower():
            shutil.move(filepath, os.path.join(etth1_folder, filename))
        elif 'etth2' in filename.lower():
            shutil.move(filepath, os.path.join(etth2_folder, filename))
        elif 'ettm1' in filename.lower():
            shutil.move(filepath, os.path.join(ettm1_folder, filename))
        elif 'ettm2' in filename.lower():
            shutil.move(filepath, os.path.join(ettm2_folder, filename))

print("文件分类完成。")
