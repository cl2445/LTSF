import os
import re
import pandas as pd

# 文件夹路径
folder_path = r"D:\时间序列\LTSF-Linear\logs\LongForecasting"
output_excel = os.path.join(folder_path, "TestLinear_Etth1_Results.xlsx")

# 预测步长
pred_steps = [96, 192, 336, 720]

# 存储数据
data = {"Pred": [], "MSE": [], "MAE": []}

# 遍历文件夹
for file in os.listdir(folder_path):
    match = re.match(r"TestLinear_Etth1_336_(\d+)\.log", file)
    if match:
        pred = int(match.group(1))  # 提取预测步长
        if pred in pred_steps:  # 只处理目标步长
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    mse_match = re.search(r"mse:\s*([\d.]+)", last_line)
                    mae_match = re.search(r"mae:\s*([\d.]+)", last_line)
                    if mse_match and mae_match:
                        data["Pred"].append(pred)
                        data["MSE"].append(float(mse_match.group(1)))
                        data["MAE"].append(float(mae_match.group(1)))

# 创建 DataFrame 并排序
df = pd.DataFrame(data).sort_values(by="Pred")

# 保存到 Excel
with pd.ExcelWriter(output_excel) as writer:
    df.to_excel(writer, sheet_name="FLinear", index=False)

print(f"Excel 文件已保存到: {output_excel}")

