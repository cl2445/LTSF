import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# 生成示例数据
t = np.linspace(0, 20, 500)
data = np.sin(t) + 0.5 * np.random.randn(500)

# 转换为 PyTorch 张量
data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

# 定义核大小
kernel_size = 21

# 创建分解模块实例
decomp = series_decomp(kernel_size)

# 应用分解模块
residual, moving_mean = decomp(data_tensor)

# 转换为 numpy 数组
residual_np = residual.squeeze().detach().numpy()
moving_mean_np = moving_mean.squeeze().detach().numpy()

# Calculate the mean of the residuals
residual_mean = np.mean(residual_np)
print(residual_mean)
plt.plot(t, residual_np)
# 绘制结果
plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.plot(t, data, label='Original Data')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, moving_mean_np, label='Moving Average', color='orange')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, residual_np, label='Residual', color='green')
plt.legend()

plt.show()
