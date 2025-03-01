import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 定义 moving_avg 和 series_decomp 类
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
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
data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

# 测试不同的 kernel_size
kernel_sizes = [5, 21, 51]
results = []

for kernel_size in kernel_sizes:
    decomp = series_decomp(kernel_size)
    residual, moving_mean = decomp(data_tensor)
    results.append((kernel_size, residual.squeeze().detach().numpy(), moving_mean.squeeze().detach().numpy()))

# 绘制结果
plt.figure(figsize=(18, 12))

plt.subplot(4, 1, 1)
plt.plot(t, data, label='Original Data')
plt.legend()

for i, (kernel_size, residual_np, moving_mean_np) in enumerate(results):
    plt.subplot(4, 1, i + 2)
    plt.plot(t, moving_mean_np, label=f'Moving Average (kernel_size={kernel_size})', color='orange')
    plt.plot(t, residual_np, label=f'Residual (kernel_size={kernel_size})', color='green')
    plt.legend()

plt.show()
