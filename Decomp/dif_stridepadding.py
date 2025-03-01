import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride, padding_mode='replicate'):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        if self.padding_mode == 'constant':
            front.fill_(0)
            end.fill_(0)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        # If stride is greater than 1, use interpolation to match the size
        if self.stride > 1:
            x = nn.functional.interpolate(x, size=(x.size(2) * self.stride), mode='linear', align_corners=False)

        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size, stride=1, padding_mode='replicate'):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride, padding_mode=padding_mode)

    def forward(self, x):
        moving_mean = self.moving_avg(x)

        # Ensure moving_mean and x have the same size
        if moving_mean.size(1) != x.size(1):
            moving_mean = nn.functional.interpolate(moving_mean.permute(0, 2, 1), size=x.size(1), mode='linear',
                                                    align_corners=False).permute(0, 2, 1)

        res = x - moving_mean
        return res, moving_mean


# Generate sample data
t = np.linspace(0, 20, 500)
data = np.sin(t) + 0.5 * np.random.randn(500)
data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

# Test different stride and padding modes
params = [(21, 1, 'replicate'), (21, 2, 'replicate'), (21, 1, 'constant')]
results = []

for kernel_size, stride, padding_mode in params:
    decomp = series_decomp(kernel_size, stride=stride, padding_mode=padding_mode)
    residual, moving_mean = decomp(data_tensor)
    residual_np = residual.squeeze().detach().numpy()
    moving_mean_np = moving_mean.squeeze().detach().numpy()

    if len(moving_mean_np) != len(data):
        moving_mean_np = np.interp(t, np.linspace(0, 20, len(moving_mean_np)), moving_mean_np)
        residual_np = data - moving_mean_np

    results.append((stride, padding_mode, residual_np, moving_mean_np))

# Plot results
plt.figure(figsize=(18, 12))

plt.subplot(4, 1, 1)
plt.plot(t, data, label='Original Data')
plt.legend()

for i, (stride, padding_mode, residual_np, moving_mean_np) in enumerate(results):
    plt.subplot(4, 1, i + 2)
    plt.plot(t, moving_mean_np, label=f'Moving Average (stride={stride}, padding={padding_mode})', color='orange')
    plt.plot(t, residual_np, label=f'Residual (stride={stride}, padding={padding_mode})', color='green')
    plt.legend()

plt.tight_layout()
plt.show()
