import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def custom_decomp(self, x):
        # Implement your custom decomposition method here
        # For example, a simple high-pass filter
        low_pass = self.moving_avg(x)
        high_pass = x - low_pass
        return high_pass, low_pass

    def forward(self, x):
        high_pass, low_pass = self.custom_decomp(x)
        return high_pass, low_pass

# Example usage
if __name__ == "__main__":
    model = series_decomp(kernel_size=3)
    x = torch.randn(1, 1, 10)  # Example input
    res, moving_mean = model(x)
    print("High pass component (residual):", res)
    print("Low pass component (moving mean):", moving_mean)
