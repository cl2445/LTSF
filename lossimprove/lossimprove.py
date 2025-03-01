import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride, padding_mode='replicate'):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        if self.padding_mode == 'constant':
            front.fill_(0)
            end.fill_(0)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        if self.stride > 1:
            x = nn.functional.interpolate(x, size=(x.size(2) * self.stride), mode='linear', align_corners=False)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size, stride=1, padding_mode='replicate'):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=stride, padding_mode=padding_mode)
        self.linear = nn.Linear(1, 1)  # Adding a simple linear layer

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        if moving_mean.size(1) != x.size(1):
            moving_mean = nn.functional.interpolate(moving_mean.permute(0, 2, 1), size=x.size(1), mode='linear',
                                                    align_corners=False).permute(0, 2, 1)
        res = x - moving_mean
        res = self.linear(res)  # Adding a simple linear transformation
        return res, moving_mean


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        mae_loss = self.mae(output, target)
        return mse_loss + 0.5 * mae_loss


# Generate sample data
t = np.linspace(0, 20, 500)
data = np.sin(t) + 0.5 * np.random.randn(500)
data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

# Creating a simple validation set (20% of data)
val_data_tensor = data_tensor[:, -100:, :]

model = SeriesDecomp(kernel_size=21, stride=1, padding_mode='replicate')
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    residual, moving_mean = model(data_tensor)
    # 使用 residual 计算损失，因为它包含了模型的可训练参数
    loss = loss_fn(residual, data_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 验证步骤
    model.eval()
    with torch.no_grad():
        residual_val, moving_mean_val = model(val_data_tensor)
        val_loss = loss_fn(residual_val, val_data_tensor)
        val_losses.append(val_loss.item())

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# 绘制损失曲线
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
