import torch
import torch.nn as nn
import torch.optim as optim

# 假设有一个简单的神经网络
class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# 示例输入和标签
input_size = 10
num_classes = 3
batch_size = 5

# 创建模型
model = SimpleModel(input_size, num_classes)

# 损失函数
criterion_ce = nn.CrossEntropyLoss() # 交叉熵损失
criterion_mse = nn.MSELoss()         # 均方误差
criterion_mae = nn.L1Loss()          # 平均绝对误差

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成随机输入数据和标签
inputs = torch.randn(batch_size, input_size)
labels = torch.randint(0, num_classes, (batch_size,))

# 前向传播
outputs = model(inputs)

# 计算交叉熵损失
loss_ce = criterion_ce(outputs, labels)

# 计算MSE和MAE损失，需要将标签转换为one-hot编码
labels_one_hot = nn.functional.one_hot(labels, num_classes=num_classes).float()
loss_mse = criterion_mse(outputs, labels_one_hot)
loss_mae = criterion_mae(outputs, labels_one_hot)

print('Cross Entropy Loss:', loss_ce.item())
print('Mean Squared Error:', loss_mse.item())
print('Mean Absolute Error:', loss_mae.item())

# 反向传播和优化
optimizer.zero_grad()
loss_ce.backward(retain_graph=True)
optimizer.step()

# 重置优化器
optimizer.zero_grad()
loss_ce.backward(retain_graph=True)
optimizer.step()

# 重置优化器
optimizer.zero_grad()
loss_mae.backward()
optimizer.step()
