import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


def calculate_mse(predictions, targets):
    """
    计算均方误差 (MSE)
    """
    return torch.mean((predictions - targets) ** 2)


def calculate_mae(predictions, targets):
    """
    计算平均绝对误差 (MAE)
    """
    return torch.mean(torch.abs(predictions - targets))


def evaluate_model(model, dataloader, criterion):
    """
    使用给定的数据和评估标准评估模型
    """
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# 示例配置类
class Configs:
    def __init__(self, seq_len, pred_len, enc_in, individual):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual


# 示例用法
if __name__ == "__main__":
    configs = Configs(seq_len=10, pred_len=5, enc_in=3, individual=True)

    # 创建模型实例
    model = Model(configs)

    # 示例输入和目标数据
    inputs = torch.randn(32, 10, 3)  # [Batch size, Input length, Channel]
    targets = torch.randn(32, 5, 3)  # [Batch size, Output length, Channel]

    # 示例数据加载器
    dataloader = [(inputs, targets)]

    # 使用均方误差 (MSE) 作为评估标准
    mse_loss = evaluate_model(model, dataloader, calculate_mse)
    print(f'MSE Loss: {mse_loss}')

    # 使用平均绝对误差 (MAE) 作为评估标准
    mae_loss = evaluate_model(model, dataloader, calculate_mae)
    print(f'MAE Loss: {mae_loss}')
