import torch
import torch.nn as nn

class UnifiedLoss(nn.Module):
    def __init__(self, huber_weight=0.5, delta=1.0):
        super(UnifiedLoss, self).__init__()
        self.huber_weight = huber_weight
        self.delta = delta
        self.mae_loss = nn.L1Loss()

    def huber_loss(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.where(abs_error <= self.delta, 0.5 * error ** 2,
                                self.delta * abs_error - 0.5 * self.delta ** 2)
        return quadratic.mean()

    def forward(self, pred, true):
        huber_loss = self.huber_loss(pred, true)
        mae_loss = self.mae_loss(pred, true)
        combined_loss = self.huber_weight * huber_loss + (1 - self.huber_weight) * mae_loss
        return combined_loss


class MAEMSELoss(nn.Module):
    def __init__(self, mse_weight=0.5):
        super(MAEMSELoss, self).__init__()
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, pred, true):
        mse_loss = self.mse_loss(pred, true)
        mae_loss = self.mae_loss(pred, true)
        combined_loss = self.mse_weight * mse_loss + (1 - self.mse_weight) * mae_loss
        return combined_loss
# 示例使用
if __name__ == "__main__":
    # 模拟预测值和真实值
    y_pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y_true = torch.tensor([1.2, 2.3, 2.8])

    # 创建 UnifiedLoss 实例
    unified_loss_fn = MAEMSELoss()

    # 计算组合损失
    loss = unified_loss_fn(y_pred, y_true)

    # 打印损失值
    print(f"Combined Loss: {loss.item()}")

    # 反向传播
    loss.backward()
    print(f"Gradients of y_pred: {y_pred.grad}")

# Combined Loss: 0.0898333415389061
# Gradients of y_pred: tensor([-0.1467, -0.1700,  0.1467])

# Combined Loss: 0.14500001072883606
# Gradients of y_pred: tensor([-0.2333, -0.2667,  0.2333])